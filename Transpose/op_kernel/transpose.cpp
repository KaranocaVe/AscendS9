#include "kernel_operator.h"
#include "transpose_tiling_key.h"

#include <cstdint>

#define TRANSPOSE_MICRO_DENSE_MATRIX 1
#define TRANSPOSE_MICRO_STRIDED_MATRIX 2
#define TRANSPOSE_MICRO_RECORD_TRANS_DATA 3
#define TRANSPOSE_MICRO_RECORD_GATHER 4
#define TRANSPOSE_MICRO_STRIDED_SINGLE_MATRIX 5

namespace TransposeKernel {
using namespace AscendC;

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t TRANS_ADDR_COUNT = 16;
constexpr uint32_t COPY_BUFFER_BYTES = 64 * 1024;
constexpr uint32_t COPY_QUEUE_RESERVE_BYTES = 8 * BLOCK_BYTES;
constexpr uint32_t DATA_COPY_BLOCK_COUNT_MAX = 4095;
constexpr uint32_t DATA_COPY_BLOCK_LEN_MAX = (1U << 21) - 1;
constexpr uint64_t DATA_COPY_SHORT_PARAM_MAX = 65535;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

__aicore__ inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const uint64_t increment = alignment - remainder;
    const uint64_t maxValue = ~static_cast<uint64_t>(0);
    return value > maxValue - increment ? maxValue : value + increment;
}

__aicore__ inline uint64_t AlignDownU64(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

__aicore__ inline uint64_t GreatestCommonDivisorU64(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

__aicore__ inline bool IsPowerOfTwoU64(uint64_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

__aicore__ inline uint32_t GetPowerOfTwoShift(uint64_t value)
{
    uint32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

__aicore__ inline bool UsesDensePowerOfTwoColumnSchedule(uint64_t columnTileCount)
{
    // 列 tile 为 2 的幂时可用位分解；该性质由运行时 tile 规划决定，不绑定 batch 或公开 shape。
    return IsPowerOfTwoU64(columnTileCount);
}

template <typename T>
class KernelTranspose {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const TransposeTilingData* tilingData, TPipe* pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        // Gather-record 输入没有跨任务复用，绕过 L2 避免流式读挤占缓存。
        if (tilingData->microKernel == TRANSPOSE_MICRO_RECORD_GATHER) {
            inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_->blockDim;
    }

    // 固定轴 Strided 独立 key 不需要 RecordGather 的缓存策略判断；保留相同的
    // GM、TPipe 和任务元数据初始化，同时隔离其长 MTE2 热路径的代码布局。
    __aicore__ inline void InitStridedFixed(GM_ADDR input, GM_ADDR output,
                                             const TransposeTilingData* tilingData, TPipe* pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_->blockDim;
    }

    __aicore__ inline void ProcessCopy()
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t totalBlocks = CeilDivU64(tiling_->totalElements, blockElements);
        const uint64_t baseBlocks = totalBlocks / blockDim_;
        const uint64_t extraBlocks = totalBlocks % blockDim_;
        const uint64_t localBlocks = baseBlocks + (blockIdx_ < extraBlocks ? 1 : 0);
        if (localBlocks == 0) {
            return;
        }
        const uint64_t startBlock = blockIdx_ * baseBlocks + MinU64(blockIdx_, extraBlocks);
        const uint64_t startElement = startBlock * blockElements;
        const uint64_t localElements = MinU64(localBlocks * blockElements, tiling_->totalElements - startElement);

        InitCopyQueues();
        CopyContiguous(startElement, startElement, localElements);
    }

    __aicore__ inline void ProcessBlockSwap()
    {
        if (tiling_->microKernel == TRANSPOSE_MICRO_DENSE_MATRIX) {
            ProcessDenseMatrix();
        } else if (tiling_->microKernel == TRANSPOSE_MICRO_STRIDED_MATRIX) {
            ProcessStridedMatrix();
        } else if (tiling_->microKernel == TRANSPOSE_MICRO_STRIDED_SINGLE_MATRIX) {
            ProcessStridedSingleMatrix();
        } else if (tiling_->microKernel == TRANSPOSE_MICRO_RECORD_TRANS_DATA) {
            ProcessRecordBlockSwap();
        } else if (tiling_->microKernel == TRANSPOSE_MICRO_RECORD_GATHER) {
            ProcessGatherRecordBlockSwap();
        }
    }

    __aicore__ inline void ProcessDenseMatrix()
    {
        // B8 Dense 路径包含四路 TransData 控制分派和行块均衡；DAV_2201 预取后续
        // 指令缓存线仅影响该通用 int8 微内核，不改变 B16/B32 与任何数据搬运契约。
        if constexpr (sizeof(T) == 1) {
            ICachePreLoad(2);
        }
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t matrixCols = tiling_->inner;
        const uint64_t rowTileCount = CeilDivU64(matrixRows, tiling_->tileRows);
        const uint64_t colTileCount = CeilDivU64(matrixCols, tiling_->tileCols);
        const uint64_t rowAlignUnit = sizeof(T) == 1 ? 32 : 16;
        const uint64_t colAlignUnit = BLOCK_BYTES / sizeof(T);
        const uint64_t maxRowsAlign = AlignUpU64(tiling_->tileRows, rowAlignUnit);
        const uint64_t maxColsAlign = AlignUpU64(tiling_->tileCols, colAlignUnit);
        const uint64_t bufferBytes = maxRowsAlign * maxColsAlign * sizeof(T);
        pipe_->InitBuffer(inputQueue_, 1, bufferBytes);
        pipe_->InitBuffer(outputQueue_, tiling_->bufferNum, bufferBytes);
        const bool powerOfTwoColumnSchedule = UsesDensePowerOfTwoColumnSchedule(colTileCount);
        const uint32_t colTileShift = powerOfTwoColumnSchedule ? GetPowerOfTwoShift(colTileCount) : 0;
        const uint64_t colTileMask = powerOfTwoColumnSchedule ? colTileCount - 1 : 0;

        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            uint64_t batch = 0;
            uint64_t rowTileIdx = 0;
            uint64_t colTileIdx = 0;
            if (powerOfTwoColumnSchedule) {
                if (tiling_->batch == 1) {
                    // 单 batch 保留无除法/取模的热路径，适用于任意 2 的幂列 tile 数。
                    rowTileIdx = task >> colTileShift;
                    colTileIdx = task & colTileMask;
                } else {
                    const uint64_t tasksPerBatch = rowTileCount * colTileCount;
                    batch = task / tasksPerBatch;
                    const uint64_t taskInBatch = task % tasksPerBatch;
                    rowTileIdx = taskInBatch >> colTileShift;
                    const uint64_t rawColTileIdx = taskInBatch & colTileMask;
                    colTileIdx = (rawColTileIdx + batch) & colTileMask;
                }
            } else {
                const uint64_t tasksPerBatch = rowTileCount * colTileCount;
                batch = task / tasksPerBatch;
                const uint64_t taskInBatch = task % tasksPerBatch;
                rowTileIdx = taskInBatch / colTileCount;
                const uint64_t rawColTileIdx = taskInBatch % colTileCount;
                colTileIdx = (rawColTileIdx + batch) % colTileCount;
            }

            uint64_t rowStart = rowTileIdx * tiling_->tileRows;
            uint64_t rowsReal = MinU64(tiling_->tileRows, matrixRows - rowStart);
            uint64_t rowsAlign = AlignUpU64(rowsReal, rowAlignUnit);
            if constexpr (sizeof(T) == 1) {
                // int8 按 32 行组做商余均衡，避免最后一核形成长尾。
                const uint64_t totalRowBlocks = CeilDivU64(matrixRows, rowAlignUnit);
                const uint64_t baseRowBlocks = totalRowBlocks / rowTileCount;
                const uint64_t extraRowBlocks = totalRowBlocks % rowTileCount;
                const uint64_t localRowBlocks = baseRowBlocks + (rowTileIdx < extraRowBlocks ? 1 : 0);
                const uint64_t startRowBlock = rowTileIdx * baseRowBlocks + MinU64(rowTileIdx, extraRowBlocks);
                rowStart = startRowBlock * rowAlignUnit;
                rowsAlign = localRowBlocks * rowAlignUnit;
                rowsReal = MinU64(rowsAlign, matrixRows - rowStart);
            }

            const uint64_t colStart = colTileIdx * tiling_->tileCols;
            const uint64_t colsReal = MinU64(tiling_->tileCols, matrixCols - colStart);
            const uint64_t colsAlign = AlignUpU64(colsReal, colAlignUnit);
            CopyInDenseMatrix(batch, rowStart, colStart, rowsReal, colsReal, matrixRows, matrixCols);
            ComputeMatrix(rowsAlign, colsAlign);
            CopyOutDenseMatrix(batch, rowStart, colStart, rowsReal, rowsAlign, colsReal, matrixRows, matrixCols);
        }
    }

    __aicore__ inline void ProcessStridedMatrix()
    {
        // DAV_2201 支持在较长的非退化 Strided 控制流入口预取后续指令；固定轴
        // 退化路径不使用该预取，避免给 C5 的纯 MTE2 主路径增加无关控制指令。
        if (tiling_->rows > 1) {
            ICachePreLoad(3);
        }
        // 固定轴是否退化在一次 kernel 调用内不变。将分派移到热循环外，让固定轴路径
        // 的二维 MTE2/SliceInfo 发射不再携带多自然段回退的运行时控制流。
        if (tiling_->rows == 1) {
            ProcessStridedMatrixImpl<true>();
        } else {
            ProcessStridedMatrixImpl<false>();
        }
    }

    template <bool FIXED_AXIS_ONE>
    __aicore__ inline void ProcessStridedMatrixImpl()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t chunkRows = tiling_->tileRows;
        const uint64_t panelRows = tiling_->panelRows;
        const uint64_t matrixColsTile = tiling_->tileCols;
        const uint64_t matrixColsTileAlign = AlignUpU64(matrixColsTile, BLOCK_BYTES / sizeof(T));
        pipe_->InitBuffer(inputQueue_, tiling_->bufferNum,
                          chunkRows * matrixColsTileAlign * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 1, panelRows * matrixColsTileAlign * sizeof(T));

        const uint64_t colTileCount = CeilDivU64(tiling_->inner, matrixColsTile);
        const uint64_t colTilesPerTask = tiling_->colTilesPerTask == 0 ? 1 :
            MinU64(tiling_->colTilesPerTask, colTileCount);
        const uint64_t colTaskCount = CeilDivU64(colTileCount, colTilesPerTask);
        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            const uint64_t rawColTaskIndex = task % colTaskCount;
            const uint64_t matrixTask = task / colTaskCount;
            const uint64_t rawColTileStart = rawColTaskIndex * colTilesPerTask;
            const uint64_t rawColTileEnd = MinU64(rawColTileStart + colTilesPerTask, colTileCount);
            const uint64_t fixedIndex = matrixTask % tiling_->rows;
            const uint64_t batch = matrixTask / tiling_->rows;
            for (uint64_t rawColTile = rawColTileStart; rawColTile < rawColTileEnd; ++rawColTile) {
                // 不同矩阵轮转列 tile，避免列数与核数存在公因子时尾列长期集中到部分核。
                const uint64_t colTileIndex = (rawColTile + matrixTask) % colTileCount;
                const uint64_t colStart = colTileIndex * matrixColsTile;
                const uint64_t colsReal = MinU64(matrixColsTile, tiling_->inner - colStart);
                const uint64_t colsAlign = AlignUpU64(colsReal, BLOCK_BYTES / sizeof(T));
                for (uint64_t panelStart = 0; panelStart < matrixRows; panelStart += panelRows) {
                    const uint64_t panelRowsReal = MinU64(panelRows, matrixRows - panelStart);
                    const uint64_t panelRowsAlign = AlignUpU64(panelRowsReal, TRANS_ADDR_COUNT);
                    const uint64_t firstRowsReal = MinU64(chunkRows, panelRowsReal);
                    // 先发射首块搬入，使其与上一输出 panel 尚未完成的写回重叠。
                    CopyInStridedMatrix<FIXED_AXIS_ONE>(batch, fixedIndex, panelStart, firstRowsReal,
                                                        colStart, colsReal, colsAlign);
                    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
                    ComputeStridedMatrix<false>(outputLocal, 0, firstRowsReal,
                                                panelRowsAlign, colsAlign);
                    for (uint64_t panelOffset = chunkRows; panelOffset < panelRowsReal;
                         panelOffset += chunkRows) {
                        const uint64_t rowsReal = MinU64(chunkRows, panelRowsReal - panelOffset);
                        CopyInStridedMatrix<FIXED_AXIS_ONE>(batch, fixedIndex,
                                                            panelStart + panelOffset, rowsReal,
                                                            colStart, colsReal, colsAlign);
                        ComputeStridedMatrix<false>(outputLocal, panelOffset, rowsReal,
                                                    panelRowsAlign, colsAlign);
                    }
                    outputQueue_.EnQue(outputLocal);
                    CopyOutStridedMatrix(batch, fixedIndex, panelStart, panelRowsReal,
                                         colStart, colsReal, matrixRows, panelRowsAlign);
                }
            }
        }
    }

    __aicore__ inline void ProcessStridedSingleMatrix()
    {
        // Host 仅在单矩阵、单 fixed 轴、每任务一个列 tile 且任务数不超过核数时选择本路径。
        // 因而 blockIdx 直接就是列 tile 索引，省去通用调度中的矩阵任务商余分解。
        if (blockIdx_ >= tiling_->taskCount) {
            return;
        }
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t chunkRows = tiling_->tileRows;
        const uint64_t panelRows = tiling_->panelRows;
        const uint64_t matrixColsTile = tiling_->tileCols;
        const uint64_t matrixColsTileAlign = AlignUpU64(matrixColsTile, BLOCK_BYTES / sizeof(T));
        pipe_->InitBuffer(inputQueue_, tiling_->bufferNum,
                          chunkRows * matrixColsTileAlign * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 1, panelRows * matrixColsTileAlign * sizeof(T));

        const uint64_t colStart = blockIdx_ * matrixColsTile;
        const uint64_t colsReal = MinU64(matrixColsTile, tiling_->inner - colStart);
        const uint64_t colsAlign = AlignUpU64(colsReal, BLOCK_BYTES / sizeof(T));
        for (uint64_t panelStart = 0; panelStart < matrixRows; panelStart += panelRows) {
            const uint64_t panelRowsReal = MinU64(panelRows, matrixRows - panelStart);
            const uint64_t panelRowsAlign = AlignUpU64(panelRowsReal, TRANS_ADDR_COUNT);
            const uint64_t firstRowsReal = MinU64(chunkRows, panelRowsReal);
            CopyInStridedMatrix<true>(0, 0, panelStart, firstRowsReal, colStart, colsReal, colsAlign);
            LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
            if (firstRowsReal % TRANS_ADDR_COUNT == 0) {
                ComputeStridedMatrix<true>(outputLocal, 0, firstRowsReal, panelRowsAlign, colsAlign);
            } else {
                ComputeStridedMatrix<false>(outputLocal, 0, firstRowsReal, panelRowsAlign, colsAlign);
            }
            for (uint64_t panelOffset = chunkRows; panelOffset < panelRowsReal;
                 panelOffset += chunkRows) {
                const uint64_t rowsReal = MinU64(chunkRows, panelRowsReal - panelOffset);
                CopyInStridedMatrix<true>(0, 0, panelStart + panelOffset, rowsReal,
                                          colStart, colsReal, colsAlign);
                if (rowsReal % TRANS_ADDR_COUNT == 0) {
                    ComputeStridedMatrix<true>(outputLocal, panelOffset, rowsReal, panelRowsAlign, colsAlign);
                } else {
                    ComputeStridedMatrix<false>(outputLocal, panelOffset, rowsReal, panelRowsAlign, colsAlign);
                }
            }
            outputQueue_.EnQue(outputLocal);
            CopyOutStridedMatrix(0, 0, panelStart, panelRowsReal, colStart, colsReal,
                                 matrixRows, panelRowsAlign);
        }
    }

    __aicore__ inline void ProcessRecordBlockSwap()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t rowTile = tiling_->tileRows;
        const uint64_t colTile = tiling_->tileCols;
        const uint64_t recordAlign = AlignUpU64(tiling_->record, BLOCK_BYTES / sizeof(T));
        const bool alignedRecord = recordAlign == tiling_->record;
        const uint64_t bufferBytes = colTile * rowTile * recordAlign * sizeof(T);
        pipe_->InitBuffer(inputQueue_, 1, bufferBytes);
        if (!alignedRecord) {
            pipe_->InitBuffer(outputQueue_, 1, bufferBytes);
        }

        const uint64_t rowTileCount = CeilDivU64(matrixRows, rowTile);
        const uint64_t colTileCount = CeilDivU64(tiling_->inner, colTile);
        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            const uint64_t fixedIndex = task % tiling_->rows;
            const uint64_t matrixTask = task / tiling_->rows;
            const uint64_t colTileIndex = matrixTask % colTileCount;
            const uint64_t rowTileIndex = matrixTask / colTileCount % rowTileCount;
            const uint64_t batch = matrixTask / colTileCount / rowTileCount;
            const uint64_t rowStart = rowTileIndex * rowTile;
            const uint64_t colStart = colTileIndex * colTile;
            const uint64_t rowsReal = MinU64(rowTile, matrixRows - rowStart);
            const uint64_t colsReal = MinU64(colTile, tiling_->inner - colStart);
            const uint64_t colsAlign = AlignUpU64(colsReal, TRANS_ADDR_COUNT);
            CopyInRecordBlock(batch, fixedIndex, rowStart, colStart,
                              rowsReal, colsReal, recordAlign);
            if (alignedRecord) {
                CopyOutAlignedRecord(batch, fixedIndex, rowStart, colStart, rowsReal, colsReal, recordAlign);
            } else {
                PackPaddedRecords(colsReal, colsAlign, rowsReal, recordAlign);
                CopyOutPackedRecord(batch, fixedIndex, rowStart, colStart,
                                    rowsReal, colsReal, recordAlign);
            }
        }
    }

    __aicore__ inline void ProcessGatherRecordBlockSwap()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t rowTile = tiling_->tileRows;
        const uint64_t colTile = tiling_->tileCols;
        const uint64_t panelCols = tiling_->panelCols;
        const uint64_t recordBytes = tiling_->record * sizeof(T);
        const uint64_t inputPitchBytes = AlignUpU64(panelCols * recordBytes, BLOCK_BYTES);
        constexpr uint64_t gatherRepeatElements = 8 * BLOCK_BYTES / sizeof(T);
        const uint64_t gatherCount = rowTile * tiling_->record;
        const uint64_t storedGatherCount =
            gatherCount > gatherRepeatElements && gatherCount % gatherRepeatElements != 0 ?
                AlignUpU64(gatherCount, gatherRepeatElements) : gatherCount;
        const uint64_t outputPitchBytes =
            AlignUpU64(storedGatherCount * sizeof(T), BLOCK_BYTES);
        const uint64_t inputBufferBytes = rowTile * inputPitchBytes;
        const uint64_t outputBufferBytes = colTile * outputPitchBytes;
        const uint64_t offsetBytes =
            AlignUpU64(storedGatherCount * sizeof(uint32_t), BLOCK_BYTES);
        pipe_->InitBuffer(gatherInputQueue_, tiling_->bufferNum, inputBufferBytes);
        pipe_->InitBuffer(outputQueue_, 1, outputBufferBytes);
        pipe_->InitBuffer(gatherOffsetBuffer_, offsetBytes);
        InitGatherOffsets(inputPitchBytes);

        const uint64_t rowTileCount = CeilDivU64(matrixRows, rowTile);
        const uint64_t colTileCount = CeilDivU64(tiling_->inner, colTile);
        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            const uint64_t fixedIndex = task % tiling_->rows;
            const uint64_t matrixTask = task / tiling_->rows;
            const uint64_t colTileIndex = matrixTask % colTileCount;
            const uint64_t rowTileIndex = matrixTask / colTileCount % rowTileCount;
            const uint64_t batch = matrixTask / colTileCount / rowTileCount;
            const uint64_t rowStart = rowTileIndex * rowTile;
            const uint64_t colStart = colTileIndex * colTile;
            const uint64_t rowsReal = MinU64(rowTile, matrixRows - rowStart);
            const uint64_t colsReal = MinU64(colTile, tiling_->inner - colStart);
            const uint64_t panelCount = CeilDivU64(colsReal, panelCols);
            // 两个输入面板只差一列时，先提交较窄面板会改变第二条二维 MTE2 的 GM
            // 起点相位；总读取字节、Gather、输出面板和双缓冲深度均保持不变。
            // 条件只依赖本次 tile 的面板拓扑，普通多面板与等宽面板保持既有顺序。
            const bool smallerFirst = panelCols > 1 && panelCount == 2 &&
                colsReal == panelCols * 2 - 1;
            const uint64_t firstPanelCols = smallerFirst ? panelCols - 1 : panelCols;
            const uint64_t preloadCount = MinU64(tiling_->bufferNum, panelCount);
            for (uint64_t panel = 0; panel < preloadCount; ++panel) {
                const uint64_t panelStart = smallerFirst && panel != 0 ? firstPanelCols :
                    panel * panelCols;
                const uint64_t panelReal = smallerFirst && panel == 0 ? firstPanelCols :
                    MinU64(panelCols, colsReal - panelStart);
                CopyInGatherRecordBlock(batch, fixedIndex, rowStart, colStart + panelStart,
                                        rowsReal, panelReal, inputPitchBytes);
            }
            LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
            for (uint64_t panel = 0; panel < panelCount; ++panel) {
                const uint64_t panelStart = smallerFirst && panel != 0 ? firstPanelCols :
                    panel * panelCols;
                const uint64_t panelReal = smallerFirst && panel == 0 ? firstPanelCols :
                    MinU64(panelCols, colsReal - panelStart);
                GatherRecordPanel(outputLocal, panelStart, rowsReal, panelReal, outputPitchBytes);
                const uint64_t nextPanel = panel + tiling_->bufferNum;
                if (nextPanel < panelCount) {
                    const uint64_t nextStart = smallerFirst && nextPanel != 0 ? firstPanelCols :
                        nextPanel * panelCols;
                    const uint64_t nextReal = smallerFirst && nextPanel == 0 ? firstPanelCols :
                        MinU64(panelCols, colsReal - nextStart);
                    CopyInGatherRecordBlock(batch, fixedIndex, rowStart, colStart + nextStart,
                                            rowsReal, nextReal, inputPitchBytes);
                }
            }
            outputQueue_.EnQue(outputLocal);
            CopyOutGatherRecordBlock(batch, fixedIndex, rowStart, colStart,
                                     rowsReal, colsReal, outputPitchBytes);
        }
    }

    __aicore__ inline void ProcessGenericAffine()
    {
        uint32_t suffixStart = tiling_->rank;
        uint32_t nextInputAxis = tiling_->rank;
        uint64_t recordElements = 1;
        while (suffixStart > 0) {
            const uint32_t outputAxis = suffixStart - 1;
            const uint32_t inputAxis = tiling_->perm[outputAxis];
            if (nextInputAxis == 0 || inputAxis + 1 != nextInputAxis) {
                break;
            }
            recordElements *= tiling_->shape[inputAxis];
            nextInputAxis = inputAxis;
            suffixStart--;
        }
        const uint64_t recordCount = tiling_->totalElements / recordElements;
        InitCopyQueues();
        for (uint64_t record = blockIdx_; record < recordCount; record += blockDim_) {
            const uint64_t outputOffset = record * recordElements;
            uint64_t inputOffset = 0;
            for (uint32_t axis = 0; axis < suffixStart; ++axis) {
                const uint64_t coordinate = (outputOffset / tiling_->outputStride[axis]) %
                                            tiling_->shape[tiling_->perm[axis]];
                inputOffset += coordinate * tiling_->inputStride[tiling_->perm[axis]];
            }
            CopyContiguous(inputOffset, outputOffset, recordElements);
        }
    }

private:
    template <HardEvent EVENT>
    __aicore__ inline void SyncEvent()
    {
        const event_t eventId = static_cast<event_t>(pipe_->FetchEventID(EVENT));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }

    __aicore__ inline void InitCopyQueues()
    {
        const uint64_t availablePerBuffer = tiling_->ubSize > COPY_QUEUE_RESERVE_BYTES ?
                                                (tiling_->ubSize - COPY_QUEUE_RESERVE_BYTES) / 2 : BLOCK_BYTES;
        copyBufferBytes_ = MinU64(COPY_BUFFER_BYTES, availablePerBuffer);
        copyBufferBytes_ = copyBufferBytes_ / BLOCK_BYTES * BLOCK_BYTES;
        if (copyBufferBytes_ == 0) {
            copyBufferBytes_ = BLOCK_BYTES;
        }
        pipe_->InitBuffer(inputQueue_, 1, copyBufferBytes_);
        pipe_->InitBuffer(outputQueue_, 1, copyBufferBytes_);
    }

    __aicore__ inline void CopyContiguous(uint64_t inputOffset, uint64_t outputOffset, uint64_t elements)
    {
        const uint64_t tileElements = copyBufferBytes_ / sizeof(T);
        for (uint64_t done = 0; done < elements; done += tileElements) {
            const uint64_t current = MinU64(tileElements, elements - done);
            LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
            DataCopyExtParams inputParams{1, static_cast<uint32_t>(current * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm_[inputOffset + done], inputParams, padParams);
            inputQueue_.EnQue(inputLocal);

            inputLocal = inputQueue_.DeQue<T>();
            LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
            DataCopyParams localParams{1, static_cast<uint16_t>(AlignUpU64(current * sizeof(T), BLOCK_BYTES) /
                                                                 BLOCK_BYTES), 0, 0};
            DataCopy(outputLocal, inputLocal, localParams);
            outputQueue_.EnQue(outputLocal);
            inputQueue_.FreeTensor(inputLocal);

            outputLocal = outputQueue_.DeQue<T>();
            DataCopyExtParams outputParams{1, static_cast<uint32_t>(current * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset + done], outputLocal, outputParams);
            outputQueue_.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void CopyInDenseMatrix(uint64_t batch, uint64_t rowStart, uint64_t colStart,
                                             uint64_t rowsReal, uint64_t colsReal,
                                             uint64_t matrixRows, uint64_t matrixCols)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        const uint64_t inputOffset = batch * matrixRows * matrixCols + rowStart * matrixCols + colStart;
        const uint64_t blockLenBytes = colsReal * sizeof(T);
        const uint64_t sourceStrideBytes = (matrixCols - colsReal) * sizeof(T);
        if (rowsReal <= DATA_COPY_BLOCK_COUNT_MAX &&
            blockLenBytes <= DATA_COPY_SHORT_PARAM_MAX &&
            sourceStrideBytes <= DATA_COPY_SHORT_PARAM_MAX) {
            // 字段均可由短参数表达时减少热路径描述符开销；超限形状继续走 Ext 泛化路径。
            DataCopyParams params{static_cast<uint16_t>(rowsReal),
                                  static_cast<uint16_t>(blockLenBytes),
                                  static_cast<uint16_t>(sourceStrideBytes), 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
            inputQueue_.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(rowsReal), static_cast<uint32_t>(colsReal * sizeof(T)),
                                 static_cast<uint32_t>((matrixCols - colsReal) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void ComputeMatrix(uint64_t rowsAlign, uint64_t colsAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        if constexpr (sizeof(T) == 1) {
            ComputeMatrixB8(inputLocal, outputLocal, rowsAlign, colsAlign);
        } else {
            ComputeMatrixB16B32(inputLocal, outputLocal, rowsAlign, colsAlign);
        }
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ComputeMatrixB16B32(LocalTensor<T>& inputLocal, LocalTensor<T>& outputLocal,
                                                uint64_t rowsAlign, uint64_t colsAlign)
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(colsAlign / blockElements);
        const uint16_t dstRepStride = repeatTimes <= 1 ? 0 : static_cast<uint16_t>(rowsAlign);
        const uint16_t srcRepStride = repeatTimes <= 1 ? 0 : 1;
        TransDataTo5HDParams params{false, false, repeatTimes, dstRepStride, srcRepStride};

        for (uint64_t rowBase = 0; rowBase < rowsAlign; rowBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
                srcList[row] = reinterpret_cast<uint64_t>(
                    inputLocal[(rowBase + row) * colsAlign].GetPhyAddr());
            }
            uint64_t dstList[TRANS_ADDR_COUNT];
            if constexpr (sizeof(T) == 4) {
                for (uint64_t col = 0; col < blockElements; ++col) {
                    dstList[col * 2] = reinterpret_cast<uint64_t>(
                        outputLocal[col * rowsAlign + rowBase].GetPhyAddr());
                    dstList[col * 2 + 1] = reinterpret_cast<uint64_t>(
                        outputLocal[col * rowsAlign + rowBase + blockElements].GetPhyAddr());
                }
            } else {
                for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                    dstList[col] = reinterpret_cast<uint64_t>(
                        outputLocal[col * rowsAlign + rowBase].GetPhyAddr());
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void ComputeMatrixB8(LocalTensor<T>& inputLocal, LocalTensor<T>& outputLocal,
                                            uint64_t rowsAlign, uint64_t colsAlign)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(rowsAlign / 32);
        const uint16_t dstRepStride = repeatTimes <= 1 ? 0 : 1;
        const uint16_t srcRepStride = repeatTimes <= 1 ? 0 : static_cast<uint16_t>(colsAlign);
        // 四种 B8 转置只改变高低半区的标志，重复次数和步幅在列组间恒定。
        // 将不变参数移出热循环，避免每个列组重复构造控制描述符。
        const TransDataTo5HDParams lowLow{false, false, repeatTimes, dstRepStride, srcRepStride};
        const TransDataTo5HDParams highLow{true, false, repeatTimes, dstRepStride, srcRepStride};
        const TransDataTo5HDParams lowHigh{false, true, repeatTimes, dstRepStride, srcRepStride};
        const TransDataTo5HDParams highHigh{true, true, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t colGroup = 0; colGroup < colsAlign / 32; ++colGroup) {
            ComputeMatrixB8Group(inputLocal, outputLocal, colGroup * 32, rowsAlign, colsAlign,
                                 lowLow, highLow, lowHigh, highHigh);
        }
    }

    __aicore__ inline void ComputeMatrixB8Group(LocalTensor<T>& inputLocal, LocalTensor<T>& outputLocal,
                                                 uint64_t colBase, uint64_t rowsAlign, uint64_t colsAlign,
                                                 const TransDataTo5HDParams& lowLow,
                                                 const TransDataTo5HDParams& highLow,
                                                 const TransDataTo5HDParams& lowHigh,
                                                 const TransDataTo5HDParams& highHigh)
    {
        uint64_t srcLow[TRANS_ADDR_COUNT];
        uint64_t srcHigh[TRANS_ADDR_COUNT];
        uint64_t dstLow[TRANS_ADDR_COUNT];
        uint64_t dstHigh[TRANS_ADDR_COUNT];
        // B8 矩阵在 UB 中按连续行/列排布，四张 VA 表均为固定步长序列。
        // 仅查询每个序列的首地址，再用字节步长展开，避免每组 64 次重复
        // LocalTensor 地址查询；转置指令及其读写布局保持不变。
        const uint64_t srcLowBase = reinterpret_cast<uint64_t>(inputLocal[colBase].GetPhyAddr());
        const uint64_t srcHighBase = reinterpret_cast<uint64_t>(
            inputLocal[TRANS_ADDR_COUNT * colsAlign + colBase].GetPhyAddr());
        const uint64_t dstLowBase = reinterpret_cast<uint64_t>(outputLocal[colBase * rowsAlign].GetPhyAddr());
        const uint64_t dstHighBase = reinterpret_cast<uint64_t>(
            outputLocal[(colBase + TRANS_ADDR_COUNT) * rowsAlign].GetPhyAddr());
        const uint64_t srcRowStrideBytes = colsAlign * sizeof(T);
        const uint64_t dstColumnStrideBytes = rowsAlign * sizeof(T);
        for (uint64_t i = 0; i < TRANS_ADDR_COUNT; ++i) {
            srcLow[i] = srcLowBase + i * srcRowStrideBytes;
            srcHigh[i] = srcHighBase + i * srcRowStrideBytes;
            dstLow[i] = dstLowBase + i * dstColumnStrideBytes;
            dstHigh[i] = dstHighBase + i * dstColumnStrideBytes;
        }
        TransDataTo5HD<T>(dstLow, srcLow, lowLow);
        TransDataTo5HD<T>(dstLow, srcHigh, highLow);
        TransDataTo5HD<T>(dstHigh, srcLow, lowHigh);
        TransDataTo5HD<T>(dstHigh, srcHigh, highHigh);
    }

    __aicore__ inline void CopyOutDenseMatrix(uint64_t batch, uint64_t rowStart, uint64_t colStart,
                                              uint64_t rowsReal, uint64_t rowsAlign, uint64_t colsReal,
                                              uint64_t matrixRows, uint64_t matrixCols)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t roundedRowBytes = AlignUpU64(rowsReal * sizeof(T), BLOCK_BYTES);
        const uint64_t rowPitchBytes = rowsAlign * sizeof(T);
        const uint64_t outputOffset = batch * matrixRows * matrixCols + colStart * matrixRows + rowStart;
        const uint64_t blockLenBytes = rowsReal * sizeof(T);
        const uint64_t sourceStrideBlocks = (rowPitchBytes - roundedRowBytes) / BLOCK_BYTES;
        const uint64_t destinationStrideBytes = (matrixRows - rowsReal) * sizeof(T);
        if (colsReal <= DATA_COPY_BLOCK_COUNT_MAX &&
            blockLenBytes <= DATA_COPY_SHORT_PARAM_MAX &&
            sourceStrideBlocks <= DATA_COPY_SHORT_PARAM_MAX &&
            destinationStrideBytes <= DATA_COPY_SHORT_PARAM_MAX) {
            DataCopyParams params{static_cast<uint16_t>(colsReal),
                                  static_cast<uint16_t>(blockLenBytes),
                                  static_cast<uint16_t>(sourceStrideBlocks),
                                  static_cast<uint16_t>(destinationStrideBytes)};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
            outputQueue_.FreeTensor(outputLocal);
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(colsReal), static_cast<uint32_t>(rowsReal * sizeof(T)),
                                 static_cast<uint32_t>((rowPitchBytes - roundedRowBytes) / BLOCK_BYTES),
                                 static_cast<uint32_t>((matrixRows - rowsReal) * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        outputQueue_.FreeTensor(outputLocal);
    }

    template <bool FIXED_AXIS_ONE>
    __aicore__ inline void CopyInStridedMatrix(uint64_t batch, uint64_t fixedIndex,
                                               uint64_t rowStart, uint64_t rowsReal,
                                               uint64_t colStart, uint64_t colsReal,
                                               uint64_t colsAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{colsAlign != colsReal, 0,
            static_cast<uint8_t>(colsAlign - colsReal), static_cast<T>(0)};
        const uint64_t batchBase = batch * tiling_->outer * tiling_->rows *
                                   tiling_->cols * tiling_->inner;
        if constexpr (FIXED_AXIS_ONE) {
            // fixed 轴退化后，展平矩阵的相邻行在 GM 中连续，可合并为一条二维 DMA。
            const uint64_t inputOffset = batchBase + rowStart * tiling_->inner + colStart;
            const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
            const uint64_t sourceGapElements = tiling_->inner - colsReal;
            constexpr uint64_t maxSliceIndex = 0xffffffffULL;
            const uint64_t dstEnd = rowsReal * colsReal - 1;
            const uint64_t srcEnd = (rowsReal - 1) * tiling_->inner + colsReal - 1;
            // SliceInfo 在 GM gap 不是 32B 整数倍时使用专门的 slice MTE2 编码；仅在
            // 结果行无需 padding、整列由 32B block 组成且索引字段可安全表达时启用。
            // 其他布局保留 DataCopyPad，以维持所有非对齐和尾块的既有语义。
            if (colsReal == colsAlign && colsReal % blockElements == 0 &&
                sourceGapElements != 0 && sourceGapElements * sizeof(T) % BLOCK_BYTES != 0 &&
                dstEnd <= maxSliceIndex && srcEnd <= maxSliceIndex) {
                const uint32_t burstBlocks = static_cast<uint32_t>(colsReal / blockElements);
                SliceInfo dstSlice[1] = {{0, static_cast<uint32_t>(dstEnd), 0, burstBlocks}};
                SliceInfo srcSlice[1] = {{0, static_cast<uint32_t>(srcEnd),
                    static_cast<uint32_t>(sourceGapElements), burstBlocks}};
                DataCopy(inputLocal, inputGm_[inputOffset], dstSlice, srcSlice, 1);
                inputQueue_.EnQue(inputLocal);
                return;
            }
            DataCopyExtParams params{static_cast<uint16_t>(rowsReal),
                                     static_cast<uint32_t>(colsReal * sizeof(T)),
                                     static_cast<uint32_t>((tiling_->inner - colsReal) * sizeof(T)), 0, 0};
            DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
            inputQueue_.EnQue(inputLocal);
            return;
        }
        const uint64_t rowEnd = rowStart + rowsReal;
        uint64_t sourceRow = rowStart;
        while (sourceRow < rowEnd) {
            const uint64_t leftOuter = sourceRow / tiling_->cols;
            const uint64_t leftInner = sourceRow % tiling_->cols;
            const uint64_t segmentRows = MinU64(rowEnd - sourceRow, tiling_->cols - leftInner);
            const uint64_t inputOffset = batchBase +
                ((leftOuter * tiling_->rows + fixedIndex) * tiling_->cols + leftInner) *
                    tiling_->inner + colStart;
            constexpr uint64_t maxSliceIndex = 0xffffffffULL;
            const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
            const uint64_t sourceGapElements = tiling_->inner - colsReal;
            const uint64_t dstEnd = segmentRows * colsReal - 1;
            const uint64_t srcEnd = (segmentRows - 1) * tiling_->inner + colsReal - 1;
            // 对同一 leftInner 自然段，整 32B 列可使用 slice MTE2 直接编码非对齐
            // source gap；尾列、UB padding 和跨段情形继续走既有 DataCopyPad 语义。
            if (segmentRows > 1 && colsReal == colsAlign && colsReal % blockElements == 0 &&
                sourceGapElements != 0 && sourceGapElements * sizeof(T) % BLOCK_BYTES != 0 &&
                dstEnd <= maxSliceIndex && srcEnd <= maxSliceIndex) {
                const uint32_t burstBlocks = static_cast<uint32_t>(colsReal / blockElements);
                SliceInfo dstSlice[1] = {{0, static_cast<uint32_t>(dstEnd), 0, burstBlocks}};
                SliceInfo srcSlice[1] = {{0, static_cast<uint32_t>(srcEnd),
                    static_cast<uint32_t>(sourceGapElements), burstBlocks}};
                DataCopy(inputLocal[(sourceRow - rowStart) * colsAlign], inputGm_[inputOffset],
                    dstSlice, srcSlice, 1);
            } else {
            DataCopyExtParams params{static_cast<uint16_t>(segmentRows),
                                     static_cast<uint32_t>(colsReal * sizeof(T)),
                                     static_cast<uint32_t>((tiling_->inner - colsReal) * sizeof(T)), 0, 0};
            DataCopyPad(inputLocal[(sourceRow - rowStart) * colsAlign],
                        inputGm_[inputOffset], params, padParams);
            }
            sourceRow += segmentRows;
        }
        inputQueue_.EnQue(inputLocal);
    }

    template <bool DIRECT_VA_ADDRESS_TABLE>
    __aicore__ inline void ComputeStridedMatrix(LocalTensor<T>& outputLocal, uint64_t rowStart,
                                                uint64_t rowsReal,
                                                uint64_t outputRows,
                                                uint64_t colsAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(colsAlign / blockElements);
        TransDataTo5HDParams params{false, false, repeatTimes,
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : outputRows),
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        // 固定轴 FP32 的常见完整 chunk 可由三个完整 16 行地址组组成。该分支仅依赖
        // 运行时 tile 粒度，不依赖具体 shape；省去满 chunk 内的循环边界/安全行判断，
        // 尾 chunk 与其它 dtype 继续使用下面的泛化地址表。
        if constexpr (sizeof(T) == 4) {
            if (rowsReal == TRANS_ADDR_COUNT * 3) {
                ComputeFullFp32AddressGroup<0>(inputLocal, outputLocal, rowStart, outputRows,
                                                colsAlign, params);
                ComputeFullFp32AddressGroup<TRANS_ADDR_COUNT>(inputLocal, outputLocal,
                                                               rowStart, outputRows, colsAlign, params);
                ComputeFullFp32AddressGroup<TRANS_ADDR_COUNT * 2>(inputLocal, outputLocal,
                                                                   rowStart, outputRows, colsAlign, params);
                inputQueue_.FreeTensor(inputLocal);
                return;
            }
        }
        const uint64_t rowsAlign = AlignUpU64(rowsReal, TRANS_ADDR_COUNT);
        const uint64_t inputRowStrideBytes = colsAlign * sizeof(T);
        const uint64_t outputColumnStrideBytes = outputRows * sizeof(T);

        for (uint64_t rowBase = 0; rowBase < rowsAlign; rowBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            if constexpr (DIRECT_VA_ADDRESS_TABLE) {
                const uint64_t srcBase = reinterpret_cast<uint64_t>(
                    inputLocal[rowBase * colsAlign].GetPhyAddr());
                for (uint64_t i = 0; i < TRANS_ADDR_COUNT; ++i) {
                    srcList[i] = srcBase + i * inputRowStrideBytes;
                }
            } else {
                for (uint64_t i = 0; i < TRANS_ADDR_COUNT; ++i) {
                    const uint64_t safeRow = rowBase + i < rowsReal ? rowBase + i : 0;
                    srcList[i] = reinterpret_cast<uint64_t>(inputLocal[safeRow * colsAlign].GetPhyAddr());
                }
            }
            uint64_t dstList[TRANS_ADDR_COUNT];
            if constexpr (DIRECT_VA_ADDRESS_TABLE) {
                const uint64_t dstBase = reinterpret_cast<uint64_t>(
                    outputLocal[rowStart + rowBase].GetPhyAddr());
                if constexpr (sizeof(T) == 4) {
                    for (uint64_t col = 0; col < blockElements; ++col) {
                        const uint64_t colBase = dstBase + col * outputColumnStrideBytes;
                        dstList[col * 2] = colBase;
                        dstList[col * 2 + 1] = colBase + blockElements * sizeof(T);
                    }
                } else {
                    for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                        dstList[col] = dstBase + col * outputColumnStrideBytes;
                    }
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    for (uint64_t col = 0; col < blockElements; ++col) {
                        dstList[col * 2] = reinterpret_cast<uint64_t>(
                            outputLocal[col * outputRows + rowStart + rowBase].GetPhyAddr());
                        dstList[col * 2 + 1] = reinterpret_cast<uint64_t>(
                            outputLocal[col * outputRows + rowStart + rowBase + blockElements].GetPhyAddr());
                    }
                } else {
                    for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                        dstList[col] = reinterpret_cast<uint64_t>(
                            outputLocal[col * outputRows + rowStart + rowBase].GetPhyAddr());
                    }
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
        inputQueue_.FreeTensor(inputLocal);
    }

    template <uint64_t ROW_BASE>
    __aicore__ inline void ComputeFullFp32AddressGroup(LocalTensor<T>& inputLocal,
                                                        LocalTensor<T>& outputLocal,
                                                        uint64_t rowStart, uint64_t outputRows,
                                                        uint64_t colsAlign,
                                                        const TransDataTo5HDParams& params)
    {
        constexpr uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        uint64_t srcList[TRANS_ADDR_COUNT];
        uint64_t dstList[TRANS_ADDR_COUNT];
        for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(
                inputLocal[(ROW_BASE + row) * colsAlign].GetPhyAddr());
        }
        for (uint64_t col = 0; col < blockElements; ++col) {
            dstList[col * 2] = reinterpret_cast<uint64_t>(
                outputLocal[col * outputRows + rowStart + ROW_BASE].GetPhyAddr());
            dstList[col * 2 + 1] = reinterpret_cast<uint64_t>(
                outputLocal[col * outputRows + rowStart + ROW_BASE + blockElements].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstList, srcList, params);
    }

    __aicore__ inline void CopyOutStridedMatrix(uint64_t batch, uint64_t fixedIndex,
                                                uint64_t rowStart, uint64_t rowsReal,
                                                uint64_t colStart, uint64_t colsReal,
                                                uint64_t matrixRows, uint64_t outputRows)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t sourceStride =
            (outputRows * sizeof(T) - AlignUpU64(rowsReal * sizeof(T), BLOCK_BYTES)) / BLOCK_BYTES;
        const uint64_t outputOffset = batch * tiling_->inner * tiling_->rows * matrixRows +
                                      (colStart * tiling_->rows + fixedIndex) * matrixRows + rowStart;
        const uint64_t destinationStrideBytes =
            (tiling_->rows * matrixRows - rowsReal) * sizeof(T);
        if (colsReal <= DATA_COPY_BLOCK_COUNT_MAX &&
            rowsReal * sizeof(T) <= DATA_COPY_SHORT_PARAM_MAX &&
            sourceStride <= DATA_COPY_SHORT_PARAM_MAX &&
            destinationStrideBytes <= DATA_COPY_SHORT_PARAM_MAX) {
            // 连续输出 panel 的短描述符可避免 Ext 参数解析；超范围场景保持 Ext 泛化路径。
            DataCopyParams params{static_cast<uint16_t>(colsReal),
                                  static_cast<uint16_t>(rowsReal * sizeof(T)),
                                  static_cast<uint16_t>(sourceStride),
                                  static_cast<uint16_t>(destinationStrideBytes)};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
            outputQueue_.FreeTensor(outputLocal);
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(colsReal),
                                 static_cast<uint32_t>(rowsReal * sizeof(T)),
                                 static_cast<uint32_t>(sourceStride),
                                 static_cast<uint32_t>(destinationStrideBytes), 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyInRecordBlock(uint64_t batch, uint64_t fixedIndex,
                                             uint64_t rowStart, uint64_t colStart,
                                             uint64_t rowsReal, uint64_t colsReal,
                                             uint64_t recordAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{true, 0,
            static_cast<uint8_t>(recordAlign - tiling_->record), static_cast<T>(0)};
        const uint32_t destinationStride =
            static_cast<uint32_t>((rowsReal - 1) * recordAlign * sizeof(T) / BLOCK_BYTES);
        DataCopyExtParams params{static_cast<uint16_t>(colsReal),
                                 static_cast<uint32_t>(tiling_->record * sizeof(T)),
                                 0, destinationStride, 0};
        const uint64_t batchElements = tiling_->outer * tiling_->rows * tiling_->cols *
                                       tiling_->inner * tiling_->record;
        for (uint64_t rowOffset = 0; rowOffset < rowsReal; ++rowOffset) {
            const uint64_t matrixRow = rowStart + rowOffset;
            const uint64_t leftOuter = matrixRow / tiling_->cols;
            const uint64_t leftInner = matrixRow % tiling_->cols;
            const uint64_t inputOffset = batch * batchElements +
                ((((leftOuter * tiling_->rows + fixedIndex) * tiling_->cols + leftInner) *
                   tiling_->inner + colStart) * tiling_->record);
            DataCopyPad(inputLocal[rowOffset * recordAlign], inputGm_[inputOffset], params, padParams);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void PackPaddedRecords(uint64_t colsReal, uint64_t colsAlign,
                                             uint64_t rowsReal, uint64_t recordAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(rowsReal * recordAlign / blockElements);

        TransDataTo5HDParams firstParams{false, false, repeatTimes,
                                         static_cast<uint16_t>(repeatTimes <= 1 ? 0 : colsAlign),
                                         static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        for (uint64_t colBase = 0; colBase < colsAlign; colBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            uint64_t dstList[TRANS_ADDR_COUNT];
            for (uint64_t i = 0; i < TRANS_ADDR_COUNT; ++i) {
                const uint64_t safeCol = colBase + i < colsReal ? colBase + i : 0;
                srcList[i] = reinterpret_cast<uint64_t>(
                    inputLocal[safeCol * rowsReal * recordAlign].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(outputLocal[colBase + i * colsAlign].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, firstParams);
        }
        PipeBarrier<PIPE_V>();

        DataCopyParams compactParams{static_cast<uint16_t>(rowsReal),
                                     static_cast<uint16_t>(tiling_->record * colsAlign / blockElements),
                                     static_cast<uint16_t>((recordAlign - tiling_->record) *
                                                           colsAlign / blockElements), 0};
        DataCopy(inputLocal, outputLocal, compactParams);
        PipeBarrier<PIPE_V>();

        TransDataTo5HDParams secondParams{false, false, repeatTimes,
                                          static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1),
                                          static_cast<uint16_t>(repeatTimes <= 1 ? 0 : colsAlign)};
        for (uint64_t colBase = 0; colBase < colsAlign; colBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            uint64_t dstList[TRANS_ADDR_COUNT];
            for (uint64_t i = 0; i < TRANS_ADDR_COUNT; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(inputLocal[colBase + i * colsAlign].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(
                    outputLocal[colBase * rowsReal * recordAlign + i * rowsReal * recordAlign].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, secondParams);
        }
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOutAlignedRecord(uint64_t batch, uint64_t fixedIndex,
                                                uint64_t rowStart, uint64_t colStart,
                                                uint64_t rowsReal, uint64_t colsReal,
                                                uint64_t recordAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        SyncEvent<HardEvent::MTE2_MTE3>();
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t outputOffset = batch * tiling_->inner * tiling_->rows *
                                      matrixRows * tiling_->record +
                                      ((colStart * tiling_->rows + fixedIndex) * matrixRows + rowStart) *
                                      tiling_->record;
        DataCopyExtParams params{static_cast<uint16_t>(colsReal),
                                 static_cast<uint32_t>(rowsReal * tiling_->record * sizeof(T)), 0,
                                 static_cast<uint32_t>((tiling_->rows * matrixRows - rowsReal) *
                                                       tiling_->record * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputOffset], inputLocal, params);
        SyncEvent<HardEvent::MTE3_MTE2>();
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOutPackedRecord(uint64_t batch, uint64_t fixedIndex,
                                               uint64_t rowStart, uint64_t colStart,
                                               uint64_t rowsReal, uint64_t colsReal,
                                               uint64_t recordAlign)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t localRowBytes = rowsReal * recordAlign * sizeof(T);
        const uint64_t validRowBytes = rowsReal * tiling_->record * sizeof(T);
        const uint64_t sourceStride = (localRowBytes - AlignUpU64(validRowBytes, BLOCK_BYTES)) / BLOCK_BYTES;
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t outputOffset = batch * tiling_->inner * tiling_->rows *
                                      matrixRows * tiling_->record +
                                      ((colStart * tiling_->rows + fixedIndex) * matrixRows + rowStart) *
                                      tiling_->record;
        DataCopyExtParams params{static_cast<uint16_t>(colsReal), static_cast<uint32_t>(validRowBytes),
                                 static_cast<uint32_t>(sourceStride),
                                 static_cast<uint32_t>((tiling_->rows * matrixRows - rowsReal) *
                                                       tiling_->record * sizeof(T)), 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void InitGatherOffsets(uint64_t inputPitchBytes)
    {
        LocalTensor<uint32_t> offsets = gatherOffsetBuffer_.Get<uint32_t>();
        constexpr uint64_t maxSignedOffset = 0x7fffffffULL;
        constexpr uint64_t maxUnsigned32 = 0xffffffffULL;
        // Host tiling 已限制 DMA 相关字段为 32 位；在此基础上用 32x32 乘法
        // 保证中间结果不溢出，避免设备侧生成 64 位除法辅助函数。
        const bool countFits = tiling_->record != 0 &&
            tiling_->tileRows <= maxUnsigned32 && tiling_->record <= maxUnsigned32;
        const uint64_t validCount = countFits ?
            static_cast<uint64_t>(static_cast<uint32_t>(tiling_->tileRows)) *
                static_cast<uint64_t>(static_cast<uint32_t>(tiling_->record)) : 0;
        constexpr uint64_t gatherRepeatElements = 8 * BLOCK_BYTES / sizeof(T);
        const uint64_t storedCount =
            validCount > gatherRepeatElements && validCount % gatherRepeatElements != 0 ?
                AlignUpU64(validCount, gatherRepeatElements) : validCount;

        // offset 表按逻辑行紧凑存储，但 record 通常不是 8 的倍数，不能直接对每行
        // 调用 Vector API。先用 Arange 生成连续的 record*sizeof(T) 基表，再只对
        // 32B 对齐的中间块叠加行间步长；行首/行尾的非对齐元素保留标量修正。
        // 地址超过 int32 范围时不使用有符号 Arange，继续走原始标量路径。
        // 先验证乘法范围，再计算最后一个有效 offset；不能让异常 tiling 在
        // 这里发生无符号回绕后误入有符号 Arange 路径。
        const bool recordOffsetFits = tiling_->record != 0 &&
            tiling_->record - 1 <= maxSignedOffset / sizeof(T);
        const uint64_t recordOffset = recordOffsetFits ?
            (tiling_->record - 1) * sizeof(T) : 0;
        bool maxOffsetFits = countFits && tiling_->tileRows != 0 && recordOffsetFits;
        uint64_t rowOffset = 0;
        if (maxOffsetFits && tiling_->tileRows > 1) {
            maxOffsetFits = inputPitchBytes != 0 && inputPitchBytes <= maxUnsigned32 &&
                tiling_->tileRows - 1 <= maxUnsigned32;
            if (maxOffsetFits) {
                rowOffset = static_cast<uint64_t>(static_cast<uint32_t>(tiling_->tileRows - 1)) *
                    static_cast<uint64_t>(static_cast<uint32_t>(inputPitchBytes));
                maxOffsetFits = rowOffset <= maxSignedOffset - recordOffset;
            }
        }
        const uint64_t maxOffset = maxOffsetFits ? rowOffset + recordOffset : maxSignedOffset + 1;
        const bool canVectorize = tiling_->record >= BLOCK_BYTES / sizeof(int32_t) &&
            storedCount > 0 && storedCount <= 0x7fffffffULL && maxOffset <= 0x7fffffffULL;
        if (canVectorize) {
            LocalTensor<int32_t> offsetsI32 = gatherOffsetBuffer_.Get<int32_t>();
            constexpr uint64_t offsetAlignElements = BLOCK_BYTES / sizeof(int32_t);
            const uint64_t groupRows = offsetAlignElements /
                GreatestCommonDivisorU64(tiling_->record, offsetAlignElements);
            const uint64_t groupElements = groupRows * tiling_->record;
            // record 与 32B 边界的相位会在 groupRows 行后复位。先只构造一个
            // 对齐行组，再以 Adds 复制完整组和尾组，避免每个逻辑行都由 Scalar
            // 回填两端。仅在至少有两个完整相位时启用，其他布局保持原路径。
            const bool usePeriodicOffsetTemplate = groupRows != 0 && groupElements >= offsetAlignElements &&
                tiling_->tileRows >= groupRows * 2 && groupElements <= 0x7fffffffULL;
            if (usePeriodicOffsetTemplate) {
                Arange(offsetsI32, 0, static_cast<int32_t>(sizeof(T)),
                       static_cast<int32_t>(groupElements));
                SyncEvent<HardEvent::V_S>();
                const int64_t rowCorrection = static_cast<int64_t>(inputPitchBytes) -
                    static_cast<int64_t>(tiling_->record * sizeof(T));
                for (uint64_t row = 0; row < groupRows; ++row) {
                    const uint64_t rowStart = row * tiling_->record;
                    const uint64_t rowEnd = rowStart + tiling_->record;
                    const int64_t correction = static_cast<int64_t>(row) * rowCorrection;
                    const uint64_t firstAligned = AlignUpU64(rowStart, offsetAlignElements);
                    const uint64_t middleEnd = AlignDownU64(rowEnd, offsetAlignElements);
                    for (uint64_t index = rowStart; index < MinU64(firstAligned, rowEnd); ++index) {
                        offsetsI32.SetValue(index, static_cast<int32_t>(
                            static_cast<int64_t>(index) * sizeof(T) + correction));
                    }
                    for (uint64_t index = middleEnd; index < rowEnd; ++index) {
                        offsetsI32.SetValue(index, static_cast<int32_t>(
                            static_cast<int64_t>(index) * sizeof(T) + correction));
                    }
                }
                SyncEvent<HardEvent::S_V>();
                constexpr uint64_t addsChunkElements = 64;
                for (uint64_t row = 0; row < groupRows; ++row) {
                    const uint64_t rowStart = row * tiling_->record;
                    const uint64_t rowEnd = rowStart + tiling_->record;
                    const uint64_t firstAligned = AlignUpU64(rowStart, offsetAlignElements);
                    const uint64_t middleEnd = AlignDownU64(rowEnd, offsetAlignElements);
                    const int32_t correction = static_cast<int32_t>(
                        static_cast<int64_t>(row) * rowCorrection);
                    for (uint64_t index = firstAligned; index < middleEnd;) {
                        const uint64_t count = MinU64(addsChunkElements, middleEnd - index);
                        Adds(offsetsI32[index], offsetsI32[index], correction,
                             static_cast<int32_t>(count));
                        index += count;
                    }
                }
                PipeBarrier<PIPE_V>();
                for (uint64_t rowStart = groupRows; rowStart < tiling_->tileRows;
                     rowStart += groupRows) {
                    const uint64_t rowsThisGroup = MinU64(groupRows, tiling_->tileRows - rowStart);
                    const uint64_t copyCount = rowsThisGroup * tiling_->record;
                    const int32_t groupCorrection = static_cast<int32_t>(
                        rowStart * inputPitchBytes);
                    uint64_t index = 0;
                    // 连续 64 个 int32 可由一次 Level-0 vadds 覆盖；用 repeat
                    // 合并周期组内的整块复制，尾部仍走 Level-2 计数模式。
                    SetVectorMask<int32_t>(0, 0xffffffffffffffffULL);
                    const UnaryRepeatParams contiguousAddsParams{1, 1, 8, 8};
                    while (copyCount - index >= addsChunkElements) {
                        const uint64_t fullElements =
                            MinU64(copyCount - index, addsChunkElements * 255);
                        const uint8_t repeatTime = static_cast<uint8_t>(
                            fullElements / addsChunkElements);
                        Adds<int32_t, false>(
                            offsetsI32[rowStart * tiling_->record + index], offsetsI32[index],
                            groupCorrection, MASK_PLACEHOLDER, repeatTime, contiguousAddsParams);
                        index += static_cast<uint64_t>(repeatTime) * addsChunkElements;
                    }
                    while (index < copyCount) {
                        const uint64_t count = MinU64(addsChunkElements, copyCount - index);
                        Adds(offsetsI32[rowStart * tiling_->record + index], offsetsI32[index],
                             groupCorrection, static_cast<int32_t>(count));
                        index += count;
                    }
                }
                PipeBarrier<PIPE_V>();
                SyncEvent<HardEvent::V_S>();
                for (uint64_t index = validCount; index < storedCount; ++index) {
                    offsetsI32.SetValue(index, 0);
                }
                SyncEvent<HardEvent::S_V>();
                return;
            }
            Arange(offsetsI32, 0, static_cast<int32_t>(sizeof(T)),
                   static_cast<int32_t>(storedCount));
            // Arange 的向量结果需要先回到 Scalar，再写入非对齐边界和 padding。
            SyncEvent<HardEvent::V_S>();
            const uint64_t rowStrideElements = tiling_->record;
            // tileRows==1 时无需计算行间步长，避免对无关的极大 pitch 做有符号转换。
            const int64_t rowCorrection = tiling_->tileRows > 1 ?
                static_cast<int64_t>(inputPitchBytes) -
                    static_cast<int64_t>(tiling_->record * sizeof(T)) : 0;
            for (uint64_t row = 0; row < tiling_->tileRows; ++row) {
                const uint64_t rowStart = row * rowStrideElements;
                const uint64_t rowEnd = rowStart + rowStrideElements;
                const int64_t correction = static_cast<int64_t>(row) * rowCorrection;
                const uint64_t firstAligned = AlignUpU64(rowStart, offsetAlignElements);
                const uint64_t middleEnd =
                    AlignDownU64(rowEnd, offsetAlignElements);
                const uint64_t prefixEnd = MinU64(firstAligned, rowEnd);
                for (uint64_t index = rowStart; index < prefixEnd; ++index) {
                    offsetsI32.SetValue(index, static_cast<int32_t>(
                        static_cast<int64_t>(index) * sizeof(T) + correction));
                }
                for (uint64_t index = middleEnd; index < rowEnd; ++index) {
                    offsetsI32.SetValue(index, static_cast<int32_t>(
                        static_cast<int64_t>(index) * sizeof(T) + correction));
                }
            }
            SyncEvent<HardEvent::S_V>();
            constexpr uint64_t addsChunkElements = 64;
            for (uint64_t row = 0; row < tiling_->tileRows; ++row) {
                const uint64_t rowStart = row * rowStrideElements;
                const uint64_t rowEnd = rowStart + rowStrideElements;
                const uint64_t firstAligned = AlignUpU64(rowStart, offsetAlignElements);
                const uint64_t middleEnd =
                    AlignDownU64(rowEnd, offsetAlignElements);
                const int32_t correction = static_cast<int32_t>(
                    static_cast<int64_t>(row) * rowCorrection);
                for (uint64_t index = firstAligned; index < middleEnd;) {
                    const uint64_t count = MinU64(addsChunkElements, middleEnd - index);
                    Adds(offsetsI32[index], offsetsI32[index], correction,
                         static_cast<int32_t>(count));
                    index += count;
                }
            }
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_S>();
            // 补齐 lane 统一指向当前列首元素；仅完整行 tile 会消费这些 offset。
            for (uint64_t index = validCount; index < storedCount; ++index) {
                offsetsI32.SetValue(index, 0);
            }
            SyncEvent<HardEvent::S_V>();
            return;
        }
        for (uint64_t row = 0; row < tiling_->tileRows; ++row) {
            for (uint64_t recordOffset = 0; recordOffset < tiling_->record; ++recordOffset) {
                offsets.SetValue(row * tiling_->record + recordOffset,
                    static_cast<uint32_t>(row * inputPitchBytes + recordOffset * sizeof(T)));
            }
        }
        // 补齐 lane 统一指向当前列首元素；仅完整行 tile 会消费这些 offset。
        for (uint64_t index = validCount; index < storedCount; ++index) {
            offsets.SetValue(index, 0);
        }
        SyncEvent<HardEvent::S_V>();
    }

    __aicore__ inline void CopyInGatherRecordBlock(uint64_t batch, uint64_t fixedIndex,
                                                   uint64_t rowStart, uint64_t colStart,
                                                   uint64_t rowsReal, uint64_t colsReal,
                                                   uint64_t inputPitchBytes)
    {
        LocalTensor<T> inputLocal = gatherInputQueue_.AllocTensor<T>();
        const uint64_t recordBytes = tiling_->record * sizeof(T);
        const uint64_t blockBytes = colsReal * recordBytes;
        const uint64_t batchElements = tiling_->outer * tiling_->rows * tiling_->cols *
                                       tiling_->inner * tiling_->record;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        // leftInner 为 1 时，相邻逻辑行在 GM 上具有固定步长，可用一次二维 DMA 取代逐行发射。
        if (tiling_->cols == 1 && rowsReal <= DATA_COPY_BLOCK_COUNT_MAX &&
            blockBytes <= DATA_COPY_BLOCK_LEN_MAX) {
            const uint64_t roundedBlockBytes = AlignUpU64(blockBytes, BLOCK_BYTES);
            const uint64_t sourceRowBytes = tiling_->rows * tiling_->inner * recordBytes;
            if (sourceRowBytes >= blockBytes && inputPitchBytes >= roundedBlockBytes) {
                const uint64_t sourceStrideBytes = sourceRowBytes - blockBytes;
                const uint64_t destinationStrideBlocks =
                    (inputPitchBytes - roundedBlockBytes) / BLOCK_BYTES;
                const uint64_t uint32Max = ~static_cast<uint32_t>(0);
                if (sourceStrideBytes <= uint32Max && destinationStrideBlocks <= uint32Max) {
                    const uint64_t inputOffset = batch * batchElements +
                        (((rowStart * tiling_->rows + fixedIndex) * tiling_->inner + colStart) *
                         tiling_->record);
                    DataCopyExtParams params{static_cast<uint16_t>(rowsReal),
                        static_cast<uint32_t>(blockBytes), static_cast<uint32_t>(sourceStrideBytes),
                        static_cast<uint32_t>(destinationStrideBlocks), 0};
                    DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
                    gatherInputQueue_.EnQue(inputLocal);
                    return;
                }
            }

        }

        DataCopyExtParams params{1, static_cast<uint32_t>(blockBytes), 0, 0, 0};
        for (uint64_t rowOffset = 0; rowOffset < rowsReal; ++rowOffset) {
            const uint64_t matrixRow = rowStart + rowOffset;
            const uint64_t leftOuter = matrixRow / tiling_->cols;
            const uint64_t leftInner = matrixRow % tiling_->cols;
            const uint64_t inputOffset = batch * batchElements +
                ((((leftOuter * tiling_->rows + fixedIndex) * tiling_->cols + leftInner) *
                   tiling_->inner + colStart) * tiling_->record);
            DataCopyPad(inputLocal[rowOffset * inputPitchBytes / sizeof(T)],
                        inputGm_[inputOffset], params, padParams);
        }
        gatherInputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void GatherRecordPanel(LocalTensor<T>& outputLocal, uint64_t panelStart,
                                             uint64_t rowsReal, uint64_t panelReal,
                                             uint64_t outputPitchBytes)
    {
        LocalTensor<T> inputLocal = gatherInputQueue_.DeQue<T>();
        LocalTensor<uint32_t> offsets = gatherOffsetBuffer_.Get<uint32_t>();
        const uint64_t recordBytes = tiling_->record * sizeof(T);
        const uint64_t outputPitchElements = outputPitchBytes / sizeof(T);
        const uint32_t gatherCount = static_cast<uint32_t>(rowsReal * tiling_->record);
        constexpr uint32_t gatherRepeatElements = 8 * BLOCK_BYTES / sizeof(T);
        const uint32_t fullRepeat = gatherCount / gatherRepeatElements;
        const uint32_t fullElements = fullRepeat * gatherRepeatElements;
        const uint32_t tailElements = gatherCount - fullElements;
        const bool usePaddedFullRepeat = rowsReal == tiling_->tileRows &&
            fullRepeat != 0 && tailElements != 0;
        for (uint64_t col = 0; col < panelReal; ++col) {
            LocalTensor<T> outputColumn =
                outputLocal[(panelStart + col) * outputPitchElements];
            const uint32_t sourceBase = static_cast<uint32_t>(col * recordBytes);
            // repeat 数和尾长对整块 panel 恒定，直接调用 Level-0 接口，避免逐列重复商余与分派。
            if (usePaddedFullRepeat) {
                Gather(outputColumn, inputLocal, offsets, sourceBase,
                       static_cast<uint64_t>(gatherRepeatElements),
                       static_cast<uint8_t>(fullRepeat + 1), DEFAULT_REPEAT_STRIDE);
                continue;
            }
            if (fullRepeat != 0) {
                Gather(outputColumn, inputLocal, offsets, sourceBase,
                       static_cast<uint64_t>(gatherRepeatElements),
                       static_cast<uint8_t>(fullRepeat), DEFAULT_REPEAT_STRIDE);
            }
            if (tailElements != 0) {
                Gather(outputColumn[fullElements], inputLocal, offsets[fullElements], sourceBase,
                       static_cast<uint64_t>(tailElements), static_cast<uint8_t>(1),
                       DEFAULT_REPEAT_STRIDE);
            }
        }
        gatherInputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOutGatherRecordBlock(uint64_t batch, uint64_t fixedIndex,
                                                    uint64_t rowStart, uint64_t colStart,
                                                    uint64_t rowsReal, uint64_t colsReal,
                                                    uint64_t outputPitchBytes)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t recordBytes = tiling_->record * sizeof(T);
        const uint64_t validRowBytes = rowsReal * recordBytes;
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t outputOffset = batch * tiling_->inner * tiling_->rows *
                                      matrixRows * tiling_->record +
                                      ((colStart * tiling_->rows + fixedIndex) * matrixRows + rowStart) *
                                      tiling_->record;
        DataCopyExtParams params{static_cast<uint16_t>(colsReal), static_cast<uint32_t>(validRowBytes),
                                 static_cast<uint32_t>(
                                     (outputPitchBytes - AlignUpU64(validRowBytes, BLOCK_BYTES)) /
                                     BLOCK_BYTES),
                                 static_cast<uint32_t>(
                                     (tiling_->rows * matrixRows - rowsReal) * recordBytes), 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    const TransposeTilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    uint64_t blockIdx_{0};
    uint64_t blockDim_{1};
    uint64_t copyBufferBytes_{0};
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECIN, 2> gatherInputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> gatherOffsetBuffer_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
};

// 固定轴多矩阵专用实现：仅保留既有的 VECIN 双队列、VECOUT 输出队列、二维 Slice MTE2、
// VA 转置和连续 MTE3 写回。Host 只在 rows == 1 时选择本类，因此无需携带其它微内核。
template <typename T>
class KernelTransposeFixedCompact {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const TransposeTilingData* tilingData, TPipe* pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_->blockDim;
    }

    __aicore__ inline void Process()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t chunkRows = tiling_->tileRows;
        const uint64_t panelRows = tiling_->panelRows;
        const uint64_t tileCols = tiling_->tileCols;
        const uint64_t tileColsAlign = AlignUpU64(tileCols, BLOCK_BYTES / sizeof(T));
        pipe_->InitBuffer(inputQueue_, tiling_->bufferNum, chunkRows * tileColsAlign * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 1, panelRows * tileColsAlign * sizeof(T));

        const uint64_t colTileCount = CeilDivU64(tiling_->inner, tileCols);
        const uint64_t colTilesPerTask = tiling_->colTilesPerTask == 0 ? 1 :
            MinU64(tiling_->colTilesPerTask, colTileCount);
        const uint64_t colTaskCount = CeilDivU64(colTileCount, colTilesPerTask);
        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            const uint64_t rawColTask = task % colTaskCount;
            // key 12 仅用于 rows == 1；matrixTask 直接对应 batch。
            const uint64_t batch = task / colTaskCount;
            const uint64_t rawColStart = rawColTask * colTilesPerTask;
            const uint64_t rawColEnd = MinU64(rawColStart + colTilesPerTask, colTileCount);
            for (uint64_t rawCol = rawColStart; rawCol < rawColEnd; ++rawCol) {
                const uint64_t colIndex = (rawCol + batch) % colTileCount;
                const uint64_t colStart = colIndex * tileCols;
                const uint64_t colsReal = MinU64(tileCols, tiling_->inner - colStart);
                const uint64_t colsAlign = AlignUpU64(colsReal, BLOCK_BYTES / sizeof(T));
                for (uint64_t panelStart = 0; panelStart < matrixRows; panelStart += panelRows) {
                    const uint64_t panelRowsReal = MinU64(panelRows, matrixRows - panelStart);
                    const uint64_t panelRowsAlign = AlignUpU64(panelRowsReal, TRANS_ADDR_COUNT);
                    const uint64_t firstRows = MinU64(chunkRows, panelRowsReal);
                    CopyIn(batch, panelStart, firstRows, colStart, colsReal, colsAlign);
                    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
                    Compute(outputLocal, 0, firstRows, panelRowsAlign, colsAlign);
                    for (uint64_t panelOffset = chunkRows; panelOffset < panelRowsReal;
                         panelOffset += chunkRows) {
                        const uint64_t rowsReal = MinU64(chunkRows, panelRowsReal - panelOffset);
                        CopyIn(batch, panelStart + panelOffset, rowsReal, colStart, colsReal, colsAlign);
                        Compute(outputLocal, panelOffset, rowsReal, panelRowsAlign, colsAlign);
                    }
                    outputQueue_.EnQue(outputLocal);
                    CopyOut(batch, panelStart, panelRowsReal, colStart, colsReal,
                            matrixRows, panelRowsAlign);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t batch, uint64_t rowStart, uint64_t rowsReal,
                                  uint64_t colStart, uint64_t colsReal, uint64_t colsAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        const uint64_t inputOffset = batch * tiling_->outer * tiling_->cols * tiling_->inner +
            rowStart * tiling_->inner + colStart;
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t sourceGap = tiling_->inner - colsReal;
        const uint64_t dstEnd = rowsReal * colsReal - 1;
        const uint64_t srcEnd = (rowsReal - 1) * tiling_->inner + colsReal - 1;
        constexpr uint64_t maxSliceIndex = 0xffffffffULL;
        if (colsReal == colsAlign && colsReal % blockElements == 0 && sourceGap != 0 &&
            sourceGap * sizeof(T) % BLOCK_BYTES != 0 && dstEnd <= maxSliceIndex &&
            srcEnd <= maxSliceIndex) {
            const uint32_t burstBlocks = static_cast<uint32_t>(colsReal / blockElements);
            SliceInfo dstSlice[1] = {{0, static_cast<uint32_t>(dstEnd), 0, burstBlocks}};
            SliceInfo srcSlice[1] = {{0, static_cast<uint32_t>(srcEnd),
                static_cast<uint32_t>(sourceGap), burstBlocks}};
            DataCopy(inputLocal, inputGm_[inputOffset], dstSlice, srcSlice, 1);
        } else {
            DataCopyPadExtParams<T> padParams{colsAlign != colsReal, 0,
                static_cast<uint8_t>(colsAlign - colsReal), static_cast<T>(0)};
            DataCopyExtParams params{static_cast<uint16_t>(rowsReal),
                static_cast<uint32_t>(colsReal * sizeof(T)),
                static_cast<uint32_t>((tiling_->inner - colsReal) * sizeof(T)), 0, 0};
            DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T>& outputLocal, uint64_t rowStart,
                                   uint64_t rowsReal, uint64_t outputRows, uint64_t colsAlign)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(colsAlign / blockElements);
        TransDataTo5HDParams params{false, false, repeatTimes,
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : outputRows),
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        const uint64_t rowsAlign = AlignUpU64(rowsReal, TRANS_ADDR_COUNT);
        for (uint64_t rowBase = 0; rowBase < rowsAlign; rowBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            uint64_t dstList[TRANS_ADDR_COUNT];
            for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
                const uint64_t safeRow = rowBase + row < rowsReal ? rowBase + row : 0;
                srcList[row] = reinterpret_cast<uint64_t>(inputLocal[safeRow * colsAlign].GetPhyAddr());
            }
            if constexpr (sizeof(T) == 4) {
                for (uint64_t col = 0; col < blockElements; ++col) {
                    dstList[col * 2] = reinterpret_cast<uint64_t>(
                        outputLocal[col * outputRows + rowStart + rowBase].GetPhyAddr());
                    dstList[col * 2 + 1] = reinterpret_cast<uint64_t>(
                        outputLocal[col * outputRows + rowStart + rowBase + blockElements].GetPhyAddr());
                }
            } else {
                for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                    dstList[col] = reinterpret_cast<uint64_t>(
                        outputLocal[col * outputRows + rowStart + rowBase].GetPhyAddr());
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t batch, uint64_t rowStart, uint64_t rowsReal,
                                   uint64_t colStart, uint64_t colsReal, uint64_t matrixRows,
                                   uint64_t outputRows)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t sourceStride =
            (outputRows * sizeof(T) - AlignUpU64(rowsReal * sizeof(T), BLOCK_BYTES)) / BLOCK_BYTES;
        const uint64_t outputOffset = batch * tiling_->inner * matrixRows + colStart * matrixRows + rowStart;
        const uint64_t destinationStrideBytes = (matrixRows - rowsReal) * sizeof(T);
        if (colsReal <= DATA_COPY_BLOCK_COUNT_MAX && rowsReal * sizeof(T) <= DATA_COPY_SHORT_PARAM_MAX &&
            sourceStride <= DATA_COPY_SHORT_PARAM_MAX && destinationStrideBytes <= DATA_COPY_SHORT_PARAM_MAX) {
            DataCopyParams params{static_cast<uint16_t>(colsReal),
                                  static_cast<uint16_t>(rowsReal * sizeof(T)),
                                  static_cast<uint16_t>(sourceStride),
                                  static_cast<uint16_t>(destinationStrideBytes)};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(colsReal),
                                     static_cast<uint32_t>(rowsReal * sizeof(T)),
                                     static_cast<uint32_t>(sourceStride),
                                     static_cast<uint32_t>(destinationStrideBytes), 0};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

    const TransposeTilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    uint64_t blockIdx_{0};
    uint64_t blockDim_{1};
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
};

#if defined(TRANSPOSE_PHASE_COMPACT_FP32)
// FP32 固定轴 Strided 的相位压缩路径。每行先向左扩到 32B 边界，以原生二维 DMA
// 分段搬入；随后按相位批量 GatherMask 压缩为紧凑且 32B 对齐的行，再交给 VA 转置。
// Host 仅在 dtype、相位斜率、DMA 字段和 UB 预算全部满足时选择本类；不满足的尾列仍
// 在本类内使用既有的安全搬运与 VA 地址表。
template <typename T, bool PhaseMajor, bool TwoSlot>
class KernelTransposeFixedPhaseCompact {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const TransposeTilingData* tilingData, TPipe* pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_->blockDim;
    }

    __aicore__ inline void Process()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t chunkRows = tiling_->tileRows;
        const uint64_t panelRows = tiling_->panelRows;
        const uint64_t tileCols = tiling_->tileCols;
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t tileColsAlign = AlignUpU64(tileCols, blockElements);
        phaseRawPitch_ = tileColsAlign + blockElements;
        pipe_->InitBuffer(phaseInputQueue_, TwoSlot ? 2 : 1,
                          chunkRows * phaseRawPitch_ * sizeof(T));
        pipe_->InitBuffer(compactInputBuffer_, chunkRows * tileColsAlign * sizeof(T));
        // 每相位使用一个 32B 槽位；有效 mask word 不足 32B 的余量不参与计算。
        const uint64_t maskRows = PhaseMajor ? chunkRows : blockElements;
        pipe_->InitBuffer(phaseMaskBuffer_, maskRows * blockElements * sizeof(uint32_t));
        pipe_->InitBuffer(outputQueue_, 1, panelRows * tileColsAlign * sizeof(T));
        if constexpr (PhaseMajor) {
            InitPhaseMajorMasks();
        } else {
            InitPhaseMasks();
        }
        PipeBarrier<PIPE_ALL>();

        const uint64_t colTileCount = CeilDivU64(tiling_->inner, tileCols);
        const uint64_t colTilesPerTask = tiling_->colTilesPerTask == 0 ? 1 :
            MinU64(tiling_->colTilesPerTask, colTileCount);
        const uint64_t colTaskCount = CeilDivU64(colTileCount, colTilesPerTask);
        for (uint64_t task = blockIdx_; task < tiling_->taskCount; task += blockDim_) {
            const uint64_t rawColTask = task % colTaskCount;
            const uint64_t batch = task / colTaskCount;
            const uint64_t rawColStart = rawColTask * colTilesPerTask;
            const uint64_t rawColEnd = MinU64(rawColStart + colTilesPerTask, colTileCount);
            for (uint64_t rawCol = rawColStart; rawCol < rawColEnd; ++rawCol) {
                const uint64_t colIndex = (rawCol + batch) % colTileCount;
                const uint64_t colStart = colIndex * tileCols;
                const uint64_t colsReal = MinU64(tileCols, tiling_->inner - colStart);
                const uint64_t colsAlign = AlignUpU64(colsReal, blockElements);
                for (uint64_t panelStart = 0; panelStart < matrixRows; panelStart += panelRows) {
                    const uint64_t panelRowsReal = MinU64(panelRows, matrixRows - panelStart);
                    const uint64_t panelRowsAlign = AlignUpU64(panelRowsReal, TRANS_ADDR_COUNT);
                    const uint64_t firstRows = MinU64(chunkRows, panelRowsReal);
                    bool firstPhaseCompact = false;
                    if constexpr (PhaseMajor) {
                        firstPhaseCompact = CanUsePhaseMajorTile(batch, panelStart, colStart,
                            firstRows, colsReal, colsAlign);
                    } else {
                        firstPhaseCompact = CanUsePhaseCompactTile(batch, panelStart, colStart,
                            colsReal, colsAlign);
                    }
                    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
                    if constexpr (TwoSlot) {
                        // 先把当前块和下一块送入两个 VECIN 槽，随后在计算当前块时让
                        // MTE2 提前搬运下一块；每次 Compute 释放一个槽后再补下一块。
                        CopyIn(batch, panelStart, firstRows, colStart, colsReal, colsAlign,
                               firstPhaseCompact);
                        if (firstRows < panelRowsReal) {
                            const uint64_t nextRows = MinU64(chunkRows, panelRowsReal - firstRows);
                            const uint64_t nextRow = panelStart + firstRows;
                            bool nextPhaseCompact = false;
                            if constexpr (PhaseMajor) {
                                nextPhaseCompact = CanUsePhaseMajorTile(batch, nextRow, colStart,
                                    nextRows, colsReal, colsAlign);
                            } else {
                                nextPhaseCompact = CanUsePhaseCompactTile(batch, nextRow, colStart,
                                    colsReal, colsAlign);
                            }
                            CopyIn(batch, nextRow, nextRows, colStart, colsReal, colsAlign,
                                   nextPhaseCompact);
                        }
                        Compute(outputLocal, 0, firstRows, panelRowsAlign, colsAlign, batch,
                                panelStart, colStart, colsReal, firstPhaseCompact);
                        for (uint64_t panelOffset = chunkRows; panelOffset < panelRowsReal;
                             panelOffset += chunkRows) {
                            const uint64_t rowsReal = MinU64(chunkRows, panelRowsReal - panelOffset);
                            const uint64_t sourceRow = panelStart + panelOffset;
                            if (panelOffset + chunkRows < panelRowsReal) {
                                const uint64_t nextRows = MinU64(chunkRows,
                                    panelRowsReal - panelOffset - chunkRows);
                                const uint64_t nextRow = sourceRow + chunkRows;
                                bool nextPhaseCompact = false;
                                if constexpr (PhaseMajor) {
                                    nextPhaseCompact = CanUsePhaseMajorTile(batch, nextRow, colStart,
                                        nextRows, colsReal, colsAlign);
                                } else {
                                    nextPhaseCompact = CanUsePhaseCompactTile(batch, nextRow, colStart,
                                        colsReal, colsAlign);
                                }
                                CopyIn(batch, nextRow, nextRows, colStart, colsReal, colsAlign,
                                       nextPhaseCompact);
                            }
                            bool phaseCompact = false;
                            if constexpr (PhaseMajor) {
                                phaseCompact = CanUsePhaseMajorTile(batch, sourceRow, colStart,
                                    rowsReal, colsReal, colsAlign);
                            } else {
                                phaseCompact = CanUsePhaseCompactTile(batch, sourceRow, colStart,
                                    colsReal, colsAlign);
                            }
                            Compute(outputLocal, panelOffset, rowsReal, panelRowsAlign, colsAlign,
                                    batch, sourceRow, colStart, colsReal, phaseCompact);
                        }
                    } else {
                        CopyIn(batch, panelStart, firstRows, colStart, colsReal, colsAlign,
                               firstPhaseCompact);
                        Compute(outputLocal, 0, firstRows, panelRowsAlign, colsAlign, batch,
                                panelStart, colStart, colsReal, firstPhaseCompact);
                        for (uint64_t panelOffset = chunkRows; panelOffset < panelRowsReal;
                             panelOffset += chunkRows) {
                            const uint64_t rowsReal = MinU64(chunkRows, panelRowsReal - panelOffset);
                            const uint64_t sourceRow = panelStart + panelOffset;
                            bool phaseCompact = false;
                            if constexpr (PhaseMajor) {
                                phaseCompact = CanUsePhaseMajorTile(batch, sourceRow, colStart,
                                    rowsReal, colsReal, colsAlign);
                            } else {
                                phaseCompact = CanUsePhaseCompactTile(batch, sourceRow, colStart,
                                    colsReal, colsAlign);
                            }
                            CopyIn(batch, sourceRow, rowsReal, colStart, colsReal, colsAlign,
                                   phaseCompact);
                            Compute(outputLocal, panelOffset, rowsReal, panelRowsAlign, colsAlign,
                                    batch, sourceRow, colStart, colsReal, phaseCompact);
                        }
                    }
                    outputQueue_.EnQue(outputLocal);
                    CopyOut(batch, panelStart, panelRowsReal, colStart, colsReal,
                            matrixRows, panelRowsAlign);
                }
            }
        }
    }

private:
    __aicore__ inline uint64_t GetInputOffset(uint64_t batch, uint64_t rowStart,
                                               uint64_t colStart) const
    {
        return batch * tiling_->outer * tiling_->cols * tiling_->inner +
            rowStart * tiling_->inner + colStart;
    }

    __aicore__ inline uint64_t GetInputPhase(uint64_t inputOffset) const
    {
        const uint64_t bytePhase = reinterpret_cast<uint64_t>(inputGm_.GetPhyAddr(inputOffset)) &
            (BLOCK_BYTES - 1);
        return bytePhase / sizeof(T);
    }

    __aicore__ inline bool CanUsePhaseCompactTile(uint64_t batch, uint64_t rowStart,
                                                   uint64_t colStart, uint64_t colsReal,
                                                   uint64_t colsAlign) const
    {
        if constexpr (sizeof(T) != 4) {
            return false;
        }
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        if (colsReal != colsAlign || colsAlign + blockElements != phaseRawPitch_ ||
            colStart > tiling_->inner || phaseRawPitch_ > tiling_->inner - colStart) {
            return false;
        }
        const uint64_t inputOffset = GetInputOffset(batch, rowStart, colStart);
        const uint64_t bytePhase = reinterpret_cast<uint64_t>(inputGm_.GetPhyAddr(inputOffset)) &
            (BLOCK_BYTES - 1);
        const uint64_t phase = bytePhase / sizeof(T);
        return bytePhase % sizeof(T) == 0 && phase < blockElements && inputOffset >= phase;
    }

    __aicore__ inline bool CanUsePhaseMajorTile(uint64_t batch, uint64_t rowStart,
                                                 uint64_t colStart, uint64_t rowsReal,
                                                 uint64_t colsReal, uint64_t colsAlign) const
    {
        return rowsReal % (BLOCK_BYTES / sizeof(T)) == 0 &&
            CanUsePhaseCompactTile(batch, rowStart, colStart, colsReal, colsAlign);
    }

    __aicore__ inline void InitPhaseMasks()
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        constexpr uint64_t maskWordBits = sizeof(uint32_t) * 8;
        const uint64_t maskWords = CeilDivU64(phaseRawPitch_, maskWordBits);
        const uint64_t selectedElements = phaseRawPitch_ - blockElements;
        LocalTensor<uint32_t> maskLocal = phaseMaskBuffer_.Get<uint32_t>();
        for (uint64_t phase = 0; phase < blockElements; ++phase) {
            for (uint64_t word = 0; word < maskWords; ++word) {
                const uint64_t wordBegin = word * maskWordBits;
                const uint64_t wordEnd = wordBegin + maskWordBits;
                const uint64_t selectedBegin = phase > wordBegin ? phase : wordBegin;
                const uint64_t phaseEnd = phase + selectedElements;
                const uint64_t selectedEnd = phaseEnd < wordEnd ? phaseEnd : wordEnd;
                uint32_t bitMask = 0;
                if (selectedBegin < selectedEnd) {
                    const uint32_t firstBit = static_cast<uint32_t>(selectedBegin - wordBegin);
                    const uint32_t selectedBits = static_cast<uint32_t>(selectedEnd - selectedBegin);
                    // 32 位整字需避开 1U << 32 的未定义移位；其它情况只构造首尾掩码。
                    const uint32_t selectedMask = selectedBits == maskWordBits ? 0xffffffffU :
                        static_cast<uint32_t>((1U << selectedBits) - 1U);
                    bitMask = static_cast<uint32_t>(selectedMask << firstBit);
                }
                maskLocal.SetValue(phase * blockElements + word, bitMask);
            }
            // Host 已限制有效 word 数不超过一个相位槽位，但显式清零余量，避免未来调整
            // tile 上限时把旧 mask 当成当前相位的有效位。
            for (uint64_t word = maskWords; word < blockElements; ++word) {
                maskLocal.SetValue(phase * blockElements + word, 0);
            }
        }
    }

    __aicore__ inline uint32_t BuildPhaseMaskWord(uint64_t phase, uint64_t word) const
    {
        constexpr uint64_t maskWordBits = sizeof(uint32_t) * 8;
        const uint64_t wordBegin = word * maskWordBits;
        const uint64_t wordEnd = wordBegin + maskWordBits;
        const uint64_t selectedElements = phaseRawPitch_ - (BLOCK_BYTES / sizeof(T));
        const uint64_t selectedBegin = phase > wordBegin ? phase : wordBegin;
        const uint64_t phaseEnd = phase + selectedElements;
        const uint64_t selectedEnd = phaseEnd < wordEnd ? phaseEnd : wordEnd;
        if (selectedBegin >= selectedEnd) {
            return 0;
        }
        const uint32_t firstBit = static_cast<uint32_t>(selectedBegin - wordBegin);
        const uint32_t selectedBits = static_cast<uint32_t>(selectedEnd - selectedBegin);
        const uint32_t selectedMask = selectedBits == maskWordBits ? 0xffffffffU :
            static_cast<uint32_t>((1U << selectedBits) - 1U);
        return static_cast<uint32_t>(selectedMask << firstBit);
    }

    __aicore__ inline void InitPhaseMajorMasks()
    {
        constexpr uint64_t maskWordBits = sizeof(uint32_t) * 8;
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t maskWords = CeilDivU64(phaseRawPitch_, maskWordBits);
        const uint64_t rowsPerPhase = tiling_->tileRows / blockElements;
        LocalTensor<uint32_t> maskLocal = phaseMaskBuffer_.Get<uint32_t>();
        // raw 按实际相位 0..7、每相位的原始行序存储；mask 表保持相同的 repeat 顺序。
        for (uint64_t phase = 0; phase < blockElements; ++phase) {
            for (uint64_t ordinal = 0; ordinal < rowsPerPhase; ++ordinal) {
                const uint64_t slot = phase * rowsPerPhase + ordinal;
                for (uint64_t word = 0; word < blockElements; ++word) {
                    const uint32_t value = word < maskWords ? BuildPhaseMaskWord(phase, word) : 0;
                    maskLocal.SetValue(slot * blockElements + word, value);
                }
            }
        }
    }

    __aicore__ inline void CopyInPhaseMajor(LocalTensor<T>& inputLocal, uint64_t batch,
                                             uint64_t rowStart, uint64_t rowsReal,
                                             uint64_t colStart)
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t firstInputOffset = GetInputOffset(batch, rowStart, colStart);
        const uint64_t firstPhase = GetInputPhase(firstInputOffset);
        const uint64_t rowsPerPhase = rowsReal / blockElements;
        const uint64_t sourceGapBlocks =
            (blockElements * tiling_->inner - phaseRawPitch_) / blockElements;
        const uint16_t burstBlocks = static_cast<uint16_t>(phaseRawPitch_ / blockElements);
        for (uint64_t phase = 0; phase < blockElements; ++phase) {
            const uint64_t firstRow = (phase + blockElements - firstPhase) % blockElements;
            if (firstRow >= rowsReal) {
                continue;
            }
            const uint64_t groupRows = CeilDivU64(rowsReal - firstRow, blockElements);
            // 同一相位的源行间隔为完整 8 行；目标按相位组连续排列，便于一次串流 GatherMask。
            const DataCopyParams params{
                static_cast<uint16_t>(groupRows), burstBlocks,
                static_cast<uint16_t>(sourceGapBlocks), 0};
            const uint64_t dstOffset = phase * rowsPerPhase * phaseRawPitch_;
            const uint64_t srcOffset = firstInputOffset + firstRow * tiling_->inner - phase;
            DataCopy(inputLocal[dstOffset], inputGm_[srcOffset], params);
        }
        phaseInputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void CopyIn(uint64_t batch, uint64_t rowStart, uint64_t rowsReal,
                                  uint64_t colStart, uint64_t colsReal, uint64_t colsAlign,
                                  bool phaseCompact)
    {
        LocalTensor<T> inputLocal = phaseInputQueue_.template AllocTensor<T>();
        const uint64_t inputOffset = GetInputOffset(batch, rowStart, colStart);
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        if (phaseCompact) {
            if constexpr (PhaseMajor) {
                CopyInPhaseMajor(inputLocal, batch, rowStart, rowsReal, colStart);
                return;
            }
            const uint64_t sourceGapBlocks =
                (tiling_->inner - 1 - phaseRawPitch_) / blockElements;
            const uint64_t firstPhase = GetInputPhase(inputOffset);
            for (uint64_t localRow = 0; localRow < rowsReal;) {
                const uint64_t phase = (firstPhase + localRow) % blockElements;
                const uint64_t runRows = MinU64(rowsReal - localRow, blockElements - phase);
                DataCopyParams params{static_cast<uint16_t>(runRows),
                                      static_cast<uint16_t>(phaseRawPitch_ / blockElements),
                                      static_cast<uint16_t>(sourceGapBlocks), 0};
                DataCopy(inputLocal[localRow * phaseRawPitch_],
                         inputGm_[inputOffset + localRow * tiling_->inner - phase], params);
                localRow += runRows;
            }
            phaseInputQueue_.EnQue(inputLocal);
            return;
        }

        const uint64_t sourceGap = tiling_->inner - colsReal;
        const uint64_t dstEnd = rowsReal * colsReal - 1;
        const uint64_t srcEnd = (rowsReal - 1) * tiling_->inner + colsReal - 1;
        constexpr uint64_t maxSliceIndex = 0xffffffffULL;
        if (colsReal == colsAlign && colsReal % blockElements == 0 && sourceGap != 0 &&
            sourceGap * sizeof(T) % BLOCK_BYTES != 0 && dstEnd <= maxSliceIndex &&
            srcEnd <= maxSliceIndex) {
            const uint32_t burstBlocks = static_cast<uint32_t>(colsReal / blockElements);
            SliceInfo dstSlice[1] = {{0, static_cast<uint32_t>(dstEnd), 0, burstBlocks}};
            SliceInfo srcSlice[1] = {{0, static_cast<uint32_t>(srcEnd),
                static_cast<uint32_t>(sourceGap), burstBlocks}};
            DataCopy(inputLocal, inputGm_[inputOffset], dstSlice, srcSlice, 1);
        } else {
            DataCopyPadExtParams<T> padParams{colsAlign != colsReal, 0,
                static_cast<uint8_t>(colsAlign - colsReal), static_cast<T>(0)};
            DataCopyExtParams params{static_cast<uint16_t>(rowsReal),
                static_cast<uint32_t>(colsReal * sizeof(T)),
                static_cast<uint32_t>((tiling_->inner - colsReal) * sizeof(T)), 0, 0};
            DataCopyPad(inputLocal, inputGm_[inputOffset], params, padParams);
        }
        phaseInputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void CompactRows(LocalTensor<T>& compactLocal,
                                       LocalTensor<T>& rawLocal, uint64_t inputOffset,
                                       uint64_t rowsReal, uint64_t colsAlign)
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint64_t phaseCapacity = CeilDivU64(rowsReal, blockElements);
        LocalTensor<uint32_t> maskLocal = phaseMaskBuffer_.Get<uint32_t>();
        if constexpr (PhaseMajor) {
            // raw 已按相位及组内行序排列，mask 也按相同顺序递进；一次 Counter
            // GatherMask 即可完成整块压缩，输出保持 phase-major 布局供 VA 地址表读取。
            const GatherMaskParams params{
                1, static_cast<uint16_t>(rowsReal),
                static_cast<uint16_t>(phaseRawPitch_ / blockElements), 1};
            uint64_t rsvdCnt = 0;
            GatherMask(compactLocal, rawLocal, maskLocal, true,
                       static_cast<uint32_t>(phaseRawPitch_), params, rsvdCnt);
            return;
        }
        const uint64_t firstPhase = GetInputPhase(inputOffset);
        for (uint64_t phase = 0; phase < blockElements; ++phase) {
            const uint64_t firstRow = (phase + blockElements - firstPhase) % blockElements;
            if (firstRow >= rowsReal) {
                continue;
            }
            const uint64_t groupRows = CeilDivU64(rowsReal - firstRow, blockElements);
            // 相位组每个 repeat 对应原始搬入中的间隔 blockElements 行，
            // repeat stride 必须跨过完整的相位周期，而不是只跨一行。
            const GatherMaskParams params{1, static_cast<uint16_t>(groupRows),
                                          static_cast<uint16_t>(phaseRawPitch_), 0};
            uint64_t rsvdCnt = 0;
            GatherMask(compactLocal[phase * phaseCapacity * colsAlign],
                       rawLocal[firstRow * phaseRawPitch_],
                       maskLocal[phase * blockElements], true,
                       static_cast<uint32_t>(phaseRawPitch_), params, rsvdCnt);
        }
    }

    __aicore__ inline void Compute(LocalTensor<T>& outputLocal, uint64_t rowStart,
                                   uint64_t rowsReal, uint64_t outputRows, uint64_t colsAlign,
                                   uint64_t batch, uint64_t inputRowStart, uint64_t colStart,
                                   uint64_t colsReal, bool phaseCompact)
    {
        LocalTensor<T> inputLocal = phaseInputQueue_.template DeQue<T>();
        const uint64_t inputOffset = GetInputOffset(batch, inputRowStart, colStart);
        if (phaseCompact) {
            // 复用压缩缓冲前先排空前一个 VA 转置；当前块的 MTE2 已通过 TQue 到达 V。
            PipeBarrier<PIPE_V>();
            LocalTensor<T> compactLocal = compactInputBuffer_.Get<T>();
            CompactRows(compactLocal, inputLocal, inputOffset, rowsReal, colsAlign);
            PipeBarrier<PIPE_V>();
            ComputeTransData(outputLocal, compactLocal, rowStart, rowsReal, outputRows,
                             colsAlign, inputOffset, true);
        } else {
            ComputeTransData(outputLocal, inputLocal, rowStart, rowsReal, outputRows,
                             colsAlign, inputOffset, false);
        }
        phaseInputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ComputeTransData(LocalTensor<T>& outputLocal, LocalTensor<T>& inputLocal,
                                            uint64_t rowStart, uint64_t rowsReal,
                                            uint64_t outputRows, uint64_t colsAlign,
                                            uint64_t inputOffset, bool phaseCompact)
    {
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(colsAlign / blockElements);
        TransDataTo5HDParams params{false, false, repeatTimes,
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : outputRows),
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        const uint64_t rowsAlign = AlignUpU64(rowsReal, TRANS_ADDR_COUNT);
        const uint64_t phaseCapacity = phaseCompact ? CeilDivU64(rowsReal, blockElements) : 0;
        const uint64_t firstPhase = phaseCompact ? GetInputPhase(inputOffset) : 0;
        // 两个 LocalTensor 在各自 UB buffer 内连续。取一次基址后按元素步长展开 VA 表，
        // 避免相位压缩路径为每个 16 行地址重复查询物理地址。
        const uint64_t inputBase = reinterpret_cast<uint64_t>(inputLocal.GetPhyAddr());
        const uint64_t outputBase = reinterpret_cast<uint64_t>(outputLocal.GetPhyAddr());
        for (uint64_t rowBase = 0; rowBase < rowsAlign; rowBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            uint64_t dstList[TRANS_ADDR_COUNT];
            for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
                const uint64_t safeRow = rowBase + row < rowsReal ? rowBase + row : 0;
                uint64_t sourceOffset = safeRow * colsAlign;
                if (phaseCompact) {
                    const uint64_t phase = (firstPhase + safeRow) % blockElements;
                    const uint64_t firstRow = (phase + blockElements - firstPhase) % blockElements;
                    const uint64_t ordinal = (safeRow - firstRow) / blockElements;
                    sourceOffset = phase * phaseCapacity * colsAlign + ordinal * colsAlign;
                }
                srcList[row] = inputBase + sourceOffset * sizeof(T);
            }
            if constexpr (sizeof(T) == 4) {
                for (uint64_t col = 0; col < blockElements; ++col) {
                    dstList[col * 2] = outputBase +
                        (col * outputRows + rowStart + rowBase) * sizeof(T);
                    dstList[col * 2 + 1] = outputBase +
                        (col * outputRows + rowStart + rowBase + blockElements) * sizeof(T);
                }
            } else {
                for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                    dstList[col] = outputBase + (col * outputRows + rowStart + rowBase) * sizeof(T);
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void CopyOut(uint64_t batch, uint64_t rowStart, uint64_t rowsReal,
                                   uint64_t colStart, uint64_t colsReal, uint64_t matrixRows,
                                   uint64_t outputRows)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        const uint64_t sourceStride =
            (outputRows * sizeof(T) - AlignUpU64(rowsReal * sizeof(T), BLOCK_BYTES)) / BLOCK_BYTES;
        const uint64_t outputOffset = batch * tiling_->inner * matrixRows + colStart * matrixRows + rowStart;
        const uint64_t destinationStrideBytes = (matrixRows - rowsReal) * sizeof(T);
        if (colsReal <= DATA_COPY_BLOCK_COUNT_MAX && rowsReal * sizeof(T) <= DATA_COPY_SHORT_PARAM_MAX &&
            sourceStride <= DATA_COPY_SHORT_PARAM_MAX && destinationStrideBytes <= DATA_COPY_SHORT_PARAM_MAX) {
            DataCopyParams params{static_cast<uint16_t>(colsReal),
                                  static_cast<uint16_t>(rowsReal * sizeof(T)),
                                  static_cast<uint16_t>(sourceStride),
                                  static_cast<uint16_t>(destinationStrideBytes)};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        } else {
            DataCopyExtParams params{static_cast<uint16_t>(colsReal),
                                     static_cast<uint32_t>(rowsReal * sizeof(T)),
                                     static_cast<uint32_t>(sourceStride),
                                     static_cast<uint32_t>(destinationStrideBytes), 0};
            DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

    const TransposeTilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    uint64_t blockIdx_{0};
    uint64_t blockDim_{1};
    uint64_t phaseRawPitch_{0};
    TQue<TPosition::VECIN, TwoSlot ? 2 : 1> phaseInputQueue_;
    TBuf<TPosition::VECIN> compactInputBuffer_;
    TBuf<TPosition::VECCALC> phaseMaskBuffer_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
};
#endif

template <typename T>
class KernelTransposeSingleTBufMultiStripe {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const TransposeTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_->blockDim;
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
    }

    __aicore__ inline void Process()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t tileCols = tiling_->tileCols;
        const uint64_t stripeBytes = matrixRows * tileCols * sizeof(T);
        pipe_->InitBuffer(inputBuffer_, stripeBytes);
        pipe_->InitBuffer(outputBuffer_, stripeBytes);
        LocalTensor<T> inputLocal = inputBuffer_.Get<T>();
        LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / blockElements);
        TransDataTo5HDParams params{false, false, repeatTimes,
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : matrixRows),
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        for (uint64_t task = blockIdx_; task < CeilDivU64(tiling_->inner, tileCols);
             task += blockDim_) {
            const uint64_t colStart = task * tileCols;
            DataCopyPad(inputLocal, inputGm_[colStart],
                        DataCopyExtParams{static_cast<uint16_t>(matrixRows),
                                          static_cast<uint32_t>(tileCols * sizeof(T)),
                                          static_cast<uint32_t>((tiling_->inner - tileCols) * sizeof(T)),
                                          0, 0},
                        padParams);
            PipeBarrier<PIPE_ALL>();
            for (uint64_t rowBase = 0; rowBase < matrixRows; rowBase += TRANS_ADDR_COUNT) {
                uint64_t srcList[TRANS_ADDR_COUNT];
                uint64_t dstList[TRANS_ADDR_COUNT];
                const uint64_t srcBase = reinterpret_cast<uint64_t>(
                    inputLocal[rowBase * tileCols].GetPhyAddr());
                for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
                    srcList[row] = srcBase + row * tileCols * sizeof(T);
                }
                const uint64_t dstBase = reinterpret_cast<uint64_t>(
                    outputLocal[rowBase].GetPhyAddr());
                if constexpr (sizeof(T) == 4) {
                    for (uint64_t col = 0; col < blockElements; ++col) {
                        const uint64_t colBase = dstBase + col * matrixRows * sizeof(T);
                        dstList[col * 2] = colBase;
                        dstList[col * 2 + 1] = colBase + blockElements * sizeof(T);
                    }
                } else {
                    for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                        dstList[col] = dstBase + col * matrixRows * sizeof(T);
                    }
                }
                TransDataTo5HD<T>(dstList, srcList, params);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(outputGm_[colStart * matrixRows], outputLocal, matrixRows * tileCols);
        }
    }

private:
    const TransposeTilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    uint64_t blockIdx_{0};
    uint64_t blockDim_{1};
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
};

// 保留非 16-bit 单矩阵路径的原有单条带实现，避免仅服务于 16-bit 短核调度的代码布局
// 影响其它 dtype 的已验证二进制。完整条带约束仍由 Host 在 key 10 选择前保证。
template <typename T>
class KernelTransposeSingleTBuf {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const TransposeTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
    }

    __aicore__ inline void Process()
    {
        const uint64_t matrixRows = tiling_->outer * tiling_->cols;
        const uint64_t tileCols = tiling_->tileCols;
        const uint64_t stripeBytes = matrixRows * tileCols * sizeof(T);
        const uint64_t colStart = blockIdx_ * tileCols;
        pipe_->InitBuffer(inputBuffer_, stripeBytes);
        pipe_->InitBuffer(outputBuffer_, stripeBytes);
        LocalTensor<T> inputLocal = inputBuffer_.Get<T>();
        LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[colStart],
                    DataCopyExtParams{static_cast<uint16_t>(matrixRows),
                                      static_cast<uint32_t>(tileCols * sizeof(T)),
                                      static_cast<uint32_t>((tiling_->inner - tileCols) * sizeof(T)),
                                      0, 0},
                    padParams);
        PipeBarrier<PIPE_ALL>();

        const uint64_t blockElements = BLOCK_BYTES / sizeof(T);
        const uint8_t repeatTimes = static_cast<uint8_t>(tileCols / blockElements);
        TransDataTo5HDParams params{false, false, repeatTimes,
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : matrixRows),
                                    static_cast<uint16_t>(repeatTimes <= 1 ? 0 : 1)};
        for (uint64_t rowBase = 0; rowBase < matrixRows; rowBase += TRANS_ADDR_COUNT) {
            uint64_t srcList[TRANS_ADDR_COUNT];
            uint64_t dstList[TRANS_ADDR_COUNT];
            const uint64_t srcBase = reinterpret_cast<uint64_t>(
                inputLocal[rowBase * tileCols].GetPhyAddr());
            for (uint64_t row = 0; row < TRANS_ADDR_COUNT; ++row) {
                srcList[row] = srcBase + row * tileCols * sizeof(T);
            }
            const uint64_t dstBase = reinterpret_cast<uint64_t>(
                outputLocal[rowBase].GetPhyAddr());
            if constexpr (sizeof(T) == 4) {
                for (uint64_t col = 0; col < blockElements; ++col) {
                    const uint64_t colBase = dstBase + col * matrixRows * sizeof(T);
                    dstList[col * 2] = colBase;
                    dstList[col * 2 + 1] = colBase + blockElements * sizeof(T);
                }
            } else {
                for (uint64_t col = 0; col < TRANS_ADDR_COUNT; ++col) {
                    dstList[col] = dstBase + col * matrixRows * sizeof(T);
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outputGm_[colStart * matrixRows], outputLocal, matrixRows * tileCols);
    }

private:
    const TransposeTilingData* tiling_{nullptr};
    TPipe* pipe_{nullptr};
    uint64_t blockIdx_{0};
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
};

} // namespace TransposeKernel

template <uint32_t schMode>
__global__ __aicore__ void transpose(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (schMode == TRANSPOSE_KEY_STRIDED_SINGLE_TBUF) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_SINGLE_TBUF, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe singleTBufPipe;
        if constexpr (sizeof(DTYPE_INPUTS) == 2) {
            TransposeKernel::KernelTransposeSingleTBufMultiStripe<DTYPE_INPUTS> singleTBufOp;
            singleTBufOp.Init(inputs, output, &tilingData, &singleTBufPipe);
            singleTBufOp.Process();
        } else {
            TransposeKernel::KernelTransposeSingleTBuf<DTYPE_INPUTS> singleTBufOp;
            singleTBufOp.Init(inputs, output, &tilingData, &singleTBufPipe);
            singleTBufOp.Process();
        }
        return;
    }
#if defined(TRANSPOSE_PHASE_COMPACT_FP32)
    if constexpr (schMode == TRANSPOSE_KEY_STRIDED_FIXED_PHASE_MAJOR_COMPACT) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_FIXED_PHASE_MAJOR_COMPACT, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe phaseMajorCompactPipe;
        TransposeKernel::KernelTransposeFixedPhaseCompact<DTYPE_INPUTS, true, true>
            phaseMajorCompactOp;
        phaseMajorCompactOp.Init(inputs, output, &tilingData, &phaseMajorCompactPipe);
        phaseMajorCompactOp.Process();
        return;
    }
    if constexpr (schMode == TRANSPOSE_KEY_STRIDED_FIXED_PHASE_COMPACT) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_FIXED_PHASE_COMPACT, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe phaseCompactPipe;
        TransposeKernel::KernelTransposeFixedPhaseCompact<DTYPE_INPUTS, false, false>
            phaseCompactOp;
        phaseCompactOp.Init(inputs, output, &tilingData, &phaseCompactPipe);
        phaseCompactOp.Process();
        return;
    }
#endif
    if constexpr (schMode == TRANSPOSE_KEY_STRIDED_FIXED_COMPACT) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_FIXED_COMPACT, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe fixedCompactPipe;
        TransposeKernel::KernelTransposeFixedCompact<DTYPE_INPUTS> fixedCompactOp;
        fixedCompactOp.Init(inputs, output, &tilingData, &fixedCompactPipe);
        fixedCompactOp.Process();
        return;
    }
    if constexpr (schMode == TRANSPOSE_KEY_STRIDED_FIXED_MATRIX) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_FIXED_MATRIX, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe fixedStridedPipe;
        TransposeKernel::KernelTranspose<DTYPE_INPUTS> fixedStridedOp;
        fixedStridedOp.InitStridedFixed(inputs, output, &tilingData, &fixedStridedPipe);
        fixedStridedOp.ProcessStridedMatrix();
        return;
    }
    AscendC::TPipe pipe;
    TransposeKernel::KernelTranspose<DTYPE_INPUTS> op;
    op.Init(inputs, output, &tilingData, &pipe);

    if constexpr (schMode == TRANSPOSE_KEY_COPY) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_COPY, KERNEL_TYPE_AIV_ONLY);
        op.ProcessCopy();
    } else if constexpr (schMode == TRANSPOSE_KEY_DENSE_MATRIX) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_DENSE_MATRIX, KERNEL_TYPE_AIV_ONLY);
        op.ProcessDenseMatrix();
    } else if constexpr (schMode == TRANSPOSE_KEY_STRIDED_MATRIX) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_MATRIX, KERNEL_TYPE_AIV_ONLY);
        op.ProcessStridedMatrix();
    } else if constexpr (schMode == TRANSPOSE_KEY_STRIDED_NONFIXED_AXIS) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_NONFIXED_AXIS, KERNEL_TYPE_AIV_ONLY);
        AscendC::ICachePreLoad(3);
        op.ProcessStridedMatrixImpl<false>();
    } else if constexpr (schMode == TRANSPOSE_KEY_STRIDED_SINGLE_MATRIX) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_STRIDED_SINGLE_MATRIX, KERNEL_TYPE_AIV_ONLY);
        op.ProcessStridedSingleMatrix();
    } else if constexpr (schMode == TRANSPOSE_KEY_BLOCK_SWAP) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_BLOCK_SWAP, KERNEL_TYPE_AIV_ONLY);
        op.ProcessBlockSwap();
    } else if constexpr (schMode == TRANSPOSE_KEY_GENERIC_AFFINE) {
        KERNEL_TASK_TYPE(TRANSPOSE_KEY_GENERIC_AFFINE, KERNEL_TYPE_AIV_ONLY);
        op.ProcessGenericAffine();
    }
}
