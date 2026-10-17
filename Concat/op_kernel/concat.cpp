#include "kernel_operator.h"
#include "adv_api/matmul/matmul_intf.h"
#include "kernel_operator_list_tensor_intf.h"

namespace ConcatKernel {
using namespace AscendC;

constexpr uint32_t SINGLE_BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t MAX_BLOCK_COUNT = 4095;
constexpr uint64_t MAX_BLOCK_LEN = 2097151;
constexpr uint64_t MAX_UINT32 = 0xffffffffULL;
constexpr int64_t MIN_INT32_VALUE = -2147483648LL;
constexpr int64_t MAX_INT32_VALUE = 2147483647LL;
constexpr uint32_t MAX_UINT16_VALUE = 65535;
constexpr uint32_t GATHER_BLOCK_ELEMENTS = 8;
constexpr uint32_t GATHER_MAX_REPEAT_TIMES = 255;
constexpr uint32_t GATHER_MAX_ELEMENTS_PER_CALL = 255 * 64;
constexpr uint32_t MICRO_GATHER_VF = 64;
constexpr uint32_t OFFSET_OUTPUT_ALIAS_SEED_ROWS = 2;
// C220 的 int32 二元向量指令每个完整 repeat 处理 64 个元素。
constexpr uint32_t DELTA_OFFSET_VECTOR_LANES = 64;
constexpr uint32_t GATHERB_BLOCKS_PER_REPEAT = 8;
constexpr uint32_t GATHERB_MAX_REPEATS = 255;
constexpr uint32_t GATHERB_MAX_BLOCKS_PER_CALL = GATHERB_BLOCKS_PER_REPEAT * GATHERB_MAX_REPEATS;
// 与 Tiling 协议中的单个 Gather plan 分区容量保持一致。
constexpr uint32_t GATHER_PLAN_INPUT_CAPACITY = 256;
constexpr uint32_t PREPARED_RECT_COMMANDS_PER_CORE = 4;
constexpr uint32_t PREPARED_RECT_WORDS_PER_COMMAND = 4;
constexpr uint64_t PREPARED_RECT_INPUT_L2_BYPASS_FLAG = 1ULL << 63;
constexpr uint64_t PREPARED_RECT_BLOCK_LEN_MASK = (1ULL << 31) - 1;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t MaxU64(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint64_t AlignUp32(uint64_t value)
{
    return (value + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
}

__aicore__ inline uint32_t AlignUpGatherElements(uint32_t value)
{
    return (value + GATHER_BLOCK_ELEMENTS - 1) / GATHER_BLOCK_ELEMENTS * GATHER_BLOCK_ELEMENTS;
}

__aicore__ inline uint32_t AlignUpGatherBlocks(uint32_t value)
{
    return (value + GATHERB_BLOCKS_PER_REPEAT - 1) / GATHERB_BLOCKS_PER_REPEAT *
        GATHERB_BLOCKS_PER_REPEAT;
}

__aicore__ inline void SplitBalanced(uint64_t total, uint64_t parts, uint64_t index,
                                     uint64_t& start, uint64_t& count)
{
    if (parts == 0 || index >= parts) {
        start = total;
        count = 0;
        return;
    }
    const uint64_t base = total / parts;
    const uint64_t remainder = total % parts;
    start = index * base + MinU64(index, remainder);
    count = base + static_cast<uint64_t>(index < remainder);
}

template <uint32_t CopyBufferNum, bool BypassInputL2 = false>
class KernelConcat {
public:
    __aicore__ inline explicit KernelConcat(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, uint32_t inputCount,
                                uint32_t dtypeSize, uint32_t tileBytes)
    {
        inputCount_ = inputCount;
        dtypeSize_ = dtypeSize;
        tileBytes_ = tileBytes;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        inputDesc_.SetShapeAddr(shapeBuf_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        if constexpr (BypassInputL2) {
            // ProcessRows 的各输入仅沿行方向单次流式读取，避免占用 L2 的数据分配路径。
            inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_.InitBuffer(copyQueue_, CopyBufferNum, tileBytes_);
    }

    __aicore__ inline void ProcessFlat(const ConcatFlatTilingData& tilingData)
    {
        const uint64_t coreStart = static_cast<uint64_t>(blockIdx_) * tilingData.coreChunkBytes;
        if (coreStart >= tilingData.totalBytes || tilingData.coreChunkBytes == 0) {
            return;
        }
        const uint64_t coreEnd = MinU64(tilingData.totalBytes, coreStart + tilingData.coreChunkBytes);
        uint32_t inputIdx = tilingData.coreStartInput[blockIdx_];
        uint64_t inputOffset = tilingData.coreStartInputOffset[blockIdx_];
        uint64_t outputOffset = coreStart;

        while (inputIdx < inputCount_ && outputOffset < coreEnd) {
            const uint64_t inputBytes = GetInputBytes(inputIdx);
            if (inputOffset >= inputBytes) {
                if (inputBytes != 0) {
                    inputOffset -= inputBytes;
                }
                ++inputIdx;
                continue;
            }
            const uint64_t copyBytes = MinU64(coreEnd - outputOffset, inputBytes - inputOffset);
            CopyContiguous(inputIdx, inputOffset, outputOffset, copyBytes);
            outputOffset += copyBytes;
            inputOffset = 0;
            ++inputIdx;
        }
    }

    __aicore__ inline void ProcessPreparedFlat(const __gm__ ConcatPreparedFlatTilingData* tilingData)
    {
        const uint32_t taskStart = blockIdx_ == 0 ? 0 : tilingData->coreTaskEnd[blockIdx_ - 1];
        const uint32_t taskEnd = tilingData->coreTaskEnd[blockIdx_];
        uint64_t outputOffset = tilingData->coreOutputStart[blockIdx_];
        for (uint32_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx) {
            const uint64_t copyBytes = tilingData->taskBytes[taskIdx];
            CopyContiguous(tilingData->taskInput[taskIdx], tilingData->taskInputOffset[taskIdx],
                           outputOffset, copyBytes);
            outputOffset += copyBytes;
        }
    }

    __aicore__ inline void ProcessPreparedFlatTwoCommands(
        const __gm__ ConcatPreparedFlatTwoCommandsTilingData* tilingData)
    {
        const uint32_t command = blockIdx_ * 2;
        CopyPreparedCommand(tilingData->commandInput[command],
                            tilingData->commandInputOffset[command],
                            tilingData->commandOutputOffset[command],
                            tilingData->commandBytes[command]);
        CopyPreparedCommand(tilingData->commandInput[command + 1],
                            tilingData->commandInputOffset[command + 1],
                            tilingData->commandOutputOffset[command + 1],
                            tilingData->commandBytes[command + 1]);
    }

    __aicore__ inline void ProcessPreparedRect(
        const __gm__ ConcatPreparedRectTilingData* tilingData)
    {
        const uint32_t command =
            blockIdx_ * PREPARED_RECT_COMMANDS_PER_CORE;
        CopyPreparedRectCommand(tilingData, command);
        CopyPreparedRectCommand(tilingData, command + 1);
        if ((tilingData->commandData[
                (command + 2) * PREPARED_RECT_WORDS_PER_COMMAND] >> 32) != 0) {
            CopyPreparedRectCommand(tilingData, command + 2);
        }
    }

    __aicore__ inline void ProcessRows(uint64_t outer, uint64_t outputRowBytes)
    {
        uint64_t rowStart = 0;
        uint64_t rowCount = 0;
        SplitBalanced(outer, blockNum_, blockIdx_, rowStart, rowCount);
        if (rowCount == 0) {
            return;
        }

        uint64_t outputCol = 0;
        for (uint32_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const uint64_t inputRowBytes = GetInputRowBytes(inputIdx, outer);
            if (inputRowBytes != 0) {
                CopyRect(inputIdx, rowStart * inputRowBytes,
                         rowStart * outputRowBytes + outputCol, rowCount,
                         inputRowBytes, outputRowBytes, inputRowBytes);
            }
            outputCol += inputRowBytes;
        }
    }

#if defined(CONCAT_ENABLE_ROW_BYTES_TILING_KEY)
    __aicore__ inline void ProcessRowsPrecomputed(
        const __gm__ ConcatRowPrecomputedTilingData* tilingData)
    {
        uint64_t rowStart = 0;
        uint64_t rowCount = 0;
        SplitBalanced(tilingData->outer, blockNum_, blockIdx_, rowStart, rowCount);
        if (rowCount == 0) {
            return;
        }

        uint64_t outputCol = 0;
        for (uint32_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            // 行宽由 Host 根据实际动态 shape 写入；零宽输入仍保持零搬运、零写回。
            const uint64_t inputRowBytes = tilingData->inputRowBytes[inputIdx];
            if (inputRowBytes != 0) {
                CopyRect(inputIdx, rowStart * inputRowBytes,
                    rowStart * tilingData->outputRowBytes + outputCol, rowCount,
                    inputRowBytes, tilingData->outputRowBytes, inputRowBytes);
            }
            outputCol += inputRowBytes;
        }
    }
#endif

    __aicore__ inline void ProcessGrid(const ConcatGridTilingData& tilingData)
    {
        if (tilingData.rowGroups == 0 || tilingData.colGroups == 0) {
            return;
        }
        const uint32_t rowGroup = blockIdx_ / tilingData.colGroups;
        const uint32_t colGroup = blockIdx_ % tilingData.colGroups;
        if (rowGroup >= tilingData.rowGroups) {
            return;
        }

        uint64_t rowStart = 0;
        uint64_t rowCount = 0;
        SplitBalanced(tilingData.outer, tilingData.rowGroups, rowGroup, rowStart, rowCount);
        const uint64_t colStart = colGroup * tilingData.colChunkBytes;
        if (rowCount == 0 || colStart >= tilingData.outputRowBytes || tilingData.colChunkBytes == 0) {
            return;
        }
        const uint64_t colEnd = MinU64(tilingData.outputRowBytes, colStart + tilingData.colChunkBytes);
        uint64_t inputColStart = 0;
        for (uint32_t inputIdx = 0; inputIdx < inputCount_ && inputColStart < colEnd; ++inputIdx) {
            const uint64_t inputRowBytes = GetInputRowBytes(inputIdx, tilingData.outer);
            const uint64_t inputColEnd = inputColStart + inputRowBytes;
            const uint64_t copyStart = MaxU64(colStart, inputColStart);
            const uint64_t copyEnd = MinU64(colEnd, inputColEnd);
            if (copyStart < copyEnd) {
                const uint64_t srcCol = copyStart - inputColStart;
                CopyRect(inputIdx, rowStart * inputRowBytes + srcCol,
                         rowStart * tilingData.outputRowBytes + copyStart, rowCount,
                         inputRowBytes, tilingData.outputRowBytes, copyEnd - copyStart);
            }
            inputColStart = inputColEnd;
        }
    }

    __aicore__ inline void ProcessGridSpan(const __gm__ ConcatGridSpanTilingData* tilingData)
    {
        const uint32_t rowGroups = tilingData->rowGroups;
        const uint32_t colGroups = tilingData->colGroups;
        if (rowGroups == 0 || colGroups == 0) {
            return;
        }
        const uint32_t rowGroup = blockIdx_ / colGroups;
        const uint32_t colGroup = blockIdx_ % colGroups;
        if (rowGroup >= rowGroups) {
            return;
        }

        const uint64_t outer = tilingData->outer;
        const uint64_t outputRowBytes = tilingData->outputRowBytes;
        const uint64_t colChunkBytes = tilingData->colChunkBytes;
        uint64_t rowStart = 0;
        uint64_t rowCount = 0;
        SplitBalanced(outer, rowGroups, rowGroup, rowStart, rowCount);
        const uint64_t colStart = static_cast<uint64_t>(colGroup) * colChunkBytes;
        if (rowCount == 0 || colStart >= outputRowBytes || colChunkBytes == 0) {
            return;
        }

        const uint64_t span = tilingData->coreSpan[colGroup];
        uint32_t inputIdx = static_cast<uint32_t>(span & 0xffffULL);
        uint64_t inputColStart = span >> 16;
        const uint64_t colEnd = MinU64(outputRowBytes, colStart + colChunkBytes);
        for (; inputIdx < inputCount_ && inputColStart < colEnd; ++inputIdx) {
            const uint64_t inputRowBytes = tilingData->inputRowBytes[inputIdx];
            const uint64_t inputColEnd = inputColStart + inputRowBytes;
            const uint64_t copyStart = MaxU64(colStart, inputColStart);
            const uint64_t copyEnd = MinU64(colEnd, inputColEnd);
            if (copyStart < copyEnd) {
                const uint64_t srcCol = copyStart - inputColStart;
                CopyRect(inputIdx, rowStart * inputRowBytes + srcCol,
                         rowStart * outputRowBytes + copyStart, rowCount,
                         inputRowBytes, outputRowBytes, copyEnd - copyStart);
            }
            inputColStart = inputColEnd;
        }
    }

    __aicore__ inline void ProcessInputLpt(const ConcatInputLptTilingData& tilingData)
    {
        const uint32_t taskStart = tilingData.coreTaskStart[blockIdx_];
        const uint32_t requestedEnd = taskStart + static_cast<uint32_t>(tilingData.coreTaskCount[blockIdx_]);
        const uint32_t taskEnd = requestedEnd < tilingData.taskCount ? requestedEnd : tilingData.taskCount;
        for (uint32_t taskIdx = taskStart; taskIdx < taskEnd; ++taskIdx) {
            const uint32_t taskInput = tilingData.taskInput[taskIdx];
            if (taskInput >= inputCount_) {
                continue;
            }
            const uint64_t inputRowBytes = tilingData.taskRowBytes[taskIdx];
            const uint64_t rowStart = tilingData.taskRowStart[taskIdx];
            const uint64_t rowCount = tilingData.taskRowCount[taskIdx];
            if (inputRowBytes == 0 || rowCount == 0) {
                continue;
            }
            CopyRect(taskInput, rowStart * inputRowBytes,
                     rowStart * tilingData.outputRowBytes + tilingData.taskOutputCol[taskIdx], rowCount,
                     inputRowBytes, tilingData.outputRowBytes, inputRowBytes);
        }
    }

private:
    __aicore__ inline void CopyPreparedRectCommand(
        const __gm__ ConcatPreparedRectTilingData* tilingData, uint32_t command)
    {
        const uint32_t word = command * PREPARED_RECT_WORDS_PER_COMMAND;
        const uint64_t header = tilingData->commandData[word];
        const uint64_t strides = tilingData->commandData[word + 3];
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(header & 0xffffU));
        CopyTile(tilingData->commandData[word + 1], tilingData->commandData[word + 2],
                 (header >> 16) & 0xffffU, header >> 32,
                 strides & MAX_UINT32, strides >> 32);
    }

    __aicore__ inline void CopyPreparedCommand(uint32_t inputIdx, uint64_t srcOffset,
                                               uint64_t dstOffset, uint32_t copyBytes)
    {
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(inputIdx));
        CopyTile(srcOffset, dstOffset, 1, copyBytes, 0, 0);
    }

    __aicore__ inline uint64_t GetInputBytes(uint32_t inputIdx)
    {
        inputList_.GetDesc(inputDesc_, inputIdx);
        uint64_t elementCount = 1;
        const uint32_t dimCount = inputDesc_.GetDim();
        for (uint32_t dim = 0; dim < dimCount; ++dim) {
            elementCount *= inputDesc_.GetShape(dim);
        }
        return elementCount * dtypeSize_;
    }

    __aicore__ inline uint64_t GetInputRowBytes(uint32_t inputIdx, uint64_t outer)
    {
        return outer == 0 ? 0 : GetInputBytes(inputIdx) / outer;
    }

    __aicore__ inline void CopyContiguous(uint32_t inputIdx, uint64_t srcOffset,
                                          uint64_t dstOffset, uint64_t copyBytes)
    {
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(inputIdx));
        uint64_t copied = 0;
        while (copied < copyBytes) {
            const uint64_t blockLen = MinU64(MinU64(copyBytes - copied, tileBytes_), MAX_BLOCK_LEN);
            CopyTile(srcOffset + copied, dstOffset + copied, 1, blockLen, 0, 0);
            copied += blockLen;
        }
    }

    __aicore__ inline void CopyRect(uint32_t inputIdx, uint64_t srcOffset,
                                    uint64_t dstOffset, uint64_t rows,
                                    uint64_t srcRowBytes, uint64_t dstRowBytes,
                                    uint64_t copyBytes)
    {
        if (rows == 0 || copyBytes == 0 || tileBytes_ == 0) {
            return;
        }
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(inputIdx));

        uint64_t colOffset = 0;
        while (colOffset < copyBytes) {
            const uint64_t blockLen = MinU64(MinU64(copyBytes - colOffset, tileBytes_), MAX_BLOCK_LEN);
            const uint64_t localRowBytes = AlignUp32(blockLen);
            uint64_t rowsPerTile = tileBytes_ / localRowBytes;
            rowsPerTile = MinU64(rowsPerTile, MAX_BLOCK_COUNT);
            const uint64_t srcStride = srcRowBytes - blockLen;
            const uint64_t dstStride = dstRowBytes - blockLen;
            if (srcStride > MAX_UINT32 || dstStride > MAX_UINT32) {
                rowsPerTile = 1;
            }

            uint64_t rowOffset = 0;
            while (rowOffset < rows) {
                const uint64_t blockCount = MinU64(rows - rowOffset, rowsPerTile);
                const uint64_t copySrcOffset = srcOffset + rowOffset * srcRowBytes + colOffset;
                const uint64_t copyDstOffset = dstOffset + rowOffset * dstRowBytes + colOffset;
                CopyTile(copySrcOffset, copyDstOffset, blockCount, blockLen,
                         blockCount == 1 ? 0 : srcStride,
                         blockCount == 1 ? 0 : dstStride);
                rowOffset += blockCount;
            }
            colOffset += blockLen;
        }
    }

    __aicore__ inline void CopyTile(uint64_t srcOffset, uint64_t dstOffset,
                                    uint64_t blockCount, uint64_t blockLen,
                                    uint64_t srcStride, uint64_t dstStride)
    {
        LocalTensor<int8_t> local = copyQueue_.template AllocTensor<int8_t>();
        const DataCopyExtParams copyInParams{
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen),
            static_cast<uint32_t>(srcStride), 0, 0};
        const DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, inputGm_[srcOffset], copyInParams, padParams);
        copyQueue_.EnQue(local);

        local = copyQueue_.template DeQue<int8_t>();
        const DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen),
            0, static_cast<uint32_t>(dstStride), 0};
        DataCopyPad(outputGm_[dstOffset], local, copyOutParams);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe& pipe_;
    uint32_t inputCount_{0};
    uint32_t dtypeSize_{0};
    uint32_t tileBytes_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint64_t shapeBuf_[8];
    TensorDesc<int8_t> inputDesc_;
    ListTensorDesc inputList_;
    GlobalTensor<int8_t> inputGm_;
    GlobalTensor<int8_t> outputGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, CopyBufferNum> copyQueue_;
};

#if defined(CONCAT_ENABLE_ROW_PACKED_UB_TILING_KEY)
// FP16 小型 ROW 的完整输出打包路径。每个核先把输入段线性搬入 source UB，
// 用字节 offset 的 Gather 组装完整行，再以一次二维 MTE3 写回，避免每个输入段
// 各自触发一次输出 DMA。Host 只在完整均分、UB/DMA/Gather 约束成立时选择该 key。
class KernelConcatPackedRow {
public:
    __aicore__ inline explicit KernelConcatPackedRow(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatPackedRowTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if (tilingData_ == nullptr || blockNum_ == 0 || blockIdx_ >= blockNum_) {
            return;
        }
        inputCount_ = tilingData_->inputCount;
        rowsPerCore_ = tilingData_->rowsPerCore;
        outputRowElements_ = tilingData_->outputRowElements;
        outputRowBytes_ = tilingData_->outputRowBytes;
        sourceBufferBytes_ = tilingData_->sourceBufferBytes;
        outputBufferBytes_ = tilingData_->outputBufferBytes;
        offsetBufferBytes_ = tilingData_->offsetBufferBytes;
        rowStart_ = static_cast<uint64_t>(blockIdx_) * rowsPerCore_;
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        valid_ = tilingData_->dtypeSize == sizeof(half) && inputCount_ != 0 &&
            inputCount_ <= CONCAT_PACKED_ROW_INPUT_CAPACITY && rowsPerCore_ >= 2 &&
            rowStart_ < tilingData_->outer && rowsPerCore_ <= tilingData_->outer - rowStart_ &&
            outputRowElements_ != 0 && outputRowBytes_ ==
                static_cast<uint64_t>(outputRowElements_) * sizeof(half) &&
            sourceBufferBytes_ != 0 && outputBufferBytes_ != 0 && offsetBufferBytes_ != 0 &&
            outputRowBytes_ <= MAX_BLOCK_LEN && outputRowBytes_ % UB_BLOCK_BYTES == 0 &&
            static_cast<uint64_t>(rowsPerCore_) * outputRowElements_ <=
                PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL;
        if (!valid_) {
            return;
        }
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
        pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        const uint32_t alignment = UB_BLOCK_BYTES / sizeof(half);
        const uint32_t alignedOutputElements = AlignUpElements(outputRowElements_, alignment);
        const uint64_t matrixElements = static_cast<uint64_t>(rowsPerCore_) *
            alignedOutputElements;
        if (alignedOutputElements == 0 || matrixElements > MAX_UINT32 ||
            matrixElements * sizeof(half) > outputBufferBytes_ ||
            matrixElements * sizeof(int32_t) > offsetBufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<half> source = sourceBuffer_.Get<half>();
        LocalTensor<half> output = outputBuffer_.Get<half>();
        LocalTensor<int32_t> offsetStorage = offsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetStorage.ReinterpretCast<uint32_t>();
        source.SetSize(sourceBufferBytes_ / sizeof(half));
        output.SetSize(static_cast<uint32_t>(matrixElements));
        offsetStorage.SetSize(static_cast<uint32_t>(matrixElements));

        CopyInputs(source);
        GenerateOffsets(offsetStorage, alignedOutputElements, alignment);
        SyncMte2ToV();
        GatherRows(output, source, offsets, static_cast<uint32_t>(matrixElements));
        SyncVToMte3();

        const uint64_t paddingBytes = static_cast<uint64_t>(
            alignedOutputElements - outputRowElements_) * sizeof(half);
        if (paddingBytes % UB_BLOCK_BYTES != 0 ||
            paddingBytes / UB_BLOCK_BYTES > MAX_UINT32) {
            valid_ = false;
            return;
        }
        const DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(rowsPerCore_), static_cast<uint32_t>(outputRowBytes_),
            static_cast<uint32_t>(paddingBytes / UB_BLOCK_BYTES), 0, 0};
        DataCopyPad(outputGm_[rowStart_ * outputRowElements_], output, copyOutParams);
    }

private:
    static constexpr uint32_t CONCAT_PACKED_ROW_INPUT_CAPACITY = 256;
    static constexpr uint32_t PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL = 255 * 64;

    __aicore__ inline uint32_t AlignUpElements(uint32_t value, uint32_t alignment) const
    {
        return alignment == 0 ? value :
            (value + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline void FillConstant(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & 7U;
        if (lane != 0) {
            const uint32_t head = MinU32(count, 8U - lane);
            uint64_t mask[1] = {((1ULL << head) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += head;
            count -= head;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddConstant(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & 7U;
        if (lane != 0) {
            const uint32_t head = MinU32(count, 8U - lane);
            uint64_t mask[1] = {((1ULL << head) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                mask, 1, {1, 1, 8, 8});
            start += head;
            count -= head;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value,
                static_cast<int32_t>(count));
        }
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void CopyInputs(const LocalTensor<half>& source)
    {
        uint32_t cursor = 0;
        const uint32_t alignment = UB_BLOCK_BYTES / sizeof(half);
        const DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        for (uint32_t input = 0; input < inputCount_; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t payloadElements = static_cast<uint64_t>(rowsPerCore_) * inputElements;
            const uint64_t payloadBytes = payloadElements * sizeof(half);
            const uint32_t slotElements = AlignUpElements(
                static_cast<uint32_t>(payloadElements), alignment);
            if (payloadElements > MAX_UINT32 || payloadBytes > MAX_BLOCK_LEN ||
                cursor > source.GetSize() || slotElements > source.GetSize() - cursor) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<half>(input));
                const uint64_t sourceOffset = static_cast<uint64_t>(rowStart_) * inputElements;
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(payloadBytes), 0, 0, 0};
                DataCopyPad(source[cursor], inputGm_[sourceOffset],
                    copyInParams, padParams);
            }
            cursor += slotElements;
        }
        if (cursor == 0 || cursor > source.GetSize()) {
            valid_ = false;
        }
    }

    __aicore__ inline void GenerateOffsets(const LocalTensor<int32_t>& offsets,
        uint32_t alignedOutputElements, uint32_t alignment)
    {
        const uint32_t rows = rowsPerCore_;
        CreateVecIndex(offsets, static_cast<int32_t>(0), alignedOutputElements);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(half)),
            static_cast<int32_t>(alignedOutputElements));
        LocalTensor<int32_t> strides = offsets[(rows - 1U) * alignedOutputElements];
        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = 0; input < inputCount_; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t adjustment = (static_cast<int64_t>(sourceBase) -
                    static_cast<int64_t>(outputPrefix)) * sizeof(half);
                const uint64_t rowStride = static_cast<uint64_t>(inputElements) * sizeof(half);
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE ||
                    rowStride > MAX_INT32_VALUE || outputPrefix > tilingData_->outputRowElements ||
                    inputElements > tilingData_->outputRowElements - outputPrefix) {
                    valid_ = false;
                    return;
                }
                AddConstant(offsets, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(adjustment));
                FillConstant(strides, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(rowStride));
            }
            outputPrefix += inputElements;
            const uint64_t payloadElements = static_cast<uint64_t>(rows) * inputElements;
            sourceBase += AlignUpElements(static_cast<uint32_t>(payloadElements), alignment);
            if (sourceBase > MAX_UINT32) {
                valid_ = false;
                return;
            }
        }
        if (outputPrefix != tilingData_->outputRowElements) {
            valid_ = false;
            return;
        }
        FillConstant(offsets, static_cast<uint32_t>(outputPrefix),
            alignedOutputElements - static_cast<uint32_t>(outputPrefix), 0);
        FillConstant(strides, static_cast<uint32_t>(outputPrefix),
            alignedOutputElements - static_cast<uint32_t>(outputPrefix), 0);
        for (uint32_t row = 1; row + 1 < rows; ++row) {
            Add(offsets[row * alignedOutputElements], offsets[(row - 1) * alignedOutputElements],
                strides, static_cast<int32_t>(alignedOutputElements));
        }
        Add(strides, offsets[(rows - 2U) * alignedOutputElements], strides,
            static_cast<int32_t>(alignedOutputElements));
    }

    __aicore__ inline void GatherRows(const LocalTensor<half>& output,
        const LocalTensor<half>& source, const LocalTensor<uint32_t>& offsets,
        uint32_t count)
    {
        for (uint32_t done = 0; done < count; done += PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL) {
            const uint32_t current = MinU32(PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL,
                count - done);
            Gather(output[done], source, offsets[done], 0, current);
        }
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatPackedRowTilingData* tilingData_{nullptr};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t inputCount_{0};
    uint32_t rowsPerCore_{0};
    uint32_t outputRowElements_{0};
    uint64_t outputRowBytes_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint32_t offsetBufferBytes_{0};
    uint64_t rowStart_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
};
#endif

template <bool OutputPingPong, bool SourcePingPong, bool StagedSource = false,
          bool RowMajorStaged = false, bool SplitGatherOutput = false,
          bool CachedStagedOffsets = false, bool FullStageOffsets = false,
          bool CompactStageOffsets = false, bool RegisterGather = false,
          bool RollingStagedOffsets = false, bool DeltaStagedOffsets = false,
          bool InitialInputOffsetOverlap = false, bool PartitionPipeline = false,
          bool SharedOffsetSeed = false, bool CoreSwizzle = false, bool GroupPhase = false,
          bool WavefrontGather = false, bool DirectWavefrontGather = false,
          bool TwoPlaneSource = false, bool AdaptiveGroupBatch = false,
          bool OffsetOutputAlias = false, bool MixedAiv = false,
          bool SuffixPrefetch = false>
class KernelConcatRowGather {
public:
    __aicore__ inline explicit KernelConcatRowGather(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatRowGatherTilingData* tilingData,
                                GM_ADDR workspace = nullptr)
    {
        tilingData_ = tilingData;
        inputCount_ = tilingData_->inputCount;
        outputRowElements_ = tilingData_->outputRowElements;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if constexpr (MixedAiv) {
            // C220 的 GetBlockIdx() 已经返回物理 block 与 AIV subblock 展平后的全局
            // 编号；MIX 这里只需把参与同步/边界计算的 block 数扩展到 AIV 总数。
            const uint32_t subBlockNum = static_cast<uint32_t>(GetSubBlockNum());
            const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
            if (subBlockNum != 2 || subBlockIdx >= subBlockNum ||
                blockNum_ > MAX_UINT32 / subBlockNum) {
                valid_ = false;
                return;
            }
            blockNum_ *= subBlockNum;
        }
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        if constexpr (SharedOffsetSeed) {
            sharedSeedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
        }

        const uint32_t highCoreCount = tilingData_->highCoreCount;
        uint32_t logicalBlockIdx = blockIdx_;
        if constexpr (CoreSwizzle) {
            // 偶、奇物理核分别领取前、后半段逻辑行块。该双射只依赖运行时 block 数，
            // 不改变任一逻辑块的连续性、行数或输入/输出地址公式。
            const uint32_t firstHalf = (blockNum_ + 1U) / 2U;
            logicalBlockIdx = (blockIdx_ & 1U) == 0 ? blockIdx_ / 2U :
                firstHalf + blockIdx_ / 2U;
        }
        if (logicalBlockIdx < highCoreCount) {
            rowStart_ = static_cast<uint64_t>(logicalBlockIdx) * tilingData_->rowsPerCore1;
            rowCount_ = tilingData_->rowsPerCore1;
            planIndex_ = 1;
        } else {
            rowStart_ = static_cast<uint64_t>(highCoreCount) * tilingData_->rowsPerCore1 +
                static_cast<uint64_t>(logicalBlockIdx - highCoreCount) * tilingData_->rowsPerCore0;
            rowCount_ = tilingData_->rowsPerCore0;
            planIndex_ = 0;
        }
        if (planIndex_ == 0) {
            batchRows_ = tilingData_->batchRows0;
            sourceBatchRows_ = tilingData_->sourceBatchRows0;
            groupCount_ = tilingData_->groupCount0;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes0;
            outputBufferBytes_ = tilingData_->outputBufferBytes0;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes0;
            sourcePart0BufferBytes_ = tilingData_->sourcePart0BufferBytes0;
            sourcePart1BufferBytes_ = tilingData_->sourcePart1BufferBytes0;
            outputPart0BufferBytes_ = tilingData_->outputPart0BufferBytes0;
            outputPart1BufferBytes_ = tilingData_->outputPart1BufferBytes0;
        } else {
            batchRows_ = tilingData_->batchRows1;
            sourceBatchRows_ = tilingData_->sourceBatchRows1;
            groupCount_ = tilingData_->groupCount1;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes1;
            outputBufferBytes_ = tilingData_->outputBufferBytes1;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes1;
            sourcePart0BufferBytes_ = tilingData_->sourcePart0BufferBytes1;
            sourcePart1BufferBytes_ = tilingData_->sourcePart1BufferBytes1;
            outputPart0BufferBytes_ = tilingData_->outputPart0BufferBytes1;
            outputPart1BufferBytes_ = tilingData_->outputPart1BufferBytes1;
        }

        valid_ = highCoreCount <= blockNum_ && rowStart_ <= tilingData_->outer &&
                 rowCount_ <= tilingData_->outer - rowStart_ &&
                 tilingData_->dtypeSize == sizeof(int32_t) && rowCount_ != 0 &&
                 batchRows_ != 0 && sourceBatchRows_ >= batchRows_ && groupCount_ != 0 &&
                 sourceBufferBytes_ != 0 &&
                 outputBufferBytes_ != 0 && offsetBufferBytes_ != 0;
        if constexpr (PartitionPipeline) {
            valid_ = valid_ && sourceBatchRows_ == batchRows_ &&
                sourcePart0BufferBytes_ != 0 && sourcePart1BufferBytes_ != 0 &&
                outputPart0BufferBytes_ != 0 && outputPart1BufferBytes_ != 0;
        }
        if constexpr (WavefrontGather || DirectWavefrontGather) {
            valid_ = valid_ && sourceBatchRows_ == batchRows_ &&
                sourcePart0BufferBytes_ != 0 && sourcePart1BufferBytes_ != 0 &&
                outputPart0BufferBytes_ != 0 && outputPart1BufferBytes_ != 0;
        }
        if constexpr (DirectWavefrontGather) {
            // 一张 staged offset 由 batch 个 64-lane repeat 和一行 stride 组成。Host 已
            // 校验容量与 Gather 的 uint8/uint16 参数范围，这里继续做本地防御。
            const uint64_t stagedOffsetBytes = static_cast<uint64_t>(batchRows_ + 1) *
                MICRO_GATHER_VF * sizeof(int32_t);
            valid_ = valid_ && batchRows_ <= GATHER_MAX_REPEAT_TIMES &&
                stagedOffsetBytes <= outputPart1BufferBytes_;
        }
        if constexpr (SharedOffsetSeed) {
            valid_ = valid_ && workspace != nullptr && batchRows_ >= 2;
        }
        if constexpr (TwoPlaneSource) {
            // Host 已按两张 source 平面核算 UB；这里避免运行时乘法溢出导致错误的短分配。
            valid_ = valid_ && sourceBatchRows_ == batchRows_ &&
                sourceBufferBytes_ <= MAX_UINT32 / 2;
        }
        if constexpr (SourcePingPong) {
            // 双 source 槽合并在一张连续 VECIN TBuf 中分配；设备侧同样约束总长度和
            // 第二槽起点，避免异常 Tiling 导致 uint32_t 溢出或破坏 32 B 对齐。
            valid_ = valid_ && sourceBufferBytes_ <= MAX_UINT32 / 2 &&
                sourceBufferBytes_ % sizeof(int32_t) == 0;
        }
        if constexpr (OffsetOutputAlias) {
            // key43 用 outputBuffer 承载每批 offset 与 Gather 结果，VECCALC 仅保留
            // base/stride 两行；source 布局必须与当前 output 批严格一致。
            valid_ = valid_ && sourceBatchRows_ == batchRows_ &&
                batchRows_ >= 2;
        }
        if constexpr (SuffixPrefetch) {
            // key47 只借用同一张 source TBuf 的未分配尾部；完整 source 批、后缀批和
            // 后缀偏移均以 int32 元素计，因此不允许非元素对齐的异常 Tiling。
            valid_ = valid_ && sourceBatchRows_ == batchRows_ &&
                sourceBufferBytes_ % sizeof(int32_t) == 0 && batchRows_ >= 2;
        }
        if (!valid_) {
            return;
        }
        if constexpr (DirectWavefrontGather) {
            // key39：source0/source1 与完整 output 保持 key38 的所有权关系，offset
            // 仅保留一个 64-lane × batch staging 区，直接服务跨行距 Gather。
            pipe_.InitBuffer(sourcePart0Buffer_, sourcePart0BufferBytes_);
            pipe_.InitBuffer(sourcePart1Buffer_, sourcePart1BufferBytes_);
            pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
            pipe_.InitBuffer(offsetPart0Buffer_, outputPart1BufferBytes_);
        } else if constexpr (WavefrontGather) {
            // key38：两个连续 source 槽、完整 output，以及一个可复用的 dense partial
            // output/offset 对。Host 已将这五块同时纳入 UB 账本。
            pipe_.InitBuffer(sourcePart0Buffer_, sourcePart0BufferBytes_);
            pipe_.InitBuffer(sourcePart1Buffer_, sourcePart1BufferBytes_);
            pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
            pipe_.InitBuffer(outputPart0Buffer_, outputPart0BufferBytes_);
            pipe_.InitBuffer(offsetPart0Buffer_, outputPart1BufferBytes_);
        } else if constexpr (PartitionPipeline) {
            pipe_.InitBuffer(sourcePart0Buffer_, sourcePart0BufferBytes_);
            pipe_.InitBuffer(sourcePart1Buffer_, sourcePart1BufferBytes_);
            pipe_.InitBuffer(outputPart0Buffer_, outputPart0BufferBytes_);
            pipe_.InitBuffer(outputPart1Buffer_, outputPart1BufferBytes_);
            pipe_.InitBuffer(offsetPart0Buffer_, outputPart0BufferBytes_);
            pipe_.InitBuffer(offsetPart1Buffer_, outputPart1BufferBytes_);
        } else {
            // 将两个 source 槽放入一张连续 VECIN TBuf。C5 的满 UB 路径若分别申请两个
            // TBuf，会额外消耗运行时分配对齐空间并触发 VEC UB 越界；两个槽仍由独立
            // event 和 32 B 对齐的首地址隔离，不改变 DMA/Gather 的数据所有权。
            if constexpr (SourcePingPong) {
                if (!pipe_.InitBuffer(sourcePingBuffer_, sourceBufferBytes_ * 2)) {
                    valid_ = false;
                    return;
                }
            } else if constexpr (TwoPlaneSource) {
                pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_ * 2);
            } else {
                pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
            }
            if constexpr (OutputPingPong) {
                // 两个等长输出槽只处理偶数行；奇数 batch 留一行容量给 offset/source，避免增加 UB。
                pipelineBatchRows_ = batchRows_ & ~1U;
                if (pipelineBatchRows_ == 0 || outputBufferBytes_ % batchRows_ != 0) {
                    valid_ = false;
                    return;
                }
                const uint32_t bytesPerRow = outputBufferBytes_ / batchRows_;
                outputSlotBytes_ = bytesPerRow * (pipelineBatchRows_ / 2);
                if (outputSlotBytes_ == 0) {
                    valid_ = false;
                    return;
                }
                pipe_.InitBuffer(outputQueue_, DOUBLE_BUFFER_NUM, outputSlotBytes_);
            } else {
                pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
            }
            pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
        }
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        if constexpr (DirectWavefrontGather) {
            ProcessDirectWavefront();
            return;
        }
        if constexpr (WavefrontGather) {
            ProcessWavefront();
            return;
        }
        if constexpr (SharedOffsetSeed) {
            PrepareSharedOffsetSeeds();
            if (!valid_) {
                return;
            }
        }

        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        bool outputBusy = false;
        // key42 的各 group 只在循环内临时覆盖行数。常规 key7 实例在编译期消除这两个
        // 分支，保留原有统一 batch 的二进制执行路径。
        const uint32_t baseBatchRows = batchRows_;
        const uint32_t baseSourceBatchRows = sourceBatchRows_;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        uint32_t groupPhase = 0;
        if constexpr (GroupPhase) {
            // 同一逻辑 group 的输入段会在所有核上同时变热。按物理核循环错开起点，
            // 不改变任一核的 group 集合、UB 布局或输出地址，仅改变跨核访问时序。
            groupPhase = blockIdx_ % groupCount_;
        }
        for (uint32_t groupOrder = 0; groupOrder < groupCount_; ++groupOrder) {
            uint32_t group = groupOrder;
            if constexpr (GroupPhase) {
                group += groupPhase;
                if (group >= groupCount_) {
                    group -= groupCount_;
                }
                inputStart = group == 0 ? 0 : tilingData_->groupInputEnd[planBase + group - 1];
                outputStart = 0;
                for (uint32_t prefix = 0; prefix < group; ++prefix) {
                    const uint32_t prefixElements = tilingData_->groupElements[planBase + prefix];
                    if (prefixElements > MAX_UINT32 - outputStart) {
                        return;
                    }
                    outputStart += prefixElements;
                }
            }
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t inputSplit = tilingData_->groupInputSplit[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                groupSourceElements == 0) {
                return;
            }
            if constexpr (AdaptiveGroupBatch) {
                const uint32_t groupBatchRows = tilingData_->groupBatchRows[planGroup];
                const uint32_t alignedElements = AlignUpGatherElements(groupElements);
                const uint64_t matrixElements = static_cast<uint64_t>(groupBatchRows) *
                    alignedElements;
                const uint64_t matrixBytes = matrixElements * sizeof(int32_t);
                const uint64_t sourceBytes = static_cast<uint64_t>(groupSourceElements) *
                    sizeof(int32_t);
                // Host 仅会提交不超过 key7 原始三块 UB 最大容量的行数；在设备侧重复
                // 验证，防止异常 Tiling 使 SetSize 或 DMA 访问越界。
                if (baseSourceBatchRows != baseBatchRows || groupBatchRows < baseBatchRows ||
                    groupBatchRows > rowCount_ || alignedElements == 0 ||
                    matrixElements > MAX_UINT32 || sourceBytes > sourceBufferBytes_ ||
                    matrixBytes > outputBufferBytes_ || matrixBytes > offsetBufferBytes_) {
                    valid_ = false;
                    return;
                }
                batchRows_ = groupBatchRows;
                sourceBatchRows_ = groupBatchRows;
            }
            if constexpr (PartitionPipeline) {
                if (inputSplit <= inputStart || inputSplit >= inputEnd) {
                    return;
                }
            }
            bool firstBatchPreloaded = false;
            if constexpr (!PartitionPipeline) {
                if constexpr (SharedOffsetSeed) {
                    // seed 的 MTE2 完成后，首批连续 source 可立即发射；offset 行传播与
                    // 该 MTE2 并行，Gather 在同一 V pipe 中自然等待表构造结束。
                    LoadSharedOffsetSeed(planIndex_, group, groupElements);
                    if (!valid_) {
                        return;
                    }
                    const uint32_t firstBatchRows = static_cast<uint32_t>(MinU64(
                        batchRows_, rowCount_));
                    LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
                    source.SetSize(groupSourceElements);
                    CopyInputs(inputStart, inputEnd, rowStart_, firstBatchRows, source);
                    firstBatchPreloaded = true;
                } else if constexpr (TwoPlaneSource) {
                    // 首个 stage 若有两张完整平面，用一次二维 DMA 搬入；offset 表只依赖
                    // 单平面的相对布局，因此可与该 DMA 并行生成并被两个平面复用。
                    LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
                    const bool hasPlanePair = rowCount_ >= 2 * batchRows_;
                    const uint64_t sourceElements = hasPlanePair ?
                        static_cast<uint64_t>(groupSourceElements) * 2 : groupSourceElements;
                    if (sourceElements > MAX_UINT32) {
                        valid_ = false;
                        return;
                    }
                    source.SetSize(static_cast<uint32_t>(sourceElements));
                    if (hasPlanePair) {
                        CopyInputsTwoPlanes(inputStart, inputEnd, rowStart_, source,
                            groupSourceElements);
                    } else {
                        const uint32_t firstBatchRows = static_cast<uint32_t>(MinU64(
                            batchRows_, rowCount_));
                        CopyInputs(inputStart, inputEnd, rowStart_, firstBatchRows, source);
                    }
                    GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                    firstBatchPreloaded = true;
                } else if constexpr (SuffixPrefetch) {
                    // key47 的第一批仍是完整标准 source 布局。这样既能复用已有 offset
                    // 表，又能在首个 Gather 期间把下一批连续后缀写入未使用的尾部。
                    const uint64_t sourceCapacityElements =
                        sourceBufferBytes_ / sizeof(int32_t);
                    if (sourceCapacityElements == 0 || sourceCapacityElements > MAX_UINT32) {
                        valid_ = false;
                        return;
                    }
                    const uint32_t firstBatchRows = static_cast<uint32_t>(MinU64(
                        batchRows_, rowCount_));
                    LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
                    source.SetSize(static_cast<uint32_t>(sourceCapacityElements));
                    CopyInputs(inputStart, inputEnd, rowStart_, firstBatchRows, source);
                    GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                    firstBatchPreloaded = true;
                } else if constexpr (InitialInputOffsetOverlap) {
                    // 首批 source 与 offset 位于独立 UB，先发起 MTE2 后建表，直到 Gather 前再同步。
                    const uint32_t firstBatchRows = static_cast<uint32_t>(MinU64(
                        batchRows_, rowCount_));
                    LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
                    source.SetSize(groupSourceElements);
                    CopyInputs(inputStart, inputEnd, rowStart_, firstBatchRows, source);
                    GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                    firstBatchPreloaded = true;
                } else if constexpr (RowMajorStaged) {
                    GenerateRowMajorOffsets(inputStart, inputEnd, groupElements);
                    if (!valid_) {
                        return;
                    }
                } else if constexpr (OffsetOutputAlias && !SourcePingPong) {
                    PrepareOffsetOutputAliasSeeds(inputStart, inputEnd, groupElements);
                    if (!valid_) {
                        return;
                    }
                } else if constexpr (!StagedSource && !RegisterGather && !RollingStagedOffsets &&
                                     !DeltaStagedOffsets && !SuffixPrefetch) {
                    GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                }
            }
            if constexpr (SuffixPrefetch) {
                ProcessGroupSuffixPrefetch(inputStart, inputEnd, outputStart, groupElements,
                    groupSourceElements, outputBusy, planGroup, firstBatchPreloaded);
            } else {
                ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                             groupSourceElements, outputBusy, firstBatchPreloaded, inputSplit);
            }
            if constexpr (AdaptiveGroupBatch) {
                batchRows_ = baseBatchRows;
                sourceBatchRows_ = baseSourceBatchRows;
            }
            if constexpr (!GroupPhase) {
                inputStart = inputEnd;
                outputStart += groupElements;
            }
        }
    }

private:
    __aicore__ inline void FillConstantVector(const LocalTensor<int32_t>& tensor,
                                              uint32_t start, uint32_t count,
                                              int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddScalarVector(const LocalTensor<int32_t>& tensor,
                                           uint32_t start, uint32_t count,
                                           int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                          mask, 1, {1, 1, 8, 8});
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void GenerateOffsets(uint32_t inputStart, uint32_t inputEnd,
                                           uint32_t groupElements, uint32_t sourceRowOffset)
    {
        GenerateOffsetsTo(offsetBuffer_.Get<int32_t>(), batchRows_, inputStart, inputEnd,
            groupElements, sourceRowOffset);
    }

    __aicore__ inline void GenerateOffsets(uint32_t inputStart, uint32_t inputEnd,
        uint32_t groupElements, uint32_t sourceRowOffset, uint32_t tableRows)
    {
        GenerateOffsetsTo(offsetBuffer_.Get<int32_t>(), tableRows, inputStart, inputEnd,
            groupElements, sourceRowOffset);
    }

    // 首行地址只取决于输入段在连续 source 中的布局；将它和逐行增量拆开后，
    // key33 可由少量 producer 核生成并写入 workspace。
    __aicore__ inline void GenerateOffsetBase(const LocalTensor<int32_t>& offsets,
        uint32_t inputStart, uint32_t inputEnd, uint32_t groupElements,
        uint32_t sourceBatchRows, uint32_t sourceRowOffset)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        if (alignedElements == 0 || sourceBatchRows == 0 || inputEnd <= inputStart ||
            inputEnd > inputCount_) {
            valid_ = false;
            return;
        }
        CreateVecIndex(offsets, static_cast<int32_t>(0), alignedElements);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignedElements));

        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t sourceOffset = static_cast<int64_t>(sourceBase) +
                    static_cast<int64_t>(sourceRowOffset) * inputElements;
                const int64_t adjustment64 = (sourceOffset -
                    static_cast<int64_t>(outputPrefix)) * static_cast<int64_t>(sizeof(int32_t));
                if (adjustment64 < MIN_INT32_VALUE || adjustment64 > MAX_INT32_VALUE ||
                    outputPrefix > groupElements || inputElements > groupElements - outputPrefix) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(offsets, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(adjustment64));
            }
            outputPrefix += inputElements;
            const uint64_t sourceElements = static_cast<uint64_t>(sourceBatchRows) *
                inputElements;
            if (sourceElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(sourceElements));
            if (sourceBase > MAX_UINT32) {
                valid_ = false;
                return;
            }
        }
        if (outputPrefix != groupElements) {
            valid_ = false;
            return;
        }
        // Gather 仍会访问行尾对齐元素，将其固定映射到 source[0]，避免无效偏移越界。
        FillConstantVector(offsets, groupElements, alignedElements - groupElements, 0);
    }

    __aicore__ inline void GenerateOffsetStrides(const LocalTensor<int32_t>& strides,
        uint32_t inputStart, uint32_t inputEnd, uint32_t groupElements)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        if (alignedElements == 0 || inputEnd <= inputStart || inputEnd > inputCount_) {
            valid_ = false;
            return;
        }
        uint64_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                if (outputPrefix > groupElements || inputElements > groupElements - outputPrefix ||
                    static_cast<uint64_t>(inputElements) * sizeof(int32_t) > MAX_INT32_VALUE) {
                    valid_ = false;
                    return;
                }
                FillConstantVector(strides, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(inputElements * sizeof(int32_t)));
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements) {
            valid_ = false;
            return;
        }
        FillConstantVector(strides, groupElements, alignedElements - groupElements, 0);
    }

    __aicore__ inline void ExpandOffsetRows(const LocalTensor<int32_t>& offsets,
        const LocalTensor<int32_t>& strides, uint32_t tableRows, uint32_t alignedElements)
    {
        if (tableRows == 0 || alignedElements == 0) {
            valid_ = false;
            return;
        }
        for (uint32_t row = 1; row + 1 < tableRows; ++row) {
            Add(offsets[row * alignedElements], offsets[(row - 1) * alignedElements],
                strides, static_cast<int32_t>(alignedElements));
        }
        if (tableRows > 1) {
            Add(strides, offsets[(tableRows - 2) * alignedElements], strides,
                static_cast<int32_t>(alignedElements));
        }
    }

    __aicore__ inline void GenerateOffsetsTo(const LocalTensor<int32_t>& offsets,
        uint32_t tableRows, uint32_t inputStart, uint32_t inputEnd,
        uint32_t groupElements, uint32_t sourceRowOffset)
    {
        if (tableRows == 0) {
            valid_ = false;
            return;
        }
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        GenerateOffsetBase(offsets, inputStart, inputEnd, groupElements, sourceBatchRows_,
            sourceRowOffset);
        if (!valid_ || tableRows == 1) {
            return;
        }
        LocalTensor<int32_t> strides = offsets[(tableRows - 1) * alignedElements];
        GenerateOffsetStrides(strides, inputStart, inputEnd, groupElements);
        if (!valid_) {
            return;
        }
        ExpandOffsetRows(offsets, strides, tableRows, alignedElements);
    }

    // key47 的 mixed layout 仅移动连续后缀：前缀仍在标准 source 起点，后缀临时
    // 位于完整 source 批之后。标准 offset 表在 group 首批生成一次；之后只对后缀
    // 施加可逆 delta，避免每批重复 CreateVecIndex/Muls/整表行传播。
    __aicore__ inline void ApplySuffixPrefetchOffsetDelta(uint32_t groupElements,
        uint32_t suffixOutputStart, uint32_t suffixElements,
        uint32_t suffixSourceElements, bool add)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t deltaBytes = static_cast<uint64_t>(suffixSourceElements) *
            sizeof(int32_t);
        if (alignedElements == 0 || suffixOutputStart >= groupElements || suffixElements == 0 ||
            suffixElements != groupElements - suffixOutputStart || deltaBytes > MAX_INT32_VALUE) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
        const int32_t signedDelta = add ? static_cast<int32_t>(deltaBytes) :
            -static_cast<int32_t>(deltaBytes);
        for (uint32_t row = 0; row < batchRows_; ++row) {
            AddScalarVector(offsets, row * alignedElements + suffixOutputStart, suffixElements,
                signedDelta);
        }
    }

    // key43 的常驻元数据只有首行 base 和逐元素行 stride。每个 group 仅构造一次，
    // 每批完整 offset 表随后写入 VECOUT，并在 Gather 后被结果安全覆盖。
    __aicore__ inline void PrepareOffsetOutputAliasSeeds(uint32_t inputStart,
        uint32_t inputEnd, uint32_t groupElements)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t seedElements = static_cast<uint64_t>(OFFSET_OUTPUT_ALIAS_SEED_ROWS) *
            alignedElements;
        if (alignedElements == 0 || seedElements > MAX_UINT32 ||
            seedElements * sizeof(int32_t) > offsetBufferBytes_) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> seeds = offsetBuffer_.Get<int32_t>();
        seeds.SetSize(static_cast<uint32_t>(seedElements));
        LocalTensor<int32_t> base = seeds;
        LocalTensor<int32_t> strides = seeds[alignedElements];
        base.SetSize(alignedElements);
        strides.SetSize(alignedElements);
        GenerateOffsetBase(base, inputStart, inputEnd, groupElements, sourceBatchRows_, 0);
        if (!valid_) {
            return;
        }
        GenerateOffsetStrides(strides, inputStart, inputEnd, groupElements);
    }

    // `Adds(dst, base, 0)` 是经 C220 探针验证的 VECCALC→VECOUT 拷贝形式。后续 Add
    // 只读取前一输出行和不变 stride，因此不再破坏下一批所需的 seed。
    __aicore__ inline void BuildOffsetOutputAliasRows(const LocalTensor<int32_t>& offsets,
        const LocalTensor<int32_t>& base, const LocalTensor<int32_t>& strides,
        uint32_t tableRows, uint32_t alignedElements)
    {
        if (tableRows == 0 || alignedElements == 0) {
            valid_ = false;
            return;
        }
        Adds(offsets, base, static_cast<int32_t>(0), static_cast<int32_t>(alignedElements));
        for (uint32_t row = 1; row < tableRows; ++row) {
            Add(offsets[row * alignedElements], offsets[(row - 1) * alignedElements], strides,
                static_cast<int32_t>(alignedElements));
        }
    }

    __aicore__ inline uint32_t GetSharedSeedGroupCount(uint32_t planIndex) const
    {
        if (planIndex == 0) {
            return blockNum_ > tilingData_->highCoreCount ? tilingData_->groupCount0 : 0;
        }
        if (planIndex == 1) {
            return tilingData_->highCoreCount != 0 ? tilingData_->groupCount1 : 0;
        }
        return 0;
    }

    __aicore__ inline uint32_t GetSharedSeedBatchRows(uint32_t planIndex) const
    {
        return planIndex == 0 ? tilingData_->batchRows0 : tilingData_->batchRows1;
    }

    __aicore__ inline uint32_t GetSharedSeedSourceBatchRows(uint32_t planIndex) const
    {
        return planIndex == 0 ? tilingData_->sourceBatchRows0 : tilingData_->sourceBatchRows1;
    }

    __aicore__ inline uint64_t GetSharedSeedOffsetElements(uint32_t planIndex,
        uint32_t group)
    {
        uint64_t offsetElements = 0;
        if (planIndex > 1 || group >= GetSharedSeedGroupCount(planIndex)) {
            valid_ = false;
            return 0;
        }
        for (uint32_t currentPlan = 0; currentPlan <= planIndex; ++currentPlan) {
            const uint32_t groupLimit = GetSharedSeedGroupCount(currentPlan);
            const uint32_t end = currentPlan == planIndex ? group : groupLimit;
            const uint32_t planBase = currentPlan * GATHER_PLAN_INPUT_CAPACITY;
            for (uint32_t currentGroup = 0; currentGroup < end; ++currentGroup) {
                const uint32_t groupElements =
                    tilingData_->groupElements[planBase + currentGroup];
                const uint32_t alignedElements = AlignUpGatherElements(groupElements);
                if (groupElements == 0 || alignedElements == 0 ||
                    offsetElements > MAX_UINT32 - 2ULL * alignedElements) {
                    valid_ = false;
                    return 0;
                }
                offsetElements += 2ULL * alignedElements;
            }
        }
        return offsetElements;
    }

    __aicore__ inline void PrepareSharedOffsetSeeds()
    {
        const uint32_t groupCount0 = GetSharedSeedGroupCount(0);
        const uint32_t groupCount1 = GetSharedSeedGroupCount(1);
        const uint32_t totalGroups = groupCount0 + groupCount1;
        if (totalGroups == 0 || totalGroups > blockNum_) {
            valid_ = false;
        }
        if (valid_ && blockIdx_ < totalGroups) {
            const uint32_t producerPlan = blockIdx_ < groupCount0 ? 0 : 1;
            const uint32_t producerGroup = producerPlan == 0 ? blockIdx_ : blockIdx_ - groupCount0;
            const uint32_t planBase = producerPlan * GATHER_PLAN_INPUT_CAPACITY;
            const uint32_t inputStart = producerGroup == 0 ? 0 :
                tilingData_->groupInputEnd[planBase + producerGroup - 1];
            const uint32_t inputEnd = tilingData_->groupInputEnd[planBase + producerGroup];
            const uint32_t groupElements = tilingData_->groupElements[planBase + producerGroup];
            const uint32_t alignedElements = AlignUpGatherElements(groupElements);
            const uint64_t seedBytes = static_cast<uint64_t>(2) * alignedElements *
                sizeof(int32_t);
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                alignedElements == 0 || seedBytes > offsetBufferBytes_) {
                valid_ = false;
            } else {
                LocalTensor<int32_t> seedStorage = offsetBuffer_.Get<int32_t>();
                seedStorage.SetSize(2 * alignedElements);
                LocalTensor<int32_t> seedStrides = seedStorage[alignedElements];
                GenerateOffsetBase(seedStorage, inputStart, inputEnd, groupElements,
                    GetSharedSeedSourceBatchRows(producerPlan), 0);
                GenerateOffsetStrides(seedStrides, inputStart, inputEnd, groupElements);
                const uint64_t seedOffset = GetSharedSeedOffsetElements(producerPlan,
                    producerGroup);
                if (valid_) {
                    SyncVToMte3();
                    const DataCopyExtParams copyParams{
                        1, static_cast<uint32_t>(seedBytes), 0, 0, 0};
                    DataCopyPad(sharedSeedGm_[seedOffset], seedStorage, copyParams);
                }
            }
        }
        // 所有 producer 的 MTE3 写回完成后再读取；C220 的硬件 SyncAll 同时提供
        // 全核可见性，不需要软件轮询 workspace。
        SyncAll<true>();
    }

    __aicore__ inline void LoadSharedOffsetSeed(uint32_t planIndex, uint32_t group,
        uint32_t groupElements)
    {
        const uint32_t groupCount = GetSharedSeedGroupCount(planIndex);
        const uint32_t planBase = planIndex * GATHER_PLAN_INPUT_CAPACITY;
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t vectorBytes = static_cast<uint64_t>(alignedElements) * sizeof(int32_t);
        const uint64_t tableBytes = static_cast<uint64_t>(batchRows_) * vectorBytes;
        if (planIndex > 1 || group >= groupCount || groupElements == 0 ||
            tilingData_->groupElements[planBase + group] != groupElements || batchRows_ < 2 ||
            vectorBytes == 0 || vectorBytes > MAX_UINT32 || tableBytes > offsetBufferBytes_) {
            valid_ = false;
            return;
        }
        const uint64_t seedOffset = GetSharedSeedOffsetElements(planIndex, group);
        if (!valid_) {
            return;
        }
        LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
        offsets.SetSize(static_cast<uint32_t>(tableBytes / sizeof(int32_t)));
        const uint64_t gapBytes = static_cast<uint64_t>(batchRows_ - 2) * vectorBytes;
        // 在 GM->UB 的 DataCopyPad 中，dstStride 是 UB 的 32B block 间隔，而不是字节数。
        // alignedElements 已保证 vectorBytes 与 gapBytes 均为 32B 对齐。
        if (gapBytes % UB_BLOCK_BYTES != 0 || gapBytes / UB_BLOCK_BYTES > MAX_UINT32) {
            valid_ = false;
            return;
        }
        const DataCopyExtParams copyParams{
            2, static_cast<uint32_t>(vectorBytes), 0,
            static_cast<uint32_t>(gapBytes / UB_BLOCK_BYTES), 0};
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(offsets, sharedSeedGm_[seedOffset], copyParams, padParams);
        SyncMte2ToV();
        LocalTensor<int32_t> strides = offsets[(batchRows_ - 1) * alignedElements];
        ExpandOffsetRows(offsets, strides, batchRows_, alignedElements);
    }

    // 记录每个输入段在 source 中相邻行之间的字节增量。尾部补齐 lane 固定为 0，
    // 这样后续的批量 Add/Sub 不会触碰未初始化的 offset。
    __aicore__ inline void GenerateDeltaOffsets(uint32_t inputStart, uint32_t inputEnd,
        uint32_t groupElements, uint32_t rowShift, const LocalTensor<int32_t>& delta)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        if (alignedElements == 0 || delta.GetSize() < alignedElements) {
            valid_ = false;
            return;
        }
        uint32_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const uint64_t deltaBytes = static_cast<uint64_t>(inputElements) *
                    sizeof(int32_t) * rowShift;
                if (deltaBytes > MAX_INT32_VALUE ||
                    inputElements > groupElements - outputPrefix) {
                    valid_ = false;
                    return;
                }
                FillConstantVector(delta, outputPrefix, inputElements,
                    static_cast<int32_t>(deltaBytes));
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements) {
            valid_ = false;
            return;
        }
        FillConstantVector(delta, groupElements, alignedElements - groupElements, 0);
    }

    // 完整 64-lane 片段用 src1RepStride=0 广播，尾片逐行处理，避免硬件按完整
    // 64 lane 生成越界地址。单 uint64_t 参数是 count 语义，不能传位掩码。delta 位于
    // 独立 source 尾部，不与 offset 表别名。
    __aicore__ inline void ApplyDeltaToOffsetRows(
        const LocalTensor<int32_t>& offsets, const LocalTensor<int32_t>& delta,
        uint32_t rows, uint32_t alignedElements, bool add)
    {
        if (rows == 0 || alignedElements == 0) {
            return;
        }
        for (uint32_t done = 0; done < alignedElements; done += DELTA_OFFSET_VECTOR_LANES) {
            const uint32_t current = static_cast<uint32_t>(MinU64(DELTA_OFFSET_VECTOR_LANES,
                alignedElements - done));
            if (current == DELTA_OFFSET_VECTOR_LANES) {
                const uint32_t repeatStride = alignedElements / GATHER_BLOCK_ELEMENTS;
                if (repeatStride == 0 || repeatStride > 255 || rows > 255) {
                    valid_ = false;
                    return;
                }
                const BinaryRepeatParams repeatParams{
                    1, 1, 1, static_cast<uint8_t>(repeatStride),
                    static_cast<uint8_t>(repeatStride), 0};
                if (add) {
                    Add<int32_t>(offsets[done], offsets[done], delta[done],
                        static_cast<uint64_t>(DELTA_OFFSET_VECTOR_LANES),
                        static_cast<uint8_t>(rows), repeatParams);
                } else {
                    Sub<int32_t>(offsets[done], offsets[done], delta[done],
                        static_cast<uint64_t>(DELTA_OFFSET_VECTOR_LANES),
                        static_cast<uint8_t>(rows), repeatParams);
                }
                continue;
            }
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t rowOffset = row * alignedElements + done;
                if (add) {
                    Add<int32_t>(offsets[rowOffset], offsets[rowOffset], delta[done],
                        static_cast<int32_t>(current));
                } else {
                    Sub<int32_t>(offsets[rowOffset], offsets[rowOffset], delta[done],
                        static_cast<int32_t>(current));
                }
            }
        }
    }

    __aicore__ inline void ProcessGroupDeltaStagedOffsets(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t tableElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        const uint64_t requiredOffsetBytes = static_cast<uint64_t>(batchRows_) *
            alignedElements * sizeof(int32_t);
        if (sourceBatchRows_ <= batchRows_ ||
            sourceBatchRows_ > batchRows_ * 2 || batchRows_ == 0 ||
            tableElements > MAX_UINT32 || requiredOffsetBytes > offsetBufferBytes_ ||
            (static_cast<uint64_t>(groupSourceElements) + alignedElements) * sizeof(int32_t) >
                sourceBufferBytes_ ||
            tableElements * sizeof(int32_t) > outputBufferBytes_ ||
            groupElements > outputRowElements_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetStorage = offsetBuffer_.Get<int32_t>();
        source.SetSize(groupSourceElements + alignedElements);
        output.SetSize(static_cast<uint32_t>(tableElements));
        offsetStorage.SetSize(static_cast<uint32_t>(requiredOffsetBytes / sizeof(int32_t)));
        LocalTensor<uint32_t> offsets = offsetStorage.ReinterpretCast<uint32_t>();
        // delta 使用 source 尾部的独立 UB 区域，避免与 offset 表相邻时触发 C220
        // 二元向量地址生成异常；Gather 只读取 source 的前 groupSourceElements。
        LocalTensor<int32_t> delta = source[groupSourceElements];

        bool offsetsReady = false;
        for (uint64_t stageOffset = 0; stageOffset < rowCount_;
             stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputs(inputStart, inputEnd, stageStart, stageRows, source);

            // 首个 source stage 的 DMA 与 offset 建表重叠；后续 stage 复用同一张基准表。
            if (!offsetsReady) {
                GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                if (!valid_) {
                    return;
                }
                offsetsReady = true;
            }
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                const bool shifted = sourceRowOffset != 0;
                if (shifted) {
                    if (sourceRowOffset > MAX_INT32_VALUE) {
                        valid_ = false;
                        return;
                    }
                    GenerateDeltaOffsets(inputStart, inputEnd, groupElements, sourceRowOffset,
                        delta);
                    ApplyDeltaToOffsetRows(offsetStorage, delta, currentBatch,
                        alignedElements, true);
                    if (!valid_) {
                        return;
                    }
                }

                // offset 平移只访问 VECCALC，可覆盖上一批的 MTE3 等待。
                if (outputBusy) {
                    SyncMte3ToV();
                }
                GatherSegmented(output, source, offsets, currentBatch * alignedElements);
                const bool sourceStageConsumed = sourceRowOffset + currentBatch >= stageRows;
                if (sourceStageConsumed) {
                    SyncVToMte2();
                }
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);

                if (shifted) {
                    // MTE3 已经拿到 output，恢复 offset 不会影响写回；基准表可供下一 stage
                    // 继续复用。delta 会在下一个后续子批直接按当前行偏移重新生成。
                    ApplyDeltaToOffsetRows(offsetStorage, delta, currentBatch,
                        alignedElements, false);
                    if (!valid_) {
                        return;
                    }
                }
                outputBusy = true;
            }
        }
    }

    __aicore__ inline bool GetRowMajorSourceRowElements(uint32_t inputStart,
        uint32_t inputEnd, uint32_t groupElements, uint32_t& sourceRowElements)
    {
        uint64_t outputElements = 0;
        uint64_t sourceElements = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            outputElements += inputElements;
            if (inputElements != 0) {
                sourceElements += AlignUpGatherElements(inputElements);
            }
        }
        if (outputElements != groupElements || sourceElements == 0 ||
            sourceElements > MAX_UINT32) {
            return false;
        }
        sourceRowElements = static_cast<uint32_t>(sourceElements);
        return true;
    }

    __aicore__ inline void GenerateRowMajorOffsets(uint32_t inputStart, uint32_t inputEnd,
        uint32_t groupElements)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        uint32_t sourceRowElements = 0;
        if (!GetRowMajorSourceRowElements(inputStart, inputEnd, groupElements,
            sourceRowElements) ||
            static_cast<uint64_t>(sourceRowElements) * sizeof(int32_t) > MAX_INT32_VALUE) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> strides = offsets[(batchRows_ - 1) * alignedElements];
        CreateVecIndex(offsets, static_cast<int32_t>(0), alignedElements);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignedElements));

        uint32_t outputPrefix = 0;
        uint32_t sourcePrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t adjustment = static_cast<int64_t>(sourcePrefix - outputPrefix) *
                    static_cast<int64_t>(sizeof(int32_t));
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(offsets, outputPrefix, inputElements,
                    static_cast<int32_t>(adjustment));
                sourcePrefix += AlignUpGatherElements(inputElements);
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements || sourcePrefix != sourceRowElements) {
            valid_ = false;
            return;
        }
        FillConstantVector(offsets, groupElements, alignedElements - groupElements, 0);
        if (batchRows_ > 1) {
            FillConstantVector(strides, 0, alignedElements,
                static_cast<int32_t>(sourceRowElements * sizeof(int32_t)));
        }
        for (uint32_t row = 1; row + 1 < batchRows_; ++row) {
            Add(offsets[row * alignedElements], offsets[(row - 1) * alignedElements],
                strides, static_cast<int32_t>(alignedElements));
        }
        if (batchRows_ > 1) {
            Add(strides, offsets[(batchRows_ - 2) * alignedElements], strides,
                static_cast<int32_t>(alignedElements));
        }
    }

    __aicore__ inline void GenerateMicroRegisterOffsets(uint32_t inputStart,
        uint32_t inputEnd, uint32_t groupElements, uint32_t paddedElements,
        LocalTensor<int32_t>& base, LocalTensor<int32_t>& strides)
    {
        if (paddedElements == 0 || groupElements > paddedElements) {
            valid_ = false;
            return;
        }
        base.SetSize(paddedElements);
        strides.SetSize(paddedElements);
        // MicroAPI 的 Gather 索引以元素为单位；每个输入段的行 stride 也直接记录为
        // 元素数，避免普通 Gather 路径中整张字节 offset 表和逐行传播。
        CreateVecIndex(base, static_cast<int32_t>(0), paddedElements);
        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t adjustment = static_cast<int64_t>(sourceBase) -
                    static_cast<int64_t>(outputPrefix);
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE ||
                    outputPrefix > MAX_UINT32 ||
                    outputPrefix + inputElements > MAX_UINT32) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(base, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(adjustment));
                FillConstantVector(strides, static_cast<uint32_t>(outputPrefix),
                    inputElements, static_cast<int32_t>(inputElements));
                outputPrefix += inputElements;
                sourceBase += AlignUpGatherElements(static_cast<uint32_t>(
                    static_cast<uint64_t>(sourceBatchRows_) * inputElements));
                if (sourceBase > MAX_UINT32) {
                    valid_ = false;
                    return;
                }
            }
        }
        if (outputPrefix != groupElements) {
            valid_ = false;
            return;
        }
        // 尾向量和补齐向量的索引固定为 source[0]，且 stride 为 0；这些 lane 由 mask
        // 屏蔽，但显式初始化可避免任何硬件尾读依赖未定义 UB 内容。
        FillConstantVector(base, groupElements, paddedElements - groupElements, 0);
        FillConstantVector(strides, groupElements, paddedElements - groupElements, 0);
    }

// C220 的 Vector 编译目标不提供 MicroAPI 所需的 __VEC_SCOPE__ 语义。整个实现而非
// 调用点必须预处理隔离，否则即使 Host 不调度 key25，编译器仍会解析该函数体。
#if defined(__DAV_C310_VEC__) || defined(__DAV_310R6_VEC__)
    __aicore__ inline void MicroGatherSegmented(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& source, const LocalTensor<int32_t>& base,
        const LocalTensor<int32_t>& strides, uint32_t rows, uint32_t alignedElements)
    {
        auto outputAddr = reinterpret_cast<__ubuf__ int32_t*>(output.GetPhyAddr());
        auto sourceAddr = reinterpret_cast<__ubuf__ int32_t*>(source.GetPhyAddr());
        auto baseAddr = reinterpret_cast<__ubuf__ uint32_t*>(base.GetPhyAddr());
        auto strideAddr = reinterpret_cast<__ubuf__ uint32_t*>(strides.GetPhyAddr());
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<uint32_t> indexReg;
            MicroAPI::RegTensor<uint32_t> strideReg;
            MicroAPI::RegTensor<int32_t> valueReg;
            for (uint32_t done = 0; done < alignedElements; done += MICRO_GATHER_VF) {
                const uint32_t current = static_cast<uint32_t>(MinU64(
                    MICRO_GATHER_VF, alignedElements - done));
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<uint32_t>(current);
                // Host 已将 base/stride 表补齐到 64-lane 边界，因此尾向量也可用完整
                // DataCopy 载入；Gather 和输出写回由 mask 精确限制有效元素。
                MicroAPI::DataCopy(indexReg, baseAddr + done);
                MicroAPI::DataCopy(strideReg, strideAddr + done);
                for (uint16_t row = 0; row < rows; ++row) {
                    if (row != 0) {
                        MicroAPI::Add(indexReg, indexReg, strideReg, mask);
                    }
                    MicroAPI::DataCopyGather(valueReg, sourceAddr, indexReg, mask);
                    MicroAPI::DataCopy(outputAddr + static_cast<uint32_t>(row) * alignedElements + done,
                        valueReg, mask);
                }
            }
        }
    }
#else
    __aicore__ inline void MicroGatherSegmented(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& source, const LocalTensor<int32_t>& base,
        const LocalTensor<int32_t>& strides, uint32_t rows, uint32_t alignedElements)
    {
        (void)output;
        (void)source;
        (void)base;
        (void)strides;
        (void)rows;
        (void)alignedElements;
        valid_ = false;
    }
#endif

    __aicore__ inline void ProcessGroupMicroRegister(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint32_t paddedElements = (alignedElements + MICRO_GATHER_VF - 1) /
            MICRO_GATHER_VF * MICRO_GATHER_VF;
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetStorage = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> base = offsetStorage;
        LocalTensor<int32_t> strides = offsetStorage[paddedElements];
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsetStorage.SetSize(2 * paddedElements);
        GenerateMicroRegisterOffsets(inputStart, inputEnd, groupElements, paddedElements,
            base, strides);
        if (!valid_) {
            return;
        }

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            MicroGatherSegmented(output, source, base, strides, currentBatch, alignedElements);
            SyncVToMte2();
            SyncVToMte3();

            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }
    }

    __aicore__ inline LocalTensor<int32_t> GetPartitionSource(uint32_t part)
    {
        return part == 0 ? sourcePart0Buffer_.Get<int32_t>() :
                           sourcePart1Buffer_.Get<int32_t>();
    }

    __aicore__ inline LocalTensor<int32_t> GetPartitionOutput(uint32_t part)
    {
        return part == 0 ? outputPart0Buffer_.Get<int32_t>() :
                           outputPart1Buffer_.Get<int32_t>();
    }

    __aicore__ inline LocalTensor<int32_t> GetPartitionOffsets(uint32_t part)
    {
        return part == 0 ? offsetPart0Buffer_.Get<int32_t>() :
                           offsetPart1Buffer_.Get<int32_t>();
    }

    __aicore__ inline uint32_t GetPartitionSourceBufferBytes(uint32_t part) const
    {
        return part == 0 ? sourcePart0BufferBytes_ : sourcePart1BufferBytes_;
    }

    __aicore__ inline uint32_t GetPartitionOutputBufferBytes(uint32_t part) const
    {
        return part == 0 ? outputPart0BufferBytes_ : outputPart1BufferBytes_;
    }

    __aicore__ inline void GetPartitionLayout(uint32_t inputStart, uint32_t inputEnd,
        uint32_t& outputElements, uint32_t& sourceElements)
    {
        uint64_t output = 0;
        uint64_t source = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t sourceRawElements = static_cast<uint64_t>(sourceBatchRows_) *
                inputElements;
            if (sourceRawElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            output += inputElements;
            source += AlignUpGatherElements(static_cast<uint32_t>(sourceRawElements));
            if (output > MAX_UINT32 || source > MAX_UINT32) {
                valid_ = false;
                return;
            }
        }
        if (output == 0 || source == 0) {
            valid_ = false;
            return;
        }
        outputElements = static_cast<uint32_t>(output);
        sourceElements = static_cast<uint32_t>(source);
    }

    __aicore__ inline void StartPartitionCopy(uint32_t part, uint32_t inputStart,
        uint32_t inputEnd, uint64_t batchStart, uint32_t currentBatch,
        uint32_t sourceElements, event_t readyEvent)
    {
        LocalTensor<int32_t> source = GetPartitionSource(part);
        source.SetSize(sourceElements);
        CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
        SetFlag<HardEvent::MTE2_V>(readyEvent);
    }

    __aicore__ inline void CopyPartitionOutput(uint32_t part,
        const LocalTensor<int32_t>& output, uint32_t outputElements,
        uint32_t alignedElements, uint64_t batchStart, uint32_t batchRows,
        uint32_t outputStart, uint32_t partitionOutputStart, event_t outputReuse)
    {
        if (outputElements == 0 || alignedElements < outputElements ||
            outputStart + partitionOutputStart > outputRowElements_ ||
            outputElements > outputRowElements_ - outputStart - partitionOutputStart) {
            valid_ = false;
            return;
        }
        const event_t vectorToMte3 = static_cast<event_t>(
            pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart +
            partitionOutputStart;
        const DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(batchRows),
            static_cast<uint32_t>(outputElements * sizeof(int32_t)),
            static_cast<uint32_t>((alignedElements - outputElements) /
                GATHER_BLOCK_ELEMENTS),
            static_cast<uint32_t>((outputRowElements_ - outputElements) * sizeof(int32_t)), 0};
        DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
        SetFlag<HardEvent::MTE3_V>(outputReuse);
    }

    __aicore__ inline void CopyDenseRows(const LocalTensor<int32_t>& dst,
        const LocalTensor<int32_t>& src, uint32_t rows, uint32_t dstRowElements,
        uint32_t srcRowElements, uint32_t elements)
    {
        if (rows == 0 || rows > 255 || elements == 0 ||
            dstRowElements < elements || srcRowElements < elements ||
            dstRowElements % GATHER_BLOCK_ELEMENTS != 0 ||
            srcRowElements % GATHER_BLOCK_ELEMENTS != 0 ||
            dstRowElements / GATHER_BLOCK_ELEMENTS > MAX_UINT16_VALUE ||
            srcRowElements / GATHER_BLOCK_ELEMENTS > MAX_UINT16_VALUE) {
            valid_ = false;
            return;
        }
        const CopyRepeatParams copyParams{1, 1,
            static_cast<uint16_t>(dstRowElements / GATHER_BLOCK_ELEMENTS),
            static_cast<uint16_t>(srcRowElements / GATHER_BLOCK_ELEMENTS)};
        for (uint32_t offset = 0; offset < elements; offset += MICRO_GATHER_VF) {
            const uint32_t current = static_cast<uint32_t>(MinU64(
                MICRO_GATHER_VF, elements - offset));
            const uint64_t maskValue = current == MICRO_GATHER_VF ? ~0ULL :
                (1ULL << current) - 1ULL;
            uint64_t mask[2] = {maskValue, 0};
            Copy<int32_t>(dst[offset], src[offset], mask, static_cast<uint8_t>(rows),
                copyParams);
        }
    }

    // 将一个输出列 chunk 的 offset 直接排成 Gather Level-0 所需的 repeat-major 布局：
    // [row0 的 64 lane][row1 的 64 lane]...。末尾另留一行 stride，以一次 Add 生成
    // 后续行，避免先构造完整 row-major 表再做 UB Copy 转置。
    __aicore__ inline void GenerateDirectWavefrontOffsetChunk(
        const LocalTensor<int32_t>& offsetStage, uint32_t tableRows,
        uint32_t inputStart, uint32_t inputEnd, uint32_t partitionElements,
        uint32_t chunkOffset, uint32_t chunkElements)
    {
        const uint64_t stageElements = static_cast<uint64_t>(tableRows + 1) *
            MICRO_GATHER_VF;
        const uint64_t chunkEnd = static_cast<uint64_t>(chunkOffset) + chunkElements;
        if (tableRows == 0 || tableRows > GATHER_MAX_REPEAT_TIMES ||
            inputStart >= inputEnd || inputEnd > inputCount_ || partitionElements == 0 ||
            chunkElements == 0 || chunkElements > MICRO_GATHER_VF ||
            chunkOffset > partitionElements || chunkEnd > partitionElements ||
            chunkOffset > MAX_INT32_VALUE || offsetStage.GetSize() < stageElements) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> offsets = offsetStage;
        LocalTensor<int32_t> strides = offsetStage[tableRows * MICRO_GATHER_VF];
        offsets.SetSize(tableRows * MICRO_GATHER_VF);
        strides.SetSize(MICRO_GATHER_VF);
        CreateVecIndex(offsets, static_cast<int32_t>(chunkOffset), MICRO_GATHER_VF);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), MICRO_GATHER_VF);
        FillConstantVector(strides, 0, MICRO_GATHER_VF, 0);

        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (outputPrefix > partitionElements ||
                inputElements > partitionElements - outputPrefix) {
                valid_ = false;
                return;
            }
            const uint64_t inputEndOffset = outputPrefix + inputElements;
            if (inputElements != 0) {
                const uint64_t overlapStart = MaxU64(chunkOffset, outputPrefix);
                const uint64_t overlapEnd = MinU64(chunkEnd, inputEndOffset);
                if (overlapStart < overlapEnd) {
                    const int64_t adjustment64 = (static_cast<int64_t>(sourceBase) -
                        static_cast<int64_t>(outputPrefix)) * static_cast<int64_t>(sizeof(int32_t));
                    const uint64_t strideBytes = static_cast<uint64_t>(inputElements) *
                        sizeof(int32_t);
                    if (adjustment64 < MIN_INT32_VALUE || adjustment64 > MAX_INT32_VALUE ||
                        strideBytes > MAX_INT32_VALUE) {
                        valid_ = false;
                        return;
                    }
                    const uint32_t localStart = static_cast<uint32_t>(overlapStart - chunkOffset);
                    const uint32_t localCount = static_cast<uint32_t>(overlapEnd - overlapStart);
                    AddScalarVector(offsets, localStart, localCount,
                        static_cast<int32_t>(adjustment64));
                    FillConstantVector(strides, localStart, localCount,
                        static_cast<int32_t>(strideBytes));
                }
            }
            outputPrefix = inputEndOffset;
            const uint64_t sourceElements = static_cast<uint64_t>(sourceBatchRows_) *
                inputElements;
            if (sourceElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(sourceElements));
            if (sourceBase > MAX_UINT32) {
                valid_ = false;
                return;
            }
        }
        if (outputPrefix != partitionElements) {
            valid_ = false;
            return;
        }
        // Gather 的 mask 会屏蔽尾 lane；仍初始化成 source[0] 和零 stride，避免任何
        // 实现相关的尾 offset 读取依赖未定义 UB 内容。
        FillConstantVector(offsets, chunkElements, MICRO_GATHER_VF - chunkElements, 0);
        for (uint32_t row = 1; row < tableRows; ++row) {
            Add(offsets[row * MICRO_GATHER_VF], offsets[(row - 1) * MICRO_GATHER_VF],
                strides, static_cast<int32_t>(MICRO_GATHER_VF));
        }
    }

    __aicore__ inline void GatherDirectWavefrontPartition(
        const LocalTensor<int32_t>& fullOutput, const LocalTensor<int32_t>& source,
        const LocalTensor<int32_t>& offsetStage, uint32_t inputStart, uint32_t inputEnd,
        uint32_t partitionElements, uint32_t partitionOutputStart, uint32_t alignedGroupElements,
        uint32_t currentBatch, bool firstChunkPrepared)
    {
        if (currentBatch == 0 || currentBatch > GATHER_MAX_REPEAT_TIMES ||
            partitionElements == 0 || alignedGroupElements < partitionElements ||
            partitionOutputStart > alignedGroupElements ||
            partitionElements > alignedGroupElements - partitionOutputStart ||
            alignedGroupElements % GATHER_BLOCK_ELEMENTS != 0) {
            valid_ = false;
            return;
        }
        const uint64_t strideBlocks = alignedGroupElements / GATHER_BLOCK_ELEMENTS;
        if (currentBatch > 1 && (strideBlocks == 0 || strideBlocks > MAX_UINT16_VALUE)) {
            valid_ = false;
            return;
        }
        LocalTensor<uint32_t> offsets = offsetStage.ReinterpretCast<uint32_t>();
        const uint16_t dstRepStride = currentBatch <= 1 ? 0 :
            static_cast<uint16_t>(strideBlocks);
        for (uint32_t chunkOffset = 0; chunkOffset < partitionElements;
             chunkOffset += MICRO_GATHER_VF) {
            const uint32_t chunkElements = static_cast<uint32_t>(MinU64(MICRO_GATHER_VF,
                partitionElements - chunkOffset));
            if (!(firstChunkPrepared && chunkOffset == 0)) {
                GenerateDirectWavefrontOffsetChunk(offsetStage, currentBatch, inputStart, inputEnd,
                    partitionElements, chunkOffset, chunkElements);
                if (!valid_) {
                    return;
                }
            }
            Gather(fullOutput[partitionOutputStart + chunkOffset], source, offsets, 0,
                static_cast<uint64_t>(chunkElements), static_cast<uint8_t>(currentBatch),
                dstRepStride);
        }
    }

    __aicore__ inline void ProcessDirectWavefrontGroup(uint32_t inputStart,
        uint32_t inputEnd, uint32_t inputSplit, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        if (inputSplit <= inputStart || inputSplit >= inputEnd ||
            sourceBatchRows_ != batchRows_) {
            valid_ = false;
            return;
        }
        uint32_t outputElements[2] = {0, 0};
        uint32_t sourceElements[2] = {0, 0};
        GetPartitionLayout(inputStart, inputSplit, outputElements[0], sourceElements[0]);
        GetPartitionLayout(inputSplit, inputEnd, outputElements[1], sourceElements[1]);
        const uint32_t alignedGroupElements = AlignUpGatherElements(groupElements);
        const uint64_t fullMatrixElements = static_cast<uint64_t>(batchRows_) *
            alignedGroupElements;
        const uint64_t stagedOffsetElements = static_cast<uint64_t>(batchRows_ + 1) *
            MICRO_GATHER_VF;
        if (!valid_ || outputElements[0] == 0 || outputElements[1] == 0 ||
            outputElements[0] % GATHER_BLOCK_ELEMENTS != 0 ||
            outputElements[0] + outputElements[1] != groupElements ||
            sourceElements[0] + sourceElements[1] != groupSourceElements ||
            outputStart > outputRowElements_ || groupElements > outputRowElements_ - outputStart ||
            alignedGroupElements == 0 || fullMatrixElements > MAX_UINT32 ||
            stagedOffsetElements > MAX_UINT32 ||
            static_cast<uint64_t>(sourceElements[0]) * sizeof(int32_t) >
                sourcePart0BufferBytes_ ||
            static_cast<uint64_t>(sourceElements[1]) * sizeof(int32_t) >
                sourcePart1BufferBytes_ ||
            fullMatrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            stagedOffsetElements * sizeof(int32_t) > outputPart1BufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source0 = sourcePart0Buffer_.Get<int32_t>();
        LocalTensor<int32_t> source1 = sourcePart1Buffer_.Get<int32_t>();
        LocalTensor<int32_t> fullOutput = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetStage = offsetPart0Buffer_.Get<int32_t>();
        source0.SetSize(sourceElements[0]);
        source1.SetSize(sourceElements[1]);
        fullOutput.SetSize(static_cast<uint32_t>(fullMatrixElements));
        offsetStage.SetSize(static_cast<uint32_t>(stagedOffsetElements));

        const event_t sourceReady0 = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        const event_t sourceReady1 = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputSplit, batchStart, currentBatch, source0);
            SetFlag<HardEvent::MTE2_V>(sourceReady0);
            // 先生成第一条 64-lane offset，使其与首段连续 MTE2 重叠；随后第二段 MTE2
            // 可继续覆盖第一段 Gather 的 Vector 工作。
            GenerateDirectWavefrontOffsetChunk(offsetStage, currentBatch, inputStart, inputSplit,
                outputElements[0], 0, static_cast<uint32_t>(MinU64(MICRO_GATHER_VF,
                    outputElements[0])));
            if (!valid_) {
                break;
            }
            CopyInputs(inputSplit, inputEnd, batchStart, currentBatch, source1);
            SetFlag<HardEvent::MTE2_V>(sourceReady1);
            if (!valid_) {
                break;
            }

            WaitFlag<HardEvent::MTE2_V>(sourceReady0);
            // 直接 Gather 会写入 fullOutput；因此相对 key38 的 partial 路径需在首个
            // 写入前回收上一批 MTE3。两个 source MTE2 已在此前与该等待并行。
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherDirectWavefrontPartition(fullOutput, source0, offsetStage, inputStart, inputSplit,
                outputElements[0], 0, alignedGroupElements, currentBatch, true);
            if (!valid_) {
                break;
            }
            WaitFlag<HardEvent::MTE2_V>(sourceReady1);
            GatherDirectWavefrontPartition(fullOutput, source1, offsetStage, inputSplit, inputEnd,
                outputElements[1], outputElements[0], alignedGroupElements, currentBatch, false);
            if (!valid_) {
                break;
            }
            // 两段 Gather 都已停止读取 source，下一批 MTE2 可以先行，而完整 output 的
            // 单次整行 MTE3 继续与它重叠。
            SyncVToMte2();
            SyncVToMte3();
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) *
                    sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], fullOutput, copyOutParams);
            outputBusy = true;
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady0);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady1);
    }

    __aicore__ inline void ProcessDirectWavefront()
    {
        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        bool outputBusy = false;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t inputSplit = tilingData_->groupInputSplit[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ ||
                inputSplit <= inputStart || inputSplit >= inputEnd || groupElements == 0 ||
                groupSourceElements == 0 || outputStart > MAX_UINT32 - groupElements) {
                valid_ = false;
                return;
            }
            ProcessDirectWavefrontGroup(inputStart, inputEnd, inputSplit, outputStart,
                groupElements, groupSourceElements, outputBusy);
            if (!valid_) {
                return;
            }
            inputStart = inputEnd;
            outputStart += groupElements;
        }
        if (inputStart != inputCount_ || outputStart != outputRowElements_) {
            valid_ = false;
        }
    }

    __aicore__ inline void ProcessWavefrontGroup(uint32_t inputStart,
        uint32_t inputEnd, uint32_t inputSplit, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        if (inputSplit <= inputStart || inputSplit >= inputEnd ||
            sourceBatchRows_ != batchRows_) {
            valid_ = false;
            return;
        }
        uint32_t outputElements[2] = {0, 0};
        uint32_t sourceElements[2] = {0, 0};
        GetPartitionLayout(inputStart, inputSplit, outputElements[0], sourceElements[0]);
        GetPartitionLayout(inputSplit, inputEnd, outputElements[1], sourceElements[1]);
        if (!valid_ || outputElements[0] == 0 || outputElements[1] == 0 ||
            outputElements[0] % GATHER_BLOCK_ELEMENTS != 0 ||
            outputElements[0] + outputElements[1] != groupElements ||
            sourceElements[0] + sourceElements[1] != groupSourceElements ||
            outputStart > outputRowElements_ ||
            groupElements > outputRowElements_ - outputStart) {
            valid_ = false;
            return;
        }
        const uint32_t alignedGroupElements = AlignUpGatherElements(groupElements);
        const uint32_t alignedPartialElements[2] = {
            AlignUpGatherElements(outputElements[0]), AlignUpGatherElements(outputElements[1])};
        const uint32_t partialElements = alignedPartialElements[0] > alignedPartialElements[1] ?
            alignedPartialElements[0] : alignedPartialElements[1];
        const uint64_t fullMatrixElements = static_cast<uint64_t>(batchRows_) *
            alignedGroupElements;
        const uint64_t partialMatrixElements = static_cast<uint64_t>(batchRows_) *
            partialElements;
        if (alignedGroupElements == 0 || partialElements == 0 ||
            fullMatrixElements > MAX_UINT32 || partialMatrixElements > MAX_UINT32 ||
            static_cast<uint64_t>(sourceElements[0]) * sizeof(int32_t) >
                sourcePart0BufferBytes_ ||
            static_cast<uint64_t>(sourceElements[1]) * sizeof(int32_t) >
                sourcePart1BufferBytes_ ||
            fullMatrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            partialMatrixElements * sizeof(int32_t) > outputPart0BufferBytes_ ||
            partialMatrixElements * sizeof(int32_t) > outputPart1BufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source0 = sourcePart0Buffer_.Get<int32_t>();
        LocalTensor<int32_t> source1 = sourcePart1Buffer_.Get<int32_t>();
        LocalTensor<int32_t> fullOutput = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> partialOutput = outputPart0Buffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetStorage = offsetPart0Buffer_.Get<int32_t>();
        source0.SetSize(sourceElements[0]);
        source1.SetSize(sourceElements[1]);
        fullOutput.SetSize(static_cast<uint32_t>(fullMatrixElements));
        partialOutput.SetSize(static_cast<uint32_t>(partialMatrixElements));
        offsetStorage.SetSize(static_cast<uint32_t>(partialMatrixElements));
        LocalTensor<uint32_t> offsets = offsetStorage.ReinterpretCast<uint32_t>();

        // 两个 MTE2→V 事件标记在各自连续 source 搬入的尾部。V 只等待第一段后就
        // 开始 Gather，第二段 MTE2 因而可覆盖第一段的 Gather/Copy/建表工作。
        const event_t sourceReady0 = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        const event_t sourceReady1 = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_V>());
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputSplit, batchStart, currentBatch, source0);
            SetFlag<HardEvent::MTE2_V>(sourceReady0);
            // offset0 与第一段 MTE2 独立；先发起两侧工作后再建立 V 等待边。
            GenerateOffsetsTo(offsetStorage, batchRows_, inputStart, inputSplit,
                outputElements[0], 0);
            CopyInputs(inputSplit, inputEnd, batchStart, currentBatch, source1);
            SetFlag<HardEvent::MTE2_V>(sourceReady1);
            if (!valid_) {
                break;
            }

            WaitFlag<HardEvent::MTE2_V>(sourceReady0);
            GatherSegmented(partialOutput, source0, offsets,
                currentBatch * alignedPartialElements[0]);
            // full output 可能仍在写回上一批/上一组；只在第一次汇入前等待，尽可能
            // 用其 MTE3 时间覆盖当前第一段 MTE2 和 Gather。
            if (outputBusy) {
                SyncMte3ToV();
            }
            CopyDenseRows(fullOutput, partialOutput, currentBatch, alignedGroupElements,
                alignedPartialElements[0], outputElements[0]);
            if (!valid_) {
                break;
            }

            GenerateOffsetsTo(offsetStorage, batchRows_, inputSplit, inputEnd,
                outputElements[1], 0);
            WaitFlag<HardEvent::MTE2_V>(sourceReady1);
            GatherSegmented(partialOutput, source1, offsets,
                currentBatch * alignedPartialElements[1]);
            // 第二段 Gather 后两个 source 都不再被当前批读取，下一批 MTE2 可先行。
            SyncVToMte2();
            CopyDenseRows(fullOutput[outputElements[0]], partialOutput, currentBatch,
                alignedGroupElements, alignedPartialElements[1], outputElements[1]);
            if (!valid_) {
                break;
            }
            SyncVToMte3();
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) *
                    sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], fullOutput, copyOutParams);
            outputBusy = true;
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady0);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady1);
    }

    __aicore__ inline void ProcessWavefront()
    {
        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        bool outputBusy = false;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t inputSplit = tilingData_->groupInputSplit[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ ||
                inputSplit <= inputStart || inputSplit >= inputEnd || groupElements == 0 ||
                groupSourceElements == 0 || outputStart > MAX_UINT32 - groupElements) {
                valid_ = false;
                return;
            }
            ProcessWavefrontGroup(inputStart, inputEnd, inputSplit, outputStart,
                groupElements, groupSourceElements, outputBusy);
            if (!valid_) {
                return;
            }
            inputStart = inputEnd;
            outputStart += groupElements;
        }
        if (inputStart != inputCount_ || outputStart != outputRowElements_) {
            valid_ = false;
        }
    }

    __aicore__ inline void ProcessGroupPartitioned(uint32_t inputStart,
        uint32_t inputEnd, uint32_t inputSplit, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements)
    {
        if (inputSplit <= inputStart || inputSplit >= inputEnd ||
            sourceBatchRows_ != batchRows_) {
            valid_ = false;
            return;
        }
        uint32_t outputElements[2] = {0, 0};
        uint32_t sourceElements[2] = {0, 0};
        GetPartitionLayout(inputStart, inputSplit, outputElements[0], sourceElements[0]);
        GetPartitionLayout(inputSplit, inputEnd, outputElements[1], sourceElements[1]);
        if (!valid_ || outputElements[0] + outputElements[1] != groupElements ||
            sourceElements[0] + sourceElements[1] != groupSourceElements) {
            valid_ = false;
            return;
        }
        const uint32_t alignedElements[2] = {
            AlignUpGatherElements(outputElements[0]), AlignUpGatherElements(outputElements[1])};
        for (uint32_t part = 0; part < 2; ++part) {
            const uint64_t sourceBytes = static_cast<uint64_t>(sourceElements[part]) *
                sizeof(int32_t);
            const uint64_t matrixBytes = static_cast<uint64_t>(batchRows_) *
                alignedElements[part] * sizeof(int32_t);
            if (sourceBytes > GetPartitionSourceBufferBytes(part) ||
                matrixBytes > GetPartitionOutputBufferBytes(part)) {
                valid_ = false;
                return;
            }
        }

        LocalTensor<int32_t> offsets0 = GetPartitionOffsets(0);
        LocalTensor<int32_t> offsets1 = GetPartitionOffsets(1);
        offsets0.SetSize(batchRows_ * alignedElements[0]);
        offsets1.SetSize(batchRows_ * alignedElements[1]);
        GenerateOffsetsTo(offsets0, batchRows_, inputStart, inputSplit,
            outputElements[0], 0);
        GenerateOffsetsTo(offsets1, batchRows_, inputSplit, inputEnd,
            outputElements[1], 0);
        if (!valid_) {
            return;
        }

        event_t sourceReady[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>())};
        event_t sourceReuse[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>())};
        event_t outputReuse[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>())};
        bool outputBusy[2] = {false, false};

        const uint32_t firstBatch = static_cast<uint32_t>(MinU64(batchRows_, rowCount_));
        StartPartitionCopy(0, inputStart, inputSplit, rowStart_, firstBatch,
            sourceElements[0], sourceReady[0]);
        StartPartitionCopy(1, inputSplit, inputEnd, rowStart_, firstBatch,
            sourceElements[1], sourceReady[1]);

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            const uint64_t nextBatchOffset = batchOffset + batchRows_;

            WaitFlag<HardEvent::MTE2_V>(sourceReady[0]);
            if (outputBusy[0]) {
                WaitFlag<HardEvent::MTE3_V>(outputReuse[0]);
            }
            LocalTensor<int32_t> output0 = GetPartitionOutput(0);
            LocalTensor<int32_t> source0 = GetPartitionSource(0);
            output0.SetSize(batchRows_ * alignedElements[0]);
            source0.SetSize(sourceElements[0]);
            GatherSegmented(output0, source0, offsets0.ReinterpretCast<uint32_t>(),
                currentBatch * alignedElements[0]);
            SetFlag<HardEvent::V_MTE2>(sourceReuse[0]);
            CopyPartitionOutput(0, output0, outputElements[0], alignedElements[0],
                batchStart, currentBatch, outputStart, 0, outputReuse[0]);
            outputBusy[0] = true;
            if (!valid_) {
                break;
            }
            if (nextBatchOffset < rowCount_) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                WaitFlag<HardEvent::V_MTE2>(sourceReuse[0]);
                StartPartitionCopy(0, inputStart, inputSplit, rowStart_ + nextBatchOffset,
                    nextBatch, sourceElements[0], sourceReady[0]);
            }

            WaitFlag<HardEvent::MTE2_V>(sourceReady[1]);
            if (outputBusy[1]) {
                WaitFlag<HardEvent::MTE3_V>(outputReuse[1]);
            }
            LocalTensor<int32_t> output1 = GetPartitionOutput(1);
            LocalTensor<int32_t> source1 = GetPartitionSource(1);
            output1.SetSize(batchRows_ * alignedElements[1]);
            source1.SetSize(sourceElements[1]);
            GatherSegmented(output1, source1, offsets1.ReinterpretCast<uint32_t>(),
                currentBatch * alignedElements[1]);
            SetFlag<HardEvent::V_MTE2>(sourceReuse[1]);
            CopyPartitionOutput(1, output1, outputElements[1], alignedElements[1],
                batchStart, currentBatch, outputStart, outputElements[0], outputReuse[1]);
            outputBusy[1] = true;
            if (!valid_) {
                break;
            }
            if (nextBatchOffset < rowCount_) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                WaitFlag<HardEvent::V_MTE2>(sourceReuse[1]);
                StartPartitionCopy(1, inputSplit, inputEnd, rowStart_ + nextBatchOffset,
                    nextBatch, sourceElements[1], sourceReady[1]);
            }
        }

        for (uint32_t part = 0; part < 2; ++part) {
            if (outputBusy[part]) {
                WaitFlag<HardEvent::MTE3_V>(outputReuse[part]);
            }
            WaitFlag<HardEvent::V_MTE2>(sourceReuse[part]);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[0]);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[1]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[0]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[1]);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReuse[0]);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReuse[1]);
    }

    // key43：输出表先作为 Gather offset，再原位成为 Gather 结果。base/stride 在 group
    // 生命周期内只读，因而每批覆盖 VECOUT 后不会影响下一批 offset 的重建。
    __aicore__ inline void ProcessGroupOffsetOutputAlias(uint32_t inputStart,
        uint32_t inputEnd, uint32_t outputStart, uint32_t groupElements,
        uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t matrixElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        const uint64_t seedElements = static_cast<uint64_t>(OFFSET_OUTPUT_ALIAS_SEED_ROWS) *
            alignedElements;
        if (alignedElements == 0 || groupSourceElements == 0 || matrixElements > MAX_UINT32 ||
            seedElements > MAX_UINT32 ||
            static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) > sourceBufferBytes_ ||
            matrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            seedElements * sizeof(int32_t) > offsetBufferBytes_ ||
            outputStart > outputRowElements_ || groupElements > outputRowElements_ - outputStart) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> seedStorage = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> base = seedStorage;
        LocalTensor<int32_t> strides = seedStorage[alignedElements];
        source.SetSize(groupSourceElements);
        output.SetSize(static_cast<uint32_t>(matrixElements));
        seedStorage.SetSize(static_cast<uint32_t>(seedElements));
        base.SetSize(alignedElements);
        strides.SetSize(alignedElements);
        LocalTensor<uint32_t> offsets = output.ReinterpretCast<uint32_t>();
        offsets.SetSize(static_cast<uint32_t>(matrixElements));

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            SyncMte2ToV();
            if (outputBusy) {
                // 上一批 MTE3 尚读取同一 VECOUT 区时，禁止先覆写为下一批 offset。
                SyncMte3ToV();
            }
            BuildOffsetOutputAliasRows(output, base, strides, currentBatch, alignedElements);
            if (!valid_) {
                return;
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);
            SyncVToMte2();
            SyncVToMte3();

            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }
    }

    // key45：先让 MTE2 预取首批，再在 VECCALC 生成常驻 seed；随后每轮将下一批 source
    // 搬入与当前 output 的 offset 展开、Gather 和 MTE3 写回重叠。VECOUT 始终先等待上一
    // 批 MTE3 完成才被覆写，两个 source 槽则通过独立事件保证不会被提前覆盖。
    __aicore__ inline void ProcessGroupSourcePingPongOffsetOutputAlias(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t matrixElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        const uint64_t seedElements = static_cast<uint64_t>(OFFSET_OUTPUT_ALIAS_SEED_ROWS) *
            alignedElements;
        if (alignedElements == 0 || groupSourceElements == 0 || matrixElements > MAX_UINT32 ||
            seedElements > MAX_UINT32 ||
            static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) > sourceBufferBytes_ ||
            matrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            seedElements * sizeof(int32_t) > offsetBufferBytes_ ||
            outputStart > outputRowElements_ || groupElements > outputRowElements_ - outputStart) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> seedStorage = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> base = seedStorage;
        LocalTensor<int32_t> strides = seedStorage[alignedElements];
        output.SetSize(static_cast<uint32_t>(matrixElements));
        seedStorage.SetSize(static_cast<uint32_t>(seedElements));
        base.SetSize(alignedElements);
        strides.SetSize(alignedElements);
        LocalTensor<uint32_t> offsets = output.ReinterpretCast<uint32_t>();
        offsets.SetSize(static_cast<uint32_t>(matrixElements));

        event_t sourceReady[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>())};
        event_t sourceReuse[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>())};
        bool sourceReusePending[2] = {false, false};
        event_t outputReuse{};
        bool outputBusy = false;

        const uint32_t firstBatch = static_cast<uint32_t>(MinU64(batchRows_, rowCount_));
        StartSourcePingPongCopy(0, false, sourceReuse[0], sourceReady[0], inputStart,
            inputEnd, rowStart_, firstBatch, groupSourceElements);
        // 首批 MTE2 已在飞行；seed 仅依赖输入拓扑，和它在不同流水单元并行执行。
        PrepareOffsetOutputAliasSeeds(inputStart, inputEnd, groupElements);
        if (!valid_) {
            return;
        }

        uint32_t currentSlot = 0;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            const uint64_t nextBatchOffset = batchOffset + batchRows_;
            const uint32_t nextSlot = currentSlot ^ 1U;

            if (nextBatchOffset < rowCount_) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                StartSourcePingPongCopy(nextSlot, sourceReusePending[nextSlot],
                    sourceReuse[nextSlot], sourceReady[nextSlot], inputStart, inputEnd,
                    rowStart_ + nextBatchOffset, nextBatch, groupSourceElements);
            }
            if (outputBusy) {
                // 上一批 MTE3 仍读取 VECOUT 时，不能先展开下一批 offset。
                WaitFlag<HardEvent::MTE3_V>(outputReuse);
            }
            // 此处不依赖当前 source，故可与当前/下一批 MTE2 同时执行。
            BuildOffsetOutputAliasRows(output, base, strides, currentBatch, alignedElements);
            if (!valid_) {
                return;
            }
            WaitFlag<HardEvent::MTE2_V>(sourceReady[currentSlot]);
            LocalTensor<int32_t> source = GetSourcePingPong(currentSlot);
            source.SetSize(groupSourceElements);
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);

            SetFlag<HardEvent::V_MTE2>(sourceReuse[currentSlot]);
            sourceReusePending[currentSlot] = true;

            event_t vectorToMte3 = static_cast<event_t>(
                pipe_.FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) *
                    sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputReuse = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(outputReuse);
            outputBusy = true;
            currentSlot = nextSlot;
        }
        if (outputBusy) {
            WaitFlag<HardEvent::MTE3_V>(outputReuse);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (sourceReusePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(sourceReuse[slot]);
            }
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[0]);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[1]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[0]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[1]);
    }

    __aicore__ inline void ProcessGroup(uint32_t inputStart, uint32_t inputEnd,
                                        uint32_t outputStart, uint32_t groupElements,
                                        uint32_t groupSourceElements, bool& outputBusy,
                                        bool firstBatchPreloaded, uint32_t inputSplit)
    {
        if constexpr (PartitionPipeline) {
            ProcessGroupPartitioned(inputStart, inputEnd, inputSplit, outputStart,
                                    groupElements, groupSourceElements);
            return;
        }
        if constexpr (OffsetOutputAlias && SourcePingPong) {
            ProcessGroupSourcePingPongOffsetOutputAlias(inputStart, inputEnd, outputStart,
                groupElements, groupSourceElements);
            return;
        }
        if constexpr (OffsetOutputAlias) {
            ProcessGroupOffsetOutputAlias(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy);
            return;
        }
        if constexpr (TwoPlaneSource) {
            ProcessGroupTwoPlane(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy, firstBatchPreloaded);
            return;
        }
        if constexpr (RollingStagedOffsets) {
            ProcessGroupRollingStagedOffsets(inputStart, inputEnd, outputStart,
                groupElements, groupSourceElements, outputBusy);
            return;
        }
        if constexpr (DeltaStagedOffsets) {
            ProcessGroupDeltaStagedOffsets(inputStart, inputEnd, outputStart,
                groupElements, groupSourceElements, outputBusy);
            return;
        }
        if constexpr (RegisterGather) {
            ProcessGroupMicroRegister(inputStart, inputEnd, outputStart, groupElements,
                                      groupSourceElements, outputBusy);
            return;
        }
        if constexpr (RowMajorStaged) {
            ProcessGroupRowMajorStaged(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy);
            return;
        }
        if constexpr (SplitGatherOutput) {
            ProcessGroupSplitGatherOutput(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy);
            return;
        }
        if constexpr (FullStageOffsets) {
            ProcessGroupFullStageOffsets(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy);
            return;
        }
        if constexpr (CachedStagedOffsets) {
            ProcessGroupCachedOffsetStaged(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy);
            return;
        }
        if constexpr (StagedSource) {
            ProcessGroupStagedSource(inputStart, inputEnd, outputStart, groupElements,
                                     groupSourceElements, outputBusy);
            return;
        }
        if constexpr (SourcePingPong) {
            ProcessGroupSourcePingPong(inputStart, inputEnd, outputStart, groupElements,
                                        groupSourceElements);
            return;
        }
        if constexpr (OutputPingPong) {
            ProcessGroupOutputPingPong(inputStart, inputEnd, outputStart, groupElements,
                                       groupSourceElements, outputBusy);
            return;
        }
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            if (!firstBatchPreloaded || batchOffset != 0) {
                CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            }
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);
            SyncVToMte2();
            SyncVToMte3();

            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }
    }

    // key47：第一批保持标准 source 布局；每次 Gather 期间把下一批的连续后缀写到
    // 已分配 source 尾部。Gather 完成后才覆盖前缀，因此无需第二张完整 source 槽。
    __aicore__ inline void ProcessGroupSuffixPrefetch(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy,
        uint32_t planGroup, bool firstBatchPreloaded)
    {
        if (!valid_ || planGroup >= 2 * GATHER_PLAN_INPUT_CAPACITY) {
            valid_ = false;
            return;
        }
        const uint32_t suffixInputStart = tilingData_->groupSuffixInputStart[planGroup];
        const uint32_t suffixOutputStart = tilingData_->groupSuffixOutputStart[planGroup];
        const uint32_t suffixElements = tilingData_->groupSuffixElements[planGroup];
        const uint32_t suffixSourceElements =
            tilingData_->groupSuffixSourceElements[planGroup];
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t matrixElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        const uint64_t sourceCapacityElements = sourceBufferBytes_ / sizeof(int32_t);
        if (alignedElements == 0 || batchRows_ == 0 || groupElements > outputRowElements_ ||
            matrixElements > MAX_UINT32 ||
            matrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            matrixElements * sizeof(int32_t) > offsetBufferBytes_ ||
            sourceBufferBytes_ % sizeof(int32_t) != 0 ||
            sourceCapacityElements > MAX_UINT32 ||
            groupSourceElements > sourceCapacityElements) {
            valid_ = false;
            return;
        }

        // 无可用后缀的 group 仍执行与 key7 完全相同的单 source 路径。这样一个
        // subPlan 中只有部分 group 满足尾部容量时，不会改变其余 group 的语义。
        if (suffixInputStart <= inputStart || suffixInputStart >= inputEnd ||
            suffixOutputStart >= groupElements || suffixElements == 0 ||
            suffixElements != groupElements - suffixOutputStart ||
            suffixSourceElements == 0) {
            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy, firstBatchPreloaded, 0);
            return;
        }

        uint64_t prefixSourceElements = 0;
        uint64_t computedSuffixSourceElements = 0;
        uint64_t computedPrefixOutputElements = 0;
        uint64_t computedSuffixOutputElements = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t slotRawElements = static_cast<uint64_t>(sourceBatchRows_) *
                inputElements;
            if (slotRawElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            const uint64_t slotElements = AlignUpGatherElements(
                static_cast<uint32_t>(slotRawElements));
            if (input < suffixInputStart) {
                prefixSourceElements += slotElements;
                computedPrefixOutputElements += inputElements;
            } else {
                computedSuffixSourceElements += slotElements;
                computedSuffixOutputElements += inputElements;
            }
        }
        const bool layoutMatches = prefixSourceElements <= MAX_UINT32 &&
            computedSuffixSourceElements <= MAX_UINT32 &&
            prefixSourceElements + computedSuffixSourceElements == groupSourceElements &&
            computedPrefixOutputElements == suffixOutputStart &&
            computedSuffixOutputElements == suffixElements &&
            computedSuffixSourceElements == suffixSourceElements &&
            suffixSourceElements <= sourceCapacityElements - groupSourceElements;
        if (!layoutMatches) {
            // TilingData 可能来自旧二进制或不具备余量的 group；安全回退而不是让后缀
            // 偏移指向未初始化区域。
            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements, outputBusy, firstBatchPreloaded, 0);
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(static_cast<uint32_t>(sourceCapacityElements));
        output.SetSize(static_cast<uint32_t>(matrixElements));
        offsets.SetSize(static_cast<uint32_t>(matrixElements));

        if (!firstBatchPreloaded) {
            const uint32_t firstBatchRows = static_cast<uint32_t>(MinU64(batchRows_, rowCount_));
            CopyInputs(inputStart, inputEnd, rowStart_, firstBatchRows, source);
            GenerateOffsets(inputStart, inputEnd, groupElements, 0);
            if (!valid_) {
                return;
            }
        }

        // `false` 表示当前批后缀位于标准 [prefix, group) 位置；`true` 表示其被
        // 临时放在 [group, group + suffix) 尾部，因此后缀 offset 需要增加一个
        // suffixSourceElements 的 source-layout delta。
        bool currentSuffixAtTail = false;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_;) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);

            const uint64_t nextBatchOffset = batchOffset + currentBatch;
            const bool hasNextBatch = nextBatchOffset < rowCount_;
            const bool nextSuffixAtTail = !currentSuffixAtTail;
            if (hasNextBatch) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                const uint32_t suffixDestination = nextSuffixAtTail ? groupSourceElements :
                    static_cast<uint32_t>(prefixSourceElements);
                CopyInputsRange(suffixInputStart, inputEnd, rowStart_ + nextBatchOffset,
                    nextBatch, source, suffixDestination, suffixSourceElements);
                if (!valid_) {
                    return;
                }
            }

            // 当前 Gather 已读取完整 source。MTE2 队列中的前缀搬运因此可在后缀
            // 传输完成后覆盖 [0, prefix)，而 MTE3 同时写回当前输出。
            SyncVToMte2();
            SyncVToMte3();
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;

            if (hasNextBatch) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                CopyInputsRange(inputStart, suffixInputStart, rowStart_ + nextBatchOffset,
                    nextBatch, source, 0, static_cast<uint32_t>(prefixSourceElements));
                if (!valid_) {
                    return;
                }
                ApplySuffixPrefetchOffsetDelta(groupElements, suffixOutputStart, suffixElements,
                    suffixSourceElements, nextSuffixAtTail);
                if (!valid_) {
                    return;
                }
                currentSuffixAtTail = nextSuffixAtTail;
            }
            batchOffset = nextBatchOffset;
        }
    }

    // key40：一个 source stage 含两个 batchRows 平面。每个输入仅用一条
    // blockCount=2 的 DataCopyPad 搬入两个连续 GM 行块；两平面的 source 相对布局相同，
    // 因而用同一张 offset 表完成两次 Gather。尾部不构成完整平面对时回退到单平面。
    __aicore__ inline void ProcessGroupTwoPlane(uint32_t inputStart, uint32_t inputEnd,
        uint32_t outputStart, uint32_t groupElements, uint32_t groupSourceElements,
        bool& outputBusy, bool firstStagePreloaded)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t twoPlaneElements = static_cast<uint64_t>(groupSourceElements) * 2;
        const uint64_t matrixElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        if (alignedElements == 0 || groupSourceElements == 0 ||
            twoPlaneElements > MAX_UINT32 || matrixElements > MAX_UINT32 ||
            twoPlaneElements * sizeof(int32_t) > static_cast<uint64_t>(sourceBufferBytes_) * 2 ||
            matrixElements * sizeof(int32_t) > outputBufferBytes_ ||
            matrixElements * sizeof(int32_t) > offsetBufferBytes_) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(static_cast<uint32_t>(twoPlaneElements));
        output.SetSize(static_cast<uint32_t>(matrixElements));
        offsets.SetSize(static_cast<uint32_t>(matrixElements));

        for (uint64_t batchOffset = 0; batchOffset < rowCount_;) {
            const uint64_t remainingRows = rowCount_ - batchOffset;
            const bool hasPlanePair = remainingRows >= 2 * batchRows_;
            const uint32_t firstRows = static_cast<uint32_t>(MinU64(batchRows_, remainingRows));
            const uint64_t batchStart = rowStart_ + batchOffset;
            if (!firstStagePreloaded || batchOffset != 0) {
                if (hasPlanePair) {
                    CopyInputsTwoPlanes(inputStart, inputEnd, batchStart, source,
                        groupSourceElements);
                } else {
                    CopyInputs(inputStart, inputEnd, batchStart, firstRows, source);
                }
            }
            firstStagePreloaded = false;
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, firstRows * alignedElements);
            if (!hasPlanePair) {
                // 单平面尾批已不再读取 source，先归还 source 再提交 MTE3，令下一 stage
                // 的连续 MTE2 能与当前输出写回重叠。
                SyncVToMte2();
            }
            SyncVToMte3();
            const DataCopyExtParams firstCopyOutParams{
                static_cast<uint16_t>(firstRows),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[batchStart * outputRowElements_ + outputStart], output,
                firstCopyOutParams);
            outputBusy = true;

            if (!hasPlanePair) {
                batchOffset += firstRows;
                continue;
            }

            // 第二张平面仍使用同一 source 缓冲，因此先等待 output 槽释放，随后再读取
            // plane1；完成后才允许下一 stage 覆盖两张 source 平面。
            SyncMte3ToV();
            GatherSegmented(output, source[groupSourceElements], offsets,
                batchRows_ * alignedElements);
            SyncVToMte2();
            SyncVToMte3();
            const DataCopyExtParams secondCopyOutParams{
                static_cast<uint16_t>(batchRows_),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            const uint64_t secondOutputOffset = (batchStart + batchRows_) * outputRowElements_ +
                outputStart;
            DataCopyPad(outputGm_[secondOutputOffset], output, secondCopyOutParams);
            outputBusy = true;
            batchOffset += 2 * batchRows_;
        }
    }

    __aicore__ inline void ProcessGroupSplitGatherOutput(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        // 大批 Gather 的行数由 Host 的 UB 计划决定；按半批写回使小段输出延续较短 MTE3 stride。
        const uint32_t writeBatchRows = (batchRows_ + 1U) / 2U;
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);
            // Gather 已读完 source，下一批 MTE2 可与当前分段 MTE3 写回重叠。
            SyncVToMte2();
            SyncVToMte3();

            const uint32_t sourceStride =
                (alignedElements - groupElements) / GATHER_BLOCK_ELEMENTS;
            for (uint32_t writeOffset = 0; writeOffset < currentBatch;
                 writeOffset += writeBatchRows) {
                const uint32_t writeRows = static_cast<uint32_t>(MinU64(
                    writeBatchRows, currentBatch - writeOffset));
                const uint64_t outputOffset = (batchStart + writeOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(writeRows),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), sourceStride,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output[writeOffset * alignedElements],
                    copyOutParams);
            }
            outputBusy = true;
        }
    }

    __aicore__ inline void ProcessGroupStagedSource(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        // 一个较大的 source stage 可供多个小输出批复用；MTE3 的 blockCount 始终由
        // batchRows_ 决定，避免把 key17 的长跨行写回重新引入关键路径。
        for (uint64_t stageOffset = 0; stageOffset < rowCount_; stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputs(inputStart, inputEnd, stageStart, stageRows, source);
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                const uint32_t offsetTableRows = CompactStageOffsets ? currentBatch : batchRows_;
                GenerateOffsets(inputStart, inputEnd, groupElements, sourceRowOffset,
                    offsetTableRows);
                // offset 表与 output 不重叠，先完成下一小批的建表，再等上一小批的
                // MTE3 释放 output，尽量覆盖写回等待。
                if (outputBusy) {
                    SyncMte3ToV();
                }
                GatherSegmented(output, source, offsets, currentBatch * alignedElements);
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
                outputBusy = true;
            }
            // stage 内全部 Gather 已完成，下一 stage 的 MTE2 才可覆盖 source。
            SyncVToMte2();
        }
    }

    // offset 表的第 r 行记录 source stage 内第 r 行的数据地址。大 source stage 的
    // 后续输出子批只需把有效行按每个输入的行宽平移，不需要重新 CreateVecIndex、Muls
    // 或再传播整张二维表。恢复后下一 source stage 可直接复用同一张基准表。
    __aicore__ inline void ShiftRollingOffsetRows(const LocalTensor<int32_t>& offsets,
        uint32_t activeRows, uint32_t alignedElements, uint32_t inputStart,
        uint32_t inputEnd, uint32_t groupElements, int64_t rowShift)
    {
        if (activeRows == 0 || activeRows > batchRows_ || alignedElements < groupElements) {
            valid_ = false;
            return;
        }
        if (rowShift == 0) {
            return;
        }

        uint64_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t adjustment = rowShift * static_cast<int64_t>(inputElements) *
                    static_cast<int64_t>(sizeof(int32_t));
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE ||
                    outputPrefix > MAX_UINT32 || inputElements > groupElements - outputPrefix) {
                    valid_ = false;
                    return;
                }
                for (uint32_t row = 0; row < activeRows; ++row) {
                    const uint64_t rowOffset = static_cast<uint64_t>(row) * alignedElements +
                        outputPrefix;
                    if (rowOffset > MAX_UINT32 ||
                        inputElements > MAX_UINT32 - static_cast<uint32_t>(rowOffset)) {
                        valid_ = false;
                        return;
                    }
                    AddScalarVector(offsets, static_cast<uint32_t>(rowOffset), inputElements,
                        static_cast<int32_t>(adjustment));
                }
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void ProcessGroupRollingStagedOffsets(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t offsetElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        if (sourceBatchRows_ <= batchRows_ || alignedElements == 0 ||
            offsetElements > MAX_UINT32 ||
            offsetElements * sizeof(int32_t) > offsetBufferBytes_ ||
            static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) > sourceBufferBytes_ ||
            static_cast<uint64_t>(batchRows_) * alignedElements * sizeof(int32_t) >
                outputBufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetStorage = offsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetStorage.ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(static_cast<uint32_t>(offsetElements));

        bool offsetsReady = false;
        for (uint64_t stageOffset = 0; stageOffset < rowCount_;
             stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputs(inputStart, inputEnd, stageStart, stageRows, source);
            // 首次建表与正在进行的连续 MTE2 搬入重叠；之后保留基准表直到该 group 结束。
            if (!offsetsReady) {
                GenerateOffsets(inputStart, inputEnd, groupElements, 0);
                offsetsReady = true;
            }
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                const bool shifted = sourceRowOffset != 0;
                if (shifted) {
                    ShiftRollingOffsetRows(offsetStorage, currentBatch, alignedElements,
                        inputStart, inputEnd, groupElements,
                        static_cast<int64_t>(sourceRowOffset));
                    if (!valid_) {
                        return;
                    }
                }
                // offset 的滚动更新不触碰 output，因此可与上一子批 MTE3 写回重叠。
                if (outputBusy) {
                    SyncMte3ToV();
                }
                GatherSegmented(output, source, offsets, currentBatch * alignedElements);
                const bool sourceStageConsumed = sourceRowOffset + currentBatch >= stageRows;
                if (sourceStageConsumed) {
                    // 最后一次 Gather 已不再读取 source；下一 stage 的连续 MTE2 可以先行。
                    SyncVToMte2();
                }
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
                if (shifted) {
                    // MTE3 已收到 output 的就绪事件；恢复只访问 offset，可与写回并行。
                    ShiftRollingOffsetRows(offsetStorage, currentBatch, alignedElements,
                        inputStart, inputEnd, groupElements,
                        -static_cast<int64_t>(sourceRowOffset));
                    if (!valid_) {
                        return;
                    }
                }
                outputBusy = true;
            }
        }
    }

    __aicore__ inline void ProcessGroupFullStageOffsets(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t offsetTableElements = static_cast<uint64_t>(sourceBatchRows_) *
            alignedElements;
        if (sourceBatchRows_ <= batchRows_ ||
            offsetTableElements > MAX_UINT32 ||
            offsetTableElements * sizeof(int32_t) > offsetBufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetTable = offsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetTable.ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(static_cast<uint32_t>(offsetTableElements));

        // source 在每个 stage 内都采用相同的分输入连续布局，因此整张 offset 表只需
        // 随 group 生成一次；后续 Gather 直接使用对应行的切片。
        GenerateOffsetsTo(offsetTable, sourceBatchRows_, inputStart, inputEnd,
            groupElements, 0);
        if (!valid_) {
            return;
        }

        for (uint64_t stageOffset = 0; stageOffset < rowCount_;
             stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputs(inputStart, inputEnd, stageStart, stageRows, source);
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                if (outputBusy) {
                    SyncMte3ToV();
                }
                GatherSegmented(output, source, offsets[sourceRowOffset * alignedElements],
                    currentBatch * alignedElements);
                const bool sourceStageConsumed = sourceRowOffset + currentBatch >= stageRows;
                if (sourceStageConsumed) {
                    // 最后一批 Gather 已读完 source，可让下一 stage 的连续 MTE2 与当前
                    // MTE3 写回并行，而不会覆盖仍被 Gather 引用的 UB 数据。
                    SyncVToMte2();
                }
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
                outputBusy = true;
            }
        }
    }

    __aicore__ inline void ProcessGroupCachedOffsetStaged(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint32_t offsetTableElements = batchRows_ * alignedElements;
        if (sourceBatchRows_ <= batchRows_ ||
            static_cast<uint64_t>(sourceBatchRows_) >
                static_cast<uint64_t>(batchRows_) * 2 ||
            static_cast<uint64_t>(offsetTableElements) * 2 * sizeof(int32_t) >
                offsetBufferBytes_) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetTables = offsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> firstOffsets = offsetTables.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> secondOffsets =
            offsetTables[offsetTableElements].ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        firstOffsets.SetSize(offsetTableElements);
        secondOffsets.SetSize(offsetTableElements);

        // 两张表仅随 group 改变：第 0 子批和第 batchRows 子批可跨全部 source stage 复用。
        GenerateOffsetsTo(offsetTables, batchRows_, inputStart, inputEnd, groupElements, 0);
        GenerateOffsetsTo(offsetTables[offsetTableElements], batchRows_, inputStart, inputEnd,
            groupElements, batchRows_);

        for (uint64_t stageOffset = 0; stageOffset < rowCount_; stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputs(inputStart, inputEnd, stageStart, stageRows, source);
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                if (outputBusy) {
                    SyncMte3ToV();
                }
                if (sourceRowOffset == 0) {
                    GatherSegmented(output, source, firstOffsets,
                        currentBatch * alignedElements);
                } else {
                    GatherSegmented(output, source, secondOffsets,
                        currentBatch * alignedElements);
                }
                const bool sourceStageConsumed = sourceRowOffset + currentBatch >= stageRows;
                if (sourceStageConsumed) {
                    // 最后一个子批已完成 Vector 读取；先释放 source，让下一 stage 的 MTE2
                    // 能与当前子批的 MTE3 写回重叠。
                    SyncVToMte2();
                }
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
                outputBusy = true;
            }
        }
    }

    __aicore__ inline void CopyInputsRowMajor(uint32_t inputStart, uint32_t inputEnd,
        uint64_t rowStart, uint32_t rows, uint32_t sourceRowElements,
        const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements == 0) {
                continue;
            }
            const uint32_t alignedInputElements = AlignUpGatherElements(inputElements);
            if (sourceRowElements < alignedInputElements ||
                (sourceRowElements - alignedInputElements) % GATHER_BLOCK_ELEMENTS != 0) {
                valid_ = false;
                return;
            }
            const uint32_t rightPadding = alignedInputElements - inputElements;
            inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
            const DataCopyExtParams copyInParams{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(inputElements * sizeof(int32_t)), 0,
                (sourceRowElements - alignedInputElements) / GATHER_BLOCK_ELEMENTS, 0};
            const DataCopyPadExtParams<int32_t> padParams{
                true, 0, static_cast<uint8_t>(rightPadding), 0};
            DataCopyPad(source[sourceBase], inputGm_[rowStart * inputElements],
                copyInParams, padParams);
            sourceBase += alignedInputElements;
        }
        if (sourceBase != sourceRowElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void ProcessGroupRowMajorStaged(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        uint32_t sourceRowElements = 0;
        if (!GetRowMajorSourceRowElements(inputStart, inputEnd, groupElements,
            sourceRowElements) ||
            static_cast<uint64_t>(sourceBatchRows_) * sourceRowElements !=
                groupSourceElements) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        for (uint64_t stageOffset = 0; stageOffset < rowCount_; stageOffset += sourceBatchRows_) {
            const uint32_t stageRows = static_cast<uint32_t>(MinU64(
                sourceBatchRows_, rowCount_ - stageOffset));
            const uint64_t stageStart = rowStart_ + stageOffset;
            CopyInputsRowMajor(inputStart, inputEnd, stageStart, stageRows,
                sourceRowElements, source);
            if (!valid_) {
                return;
            }
            SyncMte2ToV();

            for (uint32_t sourceRowOffset = 0; sourceRowOffset < stageRows;
                 sourceRowOffset += batchRows_) {
                const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, stageRows - sourceRowOffset));
                if (outputBusy) {
                    SyncMte3ToV();
                }
                GatherSegmented(output, source[sourceRowOffset * sourceRowElements], offsets,
                    currentBatch * alignedElements);
                SyncVToMte3();

                const uint64_t outputOffset = (stageStart + sourceRowOffset) *
                    outputRowElements_ + outputStart;
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(currentBatch),
                    static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                    static_cast<uint32_t>((outputRowElements_ - groupElements) *
                        sizeof(int32_t)), 0};
                DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
                outputBusy = true;
            }
            // 当前 stage 的所有 Gather 已完成读取 source，下一 stage 才可覆盖该 UB 区域。
            SyncVToMte2();
        }
    }

    __aicore__ inline LocalTensor<int32_t> GetSourcePingPong(uint32_t sourceSlot)
    {
        LocalTensor<int32_t> source = sourcePingBuffer_.Get<int32_t>();
        const uint32_t slotElements = sourceBufferBytes_ / sizeof(int32_t);
        return sourceSlot == 0 ? source : source[slotElements];
    }

    __aicore__ inline void StartSourcePingPongCopy(uint32_t sourceSlot,
        bool waitForReuse, event_t reuseEvent, event_t readyEvent,
        uint32_t inputStart, uint32_t inputEnd, uint64_t batchStart,
        uint32_t currentBatch, uint32_t groupSourceElements)
    {
        // 同一 source 槽在两批之后才复用，等待前一次 Gather 完成后再让 MTE2 覆盖。
        if (waitForReuse) {
            WaitFlag<HardEvent::V_MTE2>(reuseEvent);
        }
        LocalTensor<int32_t> source = GetSourcePingPong(sourceSlot);
        source.SetSize(groupSourceElements);
        CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
        SetFlag<HardEvent::MTE2_V>(readyEvent);
    }

    __aicore__ inline void ProcessGroupSourcePingPong(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        // 两个 source 槽可能同时存在未完成的搬入/计算依赖，必须持有不同 event ID。
        // FetchEventID 不会保留 ID，两个未完成依赖共用它会使事件边无法一一匹配。
        event_t sourceReady[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>())};
        event_t sourceReuse[2] = {
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>()),
            static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>())};
        bool sourceReusePending[2] = {false, false};
        event_t outputReuse{};
        bool outputBusy = false;

        const uint32_t firstBatch = static_cast<uint32_t>(MinU64(batchRows_, rowCount_));
        StartSourcePingPongCopy(0, false, sourceReuse[0], sourceReady[0],
            inputStart, inputEnd, rowStart_, firstBatch, groupSourceElements);

        uint32_t currentSlot = 0;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            const uint64_t nextBatchOffset = batchOffset + batchRows_;
            const uint32_t nextSlot = currentSlot ^ 1U;

            // 先把下一批放到另一个 source 槽，随后才等待当前批就绪；这样 MTE2 与
            // 当前 Gather 并行，而不是像单 source 槽那样被 Vector 串行化。
            if (nextBatchOffset < rowCount_) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    batchRows_, rowCount_ - nextBatchOffset));
                StartSourcePingPongCopy(nextSlot, sourceReusePending[nextSlot],
                    sourceReuse[nextSlot], sourceReady[nextSlot], inputStart, inputEnd,
                    rowStart_ + nextBatchOffset, nextBatch, groupSourceElements);
            }

            WaitFlag<HardEvent::MTE2_V>(sourceReady[currentSlot]);
            if (outputBusy) {
                WaitFlag<HardEvent::MTE3_V>(outputReuse);
            }
            LocalTensor<int32_t> source = GetSourcePingPong(currentSlot);
            source.SetSize(groupSourceElements);
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);

            // Gather 已不再读取当前 source 槽，通知 MTE2 可在两批后复用该槽。
            SetFlag<HardEvent::V_MTE2>(sourceReuse[currentSlot]);
            sourceReusePending[currentSlot] = true;

            // output 仍为单槽，MTE3 与下一批 MTE2 重叠；下一次 Gather 前才等待其写回，
            // 避免引入会放大写回次数的输出拆分。
            event_t vectorToMte3 = static_cast<event_t>(
                pipe_.FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputReuse = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(outputReuse);
            outputBusy = true;
            currentSlot = nextSlot;
        }
        if (outputBusy) {
            WaitFlag<HardEvent::MTE3_V>(outputReuse);
        }
        // 最后一批不会再触发 MTE2 复用等待；在离开本 group 前显式消费其
        // V->MTE2 依赖，防止下一 group 的首批覆盖仍被 Gather 读取的 source 槽。
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (sourceReusePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(sourceReuse[slot]);
            }
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[0]);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady[1]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[0]);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReuse[1]);
    }

    __aicore__ inline void ProcessGroupOutputPingPong(
        uint32_t inputStart, uint32_t inputEnd, uint32_t outputStart,
        uint32_t groupElements, uint32_t groupSourceElements, bool& outputBusy)
    {
        (void)outputBusy;
        if (pipelineBatchRows_ == 0) {
            return;
        }
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        offsets.SetSize(batchRows_ * alignedElements);
        const uint32_t slotRows = pipelineBatchRows_ / 2;

        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += pipelineBatchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                pipelineBatchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            SyncMte2ToV();

            const uint32_t firstRows = static_cast<uint32_t>(MinU64(slotRows, currentBatch));
            CopyOutputSlot(source, offsets, 0, firstRows, alignedElements, groupElements,
                           batchStart, outputStart);
            if (firstRows < currentBatch) {
                CopyOutputSlot(source, offsets, firstRows, currentBatch - firstRows,
                               alignedElements, groupElements, batchStart, outputStart);
            }
            // 两个输出槽都已完成 Vector 读取，下一批 MTE2 可以覆盖 source，同时 MTE3 继续写回。
            SyncVToMte2();
        }
    }

    __aicore__ inline void CopyOutputSlot(const LocalTensor<int32_t>& source,
                                          const LocalTensor<uint32_t>& offsets,
                                          uint32_t rowOffset, uint32_t rows,
                                          uint32_t alignedElements, uint32_t groupElements,
                                          uint64_t batchStart, uint32_t outputStart)
    {
        if (rows == 0) {
            return;
        }
        LocalTensor<int32_t> output = outputQueue_.AllocTensor<int32_t>();
        output.SetSize(rows * alignedElements);
        GatherSegmented(output, source, offsets[rowOffset * alignedElements], rows * alignedElements);
        outputQueue_.EnQue(output);

        output = outputQueue_.DeQue<int32_t>();
        const uint64_t outputOffset =
            (batchStart + rowOffset) * outputRowElements_ + outputStart;
        const DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(groupElements * sizeof(int32_t)),
            (alignedElements - groupElements) / GATHER_BLOCK_ELEMENTS,
            static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
        DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void GatherSegmented(const LocalTensor<int32_t>& output,
                                           const LocalTensor<int32_t>& source,
                                           const LocalTensor<uint32_t>& offsets,
                                           uint32_t count)
    {
        // A2 的 count 重载内部 repeat 为 uint8_t；16320=255*64，且切片起点保持 32B 对齐。
        for (uint32_t done = 0; done < count; done += GATHER_MAX_ELEMENTS_PER_CALL) {
            const uint32_t current = static_cast<uint32_t>(MinU64(
                GATHER_MAX_ELEMENTS_PER_CALL, count - done));
            Gather(output[done], source, offsets[done], 0, current);
        }
    }

    __aicore__ inline void CopyInputsTwoPlanes(uint32_t inputStart, uint32_t inputEnd,
        uint64_t rowStart, const LocalTensor<int32_t>& source, uint32_t groupSourceElements)
    {
        if (batchRows_ == 0 || inputEnd <= inputStart || inputEnd > inputCount_ ||
            groupSourceElements == 0) {
            valid_ = false;
            return;
        }
        uint64_t sourceBase = 0;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t planeElements = static_cast<uint64_t>(batchRows_) * inputElements;
            const uint64_t planeBytes = planeElements * sizeof(int32_t);
            if (planeElements > MAX_UINT32 || planeBytes > MAX_UINT32 ||
                planeBytes > MAX_BLOCK_LEN) {
                valid_ = false;
                return;
            }
            const uint64_t sourceSlotElements = AlignUpGatherElements(
                static_cast<uint32_t>(planeElements));
            if (sourceSlotElements > groupSourceElements || sourceBase > groupSourceElements ||
                sourceSlotElements > groupSourceElements - sourceBase) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                const uint64_t dstGapElements = groupSourceElements - sourceSlotElements;
                if (dstGapElements % GATHER_BLOCK_ELEMENTS != 0 ||
                    dstGapElements / GATHER_BLOCK_ELEMENTS > MAX_UINT32) {
                    valid_ = false;
                    return;
                }
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                // 两个 GM block 分别是 [rowStart, rowStart+B) 和 [rowStart+B,rowStart+2B)。
                // srcStride=0 表示 GM 连续；dstStride 将第二块跳到同一输入的 plane1 槽位。
                const DataCopyExtParams copyInParams{
                    2, static_cast<uint32_t>(planeBytes), 0,
                    static_cast<uint32_t>(dstGapElements / GATHER_BLOCK_ELEMENTS), 0};
                DataCopyPad(source[static_cast<uint32_t>(sourceBase)],
                    inputGm_[rowStart * inputElements], copyInParams, padParams);
            }
            sourceBase += sourceSlotElements;
        }
        if (sourceBase != groupSourceElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void CopyInputs(uint32_t inputStart, uint32_t inputEnd,
                                      uint64_t rowStart, uint32_t rows,
                                      const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(static_cast<uint64_t>(rows) * inputElements *
                        sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(source[sourceBase], inputGm_[rowStart * inputElements],
                            copyInParams, padParams);
            }
            // 尾 stage 也沿用完整 source stage 布局，保证各输入段的 UB 起点固定。
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(
                static_cast<uint64_t>(sourceBatchRows_) * inputElements));
        }
    }

    // 将连续输入子范围写入指定 source 槽位。目标范围的槽宽始终按完整
    // sourceBatchRows_ 预留，因此尾批只缩短 GM 有效读长，不改变随后 Gather 的地址表。
    __aicore__ inline void CopyInputsRange(uint32_t inputStart, uint32_t inputEnd,
        uint64_t rowStart, uint32_t rows, const LocalTensor<int32_t>& source,
        uint32_t destinationBase, uint32_t destinationElements)
    {
        if (inputEnd <= inputStart || inputEnd > inputCount_ || rows == 0 ||
            sourceBatchRows_ == 0) {
            valid_ = false;
            return;
        }
        uint64_t sourceOffset = destinationBase;
        const uint64_t destinationEnd = static_cast<uint64_t>(destinationBase) +
            destinationElements;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t slotRawElements = static_cast<uint64_t>(sourceBatchRows_) *
                inputElements;
            if (slotRawElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            const uint32_t slotElements = AlignUpGatherElements(
                static_cast<uint32_t>(slotRawElements));
            if (sourceOffset > destinationEnd || slotElements > destinationEnd - sourceOffset) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                const uint64_t copyBytes = static_cast<uint64_t>(rows) * inputElements *
                    sizeof(int32_t);
                if (copyBytes > MAX_UINT32 || copyBytes > MAX_BLOCK_LEN) {
                    valid_ = false;
                    return;
                }
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
                DataCopyPad(source[static_cast<uint32_t>(sourceOffset)],
                    inputGm_[rowStart * inputElements], copyInParams, padParams);
            }
            sourceOffset += slotElements;
        }
        if (sourceOffset != destinationEnd) {
            valid_ = false;
        }
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatRowGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t planIndex_{0};
    uint32_t batchRows_{0};
    uint32_t sourceBatchRows_{0};
    uint32_t groupCount_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint32_t offsetBufferBytes_{0};
    uint32_t sourcePart0BufferBytes_{0};
    uint32_t sourcePart1BufferBytes_{0};
    uint32_t outputPart0BufferBytes_{0};
    uint32_t outputPart1BufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    GlobalTensor<int32_t> sharedSeedGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECIN> sourcePingBuffer_;
    TBuf<TPosition::VECIN> sourcePart0Buffer_;
    TBuf<TPosition::VECIN> sourcePart1Buffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECOUT> outputPart0Buffer_;
    TBuf<TPosition::VECOUT> outputPart1Buffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
    TBuf<TPosition::VECCALC> offsetPart0Buffer_;
    TBuf<TPosition::VECCALC> offsetPart1Buffer_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    uint32_t pipelineBatchRows_{0};
    uint32_t outputSlotBytes_{0};
};

using KernelConcatRowGatherTwoPlane = KernelConcatRowGather<
    false, false, false, false, false, false,
    false, false, false, false, false, false,
    false, false, false, false, false, false, true>;

using KernelConcatRowGatherAdaptiveGroupBatch = KernelConcatRowGather<
    false, false, false, false, false, false,
    false, false, false, false, false, true,
    false, false, false, false, false, false, false, true>;

using KernelConcatRowGatherOffsetOutputAlias = KernelConcatRowGather<
    false, false, false, false, false, false, false,
    false, false, false, false, false, false, false,
    false, false, false, false, false, false, true>;

using KernelConcatRowGatherSourcePingPongOffsetOutputAlias = KernelConcatRowGather<
    false, true, false, false, false, false, false,
    false, false, false, false, false, false, false,
    false, false, false, false, false, false, true>;

using KernelConcatRowGatherSuffixPrefetch = KernelConcatRowGather<
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, true>;

// key46：两个相邻输入组共用一张 output UB 矩阵；每组只生成一次 base/stride，随后
// 按 64-lane 片段展开 offset，并用 Gather 的 dstRepStride 直接写入对应输出列。
class KernelConcatPairGather {
public:
    __aicore__ inline explicit KernelConcatPairGather(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatPairGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if (tilingData_ == nullptr || blockIdx_ >= tilingData_->taskCount) {
            return;
        }
        inputCount_ = tilingData_->inputCount;
        inputStart_ = tilingData_->taskInputStart[blockIdx_];
        inputSplit_ = tilingData_->taskInputSplit[blockIdx_];
        inputEnd_ = tilingData_->taskInputEnd[blockIdx_];
        outputStart_ = tilingData_->taskOutputStart[blockIdx_];
        group0Elements_ = tilingData_->taskGroup0Elements[blockIdx_];
        group1Elements_ = tilingData_->taskGroup1Elements[blockIdx_];
        pairElements_ = tilingData_->taskPairElements[blockIdx_];
        sourceElements_ = tilingData_->taskSourceElements[blockIdx_];
        batchRows_ = tilingData_->taskBatchRows[blockIdx_];
        sourceBufferBytes_ = tilingData_->taskSourceBufferBytes[blockIdx_];
        outputBufferBytes_ = tilingData_->taskOutputBufferBytes[blockIdx_];
        offsetBufferBytes_ = tilingData_->taskOffsetBufferBytes[blockIdx_];
        rowStart_ = tilingData_->taskRowStart[blockIdx_];
        rowCount_ = tilingData_->taskRowCount[blockIdx_];
        outputRowElements_ = tilingData_->outputRowElements;
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        group0Aligned_ = AlignUpGatherElements(group0Elements_);
        group1Aligned_ = AlignUpGatherElements(group1Elements_);
        pairAligned_ = AlignUpGatherElements(pairElements_);
        maxGroupAligned_ = ((group0Aligned_ > group1Aligned_ ? group0Aligned_ : group1Aligned_) +
            MICRO_GATHER_VF - 1) / MICRO_GATHER_VF * MICRO_GATHER_VF;
        const uint64_t stagingElements = static_cast<uint64_t>(batchRows_ + 1) * MICRO_GATHER_VF;
        const uint64_t metadataElements = static_cast<uint64_t>(2) * maxGroupAligned_ +
            stagingElements;
        valid_ = tilingData_->taskCount != 0 && tilingData_->taskCount == blockNum_ &&
            inputCount_ != 0 && inputStart_ < inputSplit_ && inputSplit_ < inputEnd_ &&
            inputEnd_ <= inputCount_ && rowCount_ != 0 && rowStart_ <= tilingData_->outer &&
            rowCount_ <= tilingData_->outer - rowStart_ && tilingData_->dtypeSize == sizeof(int32_t) &&
            batchRows_ != 0 && batchRows_ <= GATHER_MAX_REPEAT_TIMES && group0Elements_ != 0 &&
            group1Elements_ != 0 && pairElements_ == group0Elements_ + group1Elements_ &&
            group0Elements_ % GATHER_BLOCK_ELEMENTS == 0 && outputStart_ <= outputRowElements_ &&
            pairElements_ <= outputRowElements_ - outputStart_ && pairAligned_ != 0 &&
            maxGroupAligned_ != 0 && metadataElements <= MAX_UINT32 &&
            metadataElements * sizeof(int32_t) <= offsetBufferBytes_ &&
            static_cast<uint64_t>(pairAligned_) * batchRows_ * sizeof(int32_t) <= outputBufferBytes_ &&
            sourceElements_ != 0 && static_cast<uint64_t>(sourceElements_) * sizeof(int32_t) <=
                sourceBufferBytes_ && pairAligned_ / GATHER_BLOCK_ELEMENTS <= MAX_UINT16_VALUE;
        if (!valid_) {
            return;
        }
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
        pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> metadata = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> base = metadata;
        LocalTensor<int32_t> strides = metadata[maxGroupAligned_];
        LocalTensor<int32_t> staging = metadata[2 * maxGroupAligned_];
        source.SetSize(sourceElements_);
        output.SetSize(batchRows_ * pairAligned_);
        metadata.SetSize(static_cast<uint32_t>(2 * maxGroupAligned_ +
            static_cast<uint64_t>(batchRows_ + 1) * MICRO_GATHER_VF));
        base.SetSize(maxGroupAligned_);
        strides.SetSize(maxGroupAligned_);
        staging.SetSize(static_cast<uint32_t>((batchRows_ + 1) * MICRO_GATHER_VF));

        uint32_t group0SourceElements = 0;
        if (!CalculateSourceElements(inputStart_, inputSplit_, group0SourceElements) ||
            group0SourceElements > sourceElements_) {
            valid_ = false;
            return;
        }
        GenerateGroupBaseStride(base, strides, inputStart_, inputSplit_, group0Elements_, 0);
        if (!valid_) {
            return;
        }
        bool outputBusy = false;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart_, inputSplit_, batchStart, currentBatch, source, 0);
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherGroup(output, source, staging, base, strides, currentBatch,
                group0Elements_, 0);
            SyncVToMte2();
            CopyInputs(inputSplit_, inputEnd_, batchStart, currentBatch, source, 0);
            SyncMte2ToV();
            GenerateGroupBaseStride(base, strides, inputSplit_, inputEnd_, group1Elements_, 0);
            if (!valid_) {
                return;
            }
            GatherGroup(output, source, staging, base, strides, currentBatch,
                group1Elements_, group0Aligned_);
            SyncVToMte2();
            SyncVToMte3();
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(pairElements_ * sizeof(int32_t)), 0,
                static_cast<uint32_t>((outputRowElements_ - pairElements_) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[(batchStart * outputRowElements_) + outputStart_],
                output, copyOutParams);
            outputBusy = true;
        }
        if (outputBusy) {
            SyncMte3ToV();
        }
    }

private:
    __aicore__ inline void FillConstantVector(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t head = static_cast<uint32_t>(MinU64(count,
                GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << head) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += head;
            count -= head;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddScalarVector(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t head = static_cast<uint32_t>(MinU64(count,
                GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << head) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                mask, 1, {1, 1, 8, 8});
            start += head;
            count -= head;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline bool CalculateSourceElements(uint32_t begin, uint32_t end,
        uint32_t& result)
    {
        uint64_t total = 0;
        for (uint32_t input = begin; input < end; ++input) {
            total += AlignUpGatherElements(static_cast<uint32_t>(static_cast<uint64_t>(
                batchRows_) * tilingData_->inputRowElements[input]));
            if (total > MAX_UINT32) {
                return false;
            }
        }
        result = static_cast<uint32_t>(total);
        return result != 0;
    }

    __aicore__ inline void GenerateGroupBaseStride(const LocalTensor<int32_t>& base,
        const LocalTensor<int32_t>& strides, uint32_t begin, uint32_t end,
        uint32_t groupElements, uint32_t sourceBaseOffset)
    {
        const uint32_t aligned = (AlignUpGatherElements(groupElements) +
            MICRO_GATHER_VF - 1) / MICRO_GATHER_VF * MICRO_GATHER_VF;
        if (begin >= end || end > inputCount_ || groupElements == 0 || aligned > maxGroupAligned_) {
            valid_ = false;
            return;
        }
        CreateVecIndex(base, static_cast<int32_t>(0), aligned);
        Muls(base, base, static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(aligned));
        uint64_t outputPrefix = 0;
        uint64_t sourceBase = sourceBaseOffset;
        for (uint32_t input = begin; input < end; ++input) {
            const uint32_t elements = tilingData_->inputRowElements[input];
            if (elements != 0) {
                const int64_t adjustment = (static_cast<int64_t>(sourceBase) -
                    static_cast<int64_t>(outputPrefix)) * sizeof(int32_t);
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE ||
                    outputPrefix > groupElements || elements > groupElements - outputPrefix) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(base, static_cast<uint32_t>(outputPrefix), elements,
                    static_cast<int32_t>(adjustment));
                FillConstantVector(strides, static_cast<uint32_t>(outputPrefix), elements,
                    static_cast<int32_t>(static_cast<uint64_t>(elements) * sizeof(int32_t)));
            }
            outputPrefix += elements;
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(static_cast<uint64_t>(
                batchRows_) * elements));
        }
        if (outputPrefix != groupElements || sourceBase > MAX_UINT32) {
            valid_ = false;
            return;
        }
        FillConstantVector(base, groupElements, aligned - groupElements, 0);
        FillConstantVector(strides, groupElements, aligned - groupElements, 0);
    }

    __aicore__ inline void ExpandChunk(const LocalTensor<int32_t>& staging,
        const LocalTensor<int32_t>& base, const LocalTensor<int32_t>& strides,
        uint32_t chunkOffset, uint32_t chunkElements, uint32_t rows)
    {
        if (chunkElements == 0 || chunkElements > MICRO_GATHER_VF || rows == 0 ||
            rows > GATHER_MAX_REPEAT_TIMES) {
            valid_ = false;
            return;
        }
        Adds(staging, base[chunkOffset], static_cast<int32_t>(0),
            static_cast<int32_t>(MICRO_GATHER_VF));
        for (uint32_t row = 1; row < rows; ++row) {
            Add(staging[row * MICRO_GATHER_VF], staging[(row - 1) * MICRO_GATHER_VF],
                strides[chunkOffset], static_cast<int32_t>(MICRO_GATHER_VF));
        }
        if (chunkElements < MICRO_GATHER_VF) {
            FillConstantVector(staging, chunkElements, MICRO_GATHER_VF - chunkElements, 0);
        }
    }

    __aicore__ inline void GatherGroup(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& source, const LocalTensor<int32_t>& staging,
        const LocalTensor<int32_t>& base, const LocalTensor<int32_t>& strides,
        uint32_t rows, uint32_t groupElements, uint32_t outputOffset)
    {
        LocalTensor<uint32_t> offsets = staging.ReinterpretCast<uint32_t>();
        const uint16_t dstStride = static_cast<uint16_t>(pairAligned_ / GATHER_BLOCK_ELEMENTS);
        for (uint32_t chunk = 0; chunk < groupElements; chunk += MICRO_GATHER_VF) {
            const uint32_t current = static_cast<uint32_t>(MinU64(
                MICRO_GATHER_VF, groupElements - chunk));
            ExpandChunk(staging, base, strides, chunk, current, rows);
            if (!valid_) {
                return;
            }
            Gather(output[outputOffset + chunk], source, offsets, 0,
                static_cast<uint64_t>(current), static_cast<uint8_t>(rows), dstStride);
        }
    }

    __aicore__ inline void CopyInputs(uint32_t begin, uint32_t end, uint64_t rowStart,
        uint32_t rows, const LocalTensor<int32_t>& source, uint32_t sourceOffset)
    {
        uint32_t cursor = sourceOffset;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = begin; input < end; ++input) {
            const uint32_t elements = tilingData_->inputRowElements[input];
            const uint32_t slot = AlignUpGatherElements(static_cast<uint32_t>(static_cast<uint64_t>(
                batchRows_) * elements));
            const uint64_t payload = static_cast<uint64_t>(rows) * elements * sizeof(int32_t);
            if (cursor > sourceElements_ || slot > sourceElements_ - cursor || payload > MAX_UINT32) {
                valid_ = false;
                return;
            }
            if (elements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams params{1, static_cast<uint32_t>(payload), 0, 0, 0};
                DataCopyPad(source[cursor], inputGm_[rowStart * elements], params, padParams);
            }
            cursor += slot;
        }
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t id = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(id); WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void SyncVToMte2()
    {
        event_t id = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(id); WaitFlag<HardEvent::V_MTE2>(id);
    }
    __aicore__ inline void SyncVToMte3()
    {
        event_t id = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(id); WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void SyncMte3ToV()
    {
        event_t id = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(id); WaitFlag<HardEvent::MTE3_V>(id);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatPairGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0}, blockIdx_{0}, blockNum_{1};
    uint32_t inputStart_{0}, inputSplit_{0}, inputEnd_{0}, outputStart_{0};
    uint32_t group0Elements_{0}, group1Elements_{0}, pairElements_{0}, sourceElements_{0};
    uint32_t batchRows_{0}, sourceBufferBytes_{0}, outputBufferBytes_{0}, offsetBufferBytes_{0};
    uint32_t group0Aligned_{0}, group1Aligned_{0}, pairAligned_{0}, maxGroupAligned_{0};
    uint64_t rowStart_{0}, rowCount_{0}, outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_, outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
};

// key36：每个 AIV 只处理一个连续输入带的一段连续行。它保留 key7 已验证的连续 source
// 搬入、Gather 和二维 MTE3 写回，只改变任务所有权，避免每个核反复遍历全部窄输入。
class KernelConcatBandGather {
public:
    __aicore__ inline explicit KernelConcatBandGather(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
        const __gm__ ConcatBandGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if (tilingData_ == nullptr || blockIdx_ >= tilingData_->taskCount) {
            return;
        }
        inputCount_ = tilingData_->inputCount;
        inputStart_ = tilingData_->taskInputStart[blockIdx_];
        inputEnd_ = tilingData_->taskInputEnd[blockIdx_];
        outputStart_ = tilingData_->taskOutputStart[blockIdx_];
        groupElements_ = tilingData_->taskGroupElements[blockIdx_];
        sourceElements_ = tilingData_->taskSourceElements[blockIdx_];
        batchRows_ = tilingData_->taskBatchRows[blockIdx_];
        sourceBufferBytes_ = tilingData_->taskSourceBufferBytes[blockIdx_];
        outputBufferBytes_ = tilingData_->taskOutputBufferBytes[blockIdx_];
        offsetBufferBytes_ = tilingData_->taskOffsetBufferBytes[blockIdx_];
        rowStart_ = tilingData_->taskRowStart[blockIdx_];
        rowCount_ = tilingData_->taskRowCount[blockIdx_];
        outputRowElements_ = tilingData_->outputRowElements;
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        const uint32_t alignedElements = AlignUpGatherElements(groupElements_);
        const uint64_t matrixElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        valid_ = tilingData_->taskCount != 0 && tilingData_->taskCount == blockNum_ &&
            inputCount_ != 0 && inputStart_ < inputEnd_ && inputEnd_ <= inputCount_ &&
            tilingData_->dtypeSize == sizeof(int32_t) && groupElements_ != 0 &&
            sourceElements_ != 0 && batchRows_ != 0 && rowCount_ != 0 &&
            rowStart_ <= tilingData_->outer && rowCount_ <= tilingData_->outer - rowStart_ &&
            outputStart_ <= outputRowElements_ &&
            groupElements_ <= outputRowElements_ - outputStart_ && alignedElements != 0 &&
            sourceBufferBytes_ != 0 && outputBufferBytes_ != 0 && offsetBufferBytes_ != 0 &&
            sourceBufferBytes_ % UB_BLOCK_BYTES == 0 &&
            outputBufferBytes_ % UB_BLOCK_BYTES == 0 &&
            offsetBufferBytes_ % UB_BLOCK_BYTES == 0 &&
            static_cast<uint64_t>(sourceElements_) * sizeof(int32_t) <= sourceBufferBytes_ &&
            matrixElements <= MAX_UINT32 &&
            matrixElements * sizeof(int32_t) <= outputBufferBytes_ &&
            matrixElements * sizeof(int32_t) <= offsetBufferBytes_;
        if (!valid_) {
            return;
        }
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
        pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        const uint32_t alignedElements = AlignUpGatherElements(groupElements_);
        GenerateOffsets(alignedElements);
        if (!valid_) {
            return;
        }
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(sourceElements_);
        output.SetSize(batchRows_ * alignedElements);
        offsets.SetSize(batchRows_ * alignedElements);

        bool outputBusy = false;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(batchStart, currentBatch, source);
            if (!valid_) {
                return;
            }
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);
            SyncVToMte2();
            SyncVToMte3();

            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart_;
            const uint64_t blockLenBytes = static_cast<uint64_t>(groupElements_) *
                sizeof(int32_t);
            const uint64_t dstStrideBytes = (outputRowElements_ - groupElements_) *
                sizeof(int32_t);
            if (currentBatch > MAX_UINT16_VALUE || blockLenBytes > MAX_UINT32 ||
                dstStrideBytes > MAX_UINT32) {
                valid_ = false;
                return;
            }
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch), static_cast<uint32_t>(blockLenBytes), 0,
                static_cast<uint32_t>(dstStrideBytes), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }
    }

private:
    __aicore__ inline void FillConstantVector(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(count,
                GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddScalarVector(const LocalTensor<int32_t>& tensor,
        uint32_t start, uint32_t count, int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(count,
                GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                mask, 1, {1, 1, 8, 8});
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void GenerateOffsets(uint32_t alignedElements)
    {
        if (alignedElements == 0 ||
            static_cast<uint64_t>(batchRows_) * alignedElements > MAX_UINT32) {
            valid_ = false;
            return;
        }
        LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
        offsets.SetSize(batchRows_ * alignedElements);
        CreateVecIndex(offsets, static_cast<int32_t>(0), alignedElements);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignedElements));

        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart_; input < inputEnd_; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t adjustment = (static_cast<int64_t>(sourceBase) -
                    static_cast<int64_t>(outputPrefix)) * static_cast<int64_t>(sizeof(int32_t));
                if (adjustment < MIN_INT32_VALUE || adjustment > MAX_INT32_VALUE ||
                    outputPrefix > groupElements_ || inputElements > groupElements_ - outputPrefix) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(offsets, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(adjustment));
            }
            outputPrefix += inputElements;
            const uint64_t rawElements = static_cast<uint64_t>(batchRows_) * inputElements;
            if (rawElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(rawElements));
            if (sourceBase > MAX_UINT32) {
                valid_ = false;
                return;
            }
        }
        if (outputPrefix != groupElements_ || sourceBase != sourceElements_) {
            valid_ = false;
            return;
        }
        FillConstantVector(offsets, groupElements_, alignedElements - groupElements_, 0);
        if (batchRows_ == 1) {
            return;
        }

        LocalTensor<int32_t> strides = offsets[(batchRows_ - 1) * alignedElements];
        outputPrefix = 0;
        for (uint32_t input = inputStart_; input < inputEnd_; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                if (outputPrefix > groupElements_ || inputElements > groupElements_ - outputPrefix ||
                    static_cast<uint64_t>(inputElements) * sizeof(int32_t) > MAX_INT32_VALUE) {
                    valid_ = false;
                    return;
                }
                FillConstantVector(strides, static_cast<uint32_t>(outputPrefix), inputElements,
                    static_cast<int32_t>(inputElements * sizeof(int32_t)));
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements_) {
            valid_ = false;
            return;
        }
        FillConstantVector(strides, groupElements_, alignedElements - groupElements_, 0);
        for (uint32_t row = 1; row + 1 < batchRows_; ++row) {
            Add(offsets[row * alignedElements], offsets[(row - 1) * alignedElements], strides,
                static_cast<int32_t>(alignedElements));
        }
        Add(strides, offsets[(batchRows_ - 2) * alignedElements], strides,
            static_cast<int32_t>(alignedElements));
    }

    __aicore__ inline void CopyInputs(uint64_t rowStart, uint32_t rows,
        const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart_; input < inputEnd_; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t rawSlotElements = static_cast<uint64_t>(batchRows_) * inputElements;
            const uint64_t slotElements = rawSlotElements > MAX_UINT32 ? MAX_UINT32 + 1 :
                AlignUpGatherElements(static_cast<uint32_t>(rawSlotElements));
            const uint64_t payloadBytes = static_cast<uint64_t>(rows) * inputElements *
                sizeof(int32_t);
            if (slotElements > MAX_UINT32 || sourceBase > sourceElements_ ||
                slotElements > sourceElements_ - sourceBase || payloadBytes > MAX_UINT32) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(payloadBytes), 0, 0, 0};
                DataCopyPad(source[sourceBase], inputGm_[rowStart * inputElements],
                    copyInParams, padParams);
            }
            sourceBase += static_cast<uint32_t>(slotElements);
        }
        if (sourceBase != sourceElements_) {
            valid_ = false;
        }
    }

    __aicore__ inline void GatherSegmented(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& source, const LocalTensor<uint32_t>& offsets, uint32_t count)
    {
        for (uint32_t done = 0; done < count; done += GATHER_MAX_ELEMENTS_PER_CALL) {
            const uint32_t current = static_cast<uint32_t>(MinU64(
                GATHER_MAX_ELEMENTS_PER_CALL, count - done));
            Gather(output[done], source, offsets[done], 0, current);
        }
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatBandGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t inputStart_{0};
    uint32_t inputEnd_{0};
    uint32_t outputStart_{0};
    uint32_t groupElements_{0};
    uint32_t sourceElements_{0};
    uint32_t batchRows_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint32_t offsetBufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
};

class KernelConcatRowGatherBlock {
public:
    __aicore__ inline explicit KernelConcatRowGatherBlock(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatRowGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        inputCount_ = tilingData_->inputCount;
        outputRowElements_ = tilingData_->outputRowElements;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        const uint32_t highCoreCount = tilingData_->highCoreCount;
        if (blockIdx_ < highCoreCount) {
            rowStart_ = static_cast<uint64_t>(blockIdx_) * tilingData_->rowsPerCore1;
            rowCount_ = tilingData_->rowsPerCore1;
            planIndex_ = 1;
        } else {
            rowStart_ = static_cast<uint64_t>(highCoreCount) * tilingData_->rowsPerCore1 +
                static_cast<uint64_t>(blockIdx_ - highCoreCount) * tilingData_->rowsPerCore0;
            rowCount_ = tilingData_->rowsPerCore0;
            planIndex_ = 0;
        }
        if (planIndex_ == 0) {
            batchRows_ = tilingData_->batchRows0;
            groupCount_ = tilingData_->groupCount0;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes0;
            outputBufferBytes_ = tilingData_->outputBufferBytes0;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes0;
        } else {
            batchRows_ = tilingData_->batchRows1;
            groupCount_ = tilingData_->groupCount1;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes1;
            outputBufferBytes_ = tilingData_->outputBufferBytes1;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes1;
        }
        valid_ = highCoreCount <= blockNum_ && rowStart_ <= tilingData_->outer &&
                 rowCount_ <= tilingData_->outer - rowStart_ &&
                 tilingData_->dtypeSize == sizeof(int32_t) && rowCount_ != 0 &&
                 batchRows_ != 0 && batchRows_ <= GATHERB_MAX_REPEATS && groupCount_ != 0 &&
                 sourceBufferBytes_ != 0 && outputBufferBytes_ != 0 && offsetBufferBytes_ != 0 &&
                 sourceBufferBytes_ % sizeof(int32_t) == 0 &&
                 outputBufferBytes_ % sizeof(int32_t) == 0 &&
                 offsetBufferBytes_ % sizeof(int32_t) == 0;
        if (!valid_) {
            return;
        }
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
        pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        bool outputBusy = false;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                groupSourceElements == 0 || outputStart + groupElements > outputRowElements_) {
                return;
            }
            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                         groupSourceElements, outputBusy);
            inputStart = inputEnd;
            outputStart += groupElements;
        }
    }

private:
    __aicore__ inline void FillConstantVector(const LocalTensor<int32_t>& tensor,
                                              uint32_t start, uint32_t count,
                                              int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddScalarVector(const LocalTensor<int32_t>& tensor,
                                           uint32_t start, uint32_t count,
                                           int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                          mask, 1, {1, 1, 8, 8});
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline bool GetSourceDataElements(uint32_t inputStart, uint32_t inputEnd,
                                                  uint32_t groupElements,
                                                  uint32_t& sourceDataElements)
    {
        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint32_t phase = static_cast<uint32_t>(outputPrefix &
                (GATHER_BLOCK_ELEMENTS - 1));
            const uint32_t stripeElements = AlignUpGatherElements(phase + inputElements);
            // 零宽输入不占 source stripe，必须与 Host 的 UB 规划保持一致。
            if (inputElements != 0) {
                sourceBase += static_cast<uint64_t>(batchRows_) * stripeElements;
            }
            outputPrefix += inputElements;
        }
        if (outputPrefix != groupElements || sourceBase > MAX_UINT32) {
            return false;
        }
        sourceDataElements = static_cast<uint32_t>(sourceBase);
        return true;
    }

    __aicore__ inline void GenerateBlockOffsets(uint32_t inputStart, uint32_t inputEnd,
                                                uint32_t groupElements,
                                                uint32_t sourceDataElements,
                                                uint32_t activeRows, uint32_t rowBlocks,
                                                uint32_t rowBlockStride,
                                                LocalTensor<int32_t>& offsets)
    {
        if (activeRows == 0 || activeRows > batchRows_ || rowBlocks == 0 ||
            rowBlockStride < rowBlocks) {
            valid_ = false;
            return;
        }
        const uint64_t offsetElements = static_cast<uint64_t>(activeRows) * rowBlockStride;
        if (offsetElements > offsetBufferBytes_ / sizeof(int32_t)) {
            valid_ = false;
            return;
        }
        offsets.SetSize(static_cast<uint32_t>(offsetElements));
        CreateVecIndex(offsets, static_cast<int32_t>(0), rowBlockStride);
        Muls(offsets, offsets, static_cast<int32_t>(UB_BLOCK_BYTES), rowBlockStride);
        const int64_t stashBaseBytes = static_cast<int64_t>(sourceDataElements) * sizeof(int32_t);
        if (stashBaseBytes > MAX_INT32_VALUE) {
            valid_ = false;
            return;
        }
        AddScalarVector(offsets, 0, rowBlockStride, static_cast<int32_t>(stashBaseBytes));

        LocalTensor<int32_t> rowStrides = offsets[(activeRows - 1) * rowBlockStride];
        const int64_t defaultStrideBytes = static_cast<int64_t>(rowBlockStride) * UB_BLOCK_BYTES;
        if (activeRows > 1) {
            FillConstantVector(rowStrides, 0, rowBlockStride,
                static_cast<int32_t>(defaultStrideBytes));
        }

        uint64_t outputPrefix = 0;
        uint64_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint32_t phase = static_cast<uint32_t>(outputPrefix &
                (GATHER_BLOCK_ELEMENTS - 1));
            const uint32_t stripeElements = AlignUpGatherElements(phase + inputElements);
            const uint64_t firstInterior = (outputPrefix + GATHER_BLOCK_ELEMENTS - 1) /
                GATHER_BLOCK_ELEMENTS;
            const uint64_t lastInterior = (outputPrefix + inputElements) /
                GATHER_BLOCK_ELEMENTS;
            if (lastInterior > firstInterior) {
                const uint32_t start = static_cast<uint32_t>(firstInterior);
                const uint32_t count = static_cast<uint32_t>(lastInterior - firstInterior);
                const int64_t sourceAdjustment = static_cast<int64_t>(sourceBase) * sizeof(int32_t) -
                    static_cast<int64_t>(outputPrefix / GATHER_BLOCK_ELEMENTS) * UB_BLOCK_BYTES -
                    stashBaseBytes;
                const int64_t strideAdjustment = static_cast<int64_t>(stripeElements) * sizeof(int32_t) -
                    defaultStrideBytes;
                if (sourceAdjustment < MIN_INT32_VALUE || sourceAdjustment > MAX_INT32_VALUE ||
                    strideAdjustment < MIN_INT32_VALUE || strideAdjustment > MAX_INT32_VALUE ||
                    start > rowBlocks || count > rowBlocks - start) {
                    valid_ = false;
                    return;
                }
                AddScalarVector(offsets, start, count, static_cast<int32_t>(sourceAdjustment));
                if (activeRows > 1) {
                    AddScalarVector(rowStrides, start, count,
                        static_cast<int32_t>(strideAdjustment));
                }
            }
            outputPrefix += inputElements;
            if (inputElements != 0) {
                sourceBase += static_cast<uint64_t>(batchRows_) * stripeElements;
            }
        }
        if (outputPrefix != groupElements || sourceBase != sourceDataElements) {
            valid_ = false;
            return;
        }
        for (uint32_t row = 1; row + 1 < activeRows; ++row) {
            Add(offsets[row * rowBlockStride], offsets[(row - 1) * rowBlockStride], rowStrides,
                static_cast<int32_t>(rowBlockStride));
        }
        if (activeRows > 1) {
            Add(rowStrides, offsets[(activeRows - 2) * rowBlockStride], rowStrides,
                static_cast<int32_t>(rowBlockStride));
        }
    }

    __aicore__ inline void CopyInputs(uint32_t inputStart, uint32_t inputEnd,
                                      uint64_t batchStart, uint32_t rows,
                                      const LocalTensor<int32_t>& source)
    {
        uint64_t outputPrefix = 0;
        uint32_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint32_t phase = static_cast<uint32_t>(outputPrefix &
                (GATHER_BLOCK_ELEMENTS - 1));
            const uint32_t stripeElements = AlignUpGatherElements(phase + inputElements);
            if (inputElements != 0) {
                const uint32_t rightPadding = stripeElements - phase - inputElements;
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    static_cast<uint16_t>(rows),
                    static_cast<uint32_t>(inputElements * sizeof(int32_t)), 0, 0, 0};
                const DataCopyPadExtParams<int32_t> padParams{
                    true, static_cast<uint8_t>(phase), static_cast<uint8_t>(rightPadding), 0};
                DataCopyPad(source[sourceBase], inputGm_[batchStart * inputElements],
                            copyInParams, padParams);
            }
            outputPrefix += inputElements;
            if (inputElements != 0) {
                sourceBase += batchRows_ * stripeElements;
            }
        }
    }

    __aicore__ inline void CopyBoundaryPiece(const LocalTensor<int32_t>& source,
                                             uint32_t stashBase, uint32_t rowBlockStride,
                                             uint32_t sourceBlockBase, uint32_t sourceStripeBlocks,
                                             uint32_t outputBlock, uint32_t lowLane,
                                             uint32_t highLane, uint32_t activeRows)
    {
        if (lowLane >= highLane || highLane > GATHER_BLOCK_ELEMENTS || activeRows == 0 ||
            activeRows > GATHERB_MAX_REPEATS) {
            valid_ = false;
            return;
        }
        const uint64_t maskValue = ((1ULL << (highLane - lowLane)) - 1ULL) << lowLane;
        uint64_t mask[2] = {maskValue, 0};
        const CopyRepeatParams copyParams{1, 1, static_cast<uint16_t>(rowBlockStride),
            static_cast<uint16_t>(sourceStripeBlocks)};
        Copy<int32_t>(source[stashBase + outputBlock * GATHER_BLOCK_ELEMENTS],
                      source[sourceBlockBase], mask, static_cast<uint8_t>(activeRows), copyParams);
    }

    __aicore__ inline void BuildBoundaryStash(uint32_t inputStart, uint32_t inputEnd,
                                              uint32_t groupElements, uint32_t sourceDataElements,
                                              uint32_t rowBlockStride, uint32_t activeRows,
                                              const LocalTensor<int32_t>& source)
    {
        uint64_t outputPrefix = 0;
        uint32_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint32_t phase = static_cast<uint32_t>(outputPrefix &
                (GATHER_BLOCK_ELEMENTS - 1));
            const uint32_t stripeElements = AlignUpGatherElements(phase + inputElements);
            if (inputElements != 0) {
                const uint64_t inputEndOffset = outputPrefix + inputElements;
                const uint32_t firstBlock = static_cast<uint32_t>(outputPrefix /
                    GATHER_BLOCK_ELEMENTS);
                const uint32_t lastBlock = static_cast<uint32_t>((inputEndOffset - 1) /
                    GATHER_BLOCK_ELEMENTS);
                const uint32_t firstInterior = static_cast<uint32_t>((outputPrefix +
                    GATHER_BLOCK_ELEMENTS - 1) / GATHER_BLOCK_ELEMENTS);
                const uint32_t lastInterior = static_cast<uint32_t>(inputEndOffset /
                    GATHER_BLOCK_ELEMENTS);
                const bool hasLeadingBoundary = firstBlock < firstInterior;
                const bool hasTrailingBoundary = lastBlock >= lastInterior;
                if (hasLeadingBoundary) {
                    const uint32_t lowLane = static_cast<uint32_t>(outputPrefix &
                        (GATHER_BLOCK_ELEMENTS - 1));
                    const uint32_t highLane = static_cast<uint32_t>(MinU64(inputEndOffset,
                        static_cast<uint64_t>(firstBlock + 1) * GATHER_BLOCK_ELEMENTS) -
                        static_cast<uint64_t>(firstBlock) * GATHER_BLOCK_ELEMENTS);
                    CopyBoundaryPiece(source, sourceDataElements, rowBlockStride, sourceBase,
                                      stripeElements / GATHER_BLOCK_ELEMENTS, firstBlock,
                                      lowLane, highLane, activeRows);
                }
                if (hasTrailingBoundary && (!hasLeadingBoundary || lastBlock != firstBlock)) {
                    const uint32_t lowLane = static_cast<uint32_t>(MaxU64(outputPrefix,
                        static_cast<uint64_t>(lastBlock) * GATHER_BLOCK_ELEMENTS) -
                        static_cast<uint64_t>(lastBlock) * GATHER_BLOCK_ELEMENTS);
                    const uint32_t highLane = static_cast<uint32_t>(inputEndOffset -
                        static_cast<uint64_t>(lastBlock) * GATHER_BLOCK_ELEMENTS);
                    const uint32_t sourceBlockBase = sourceBase +
                        (lastBlock - firstBlock) * GATHER_BLOCK_ELEMENTS;
                    CopyBoundaryPiece(source, sourceDataElements, rowBlockStride, sourceBlockBase,
                                      stripeElements / GATHER_BLOCK_ELEMENTS, lastBlock,
                                      lowLane, highLane, activeRows);
                }
            }
            outputPrefix += inputElements;
            if (inputElements != 0) {
                sourceBase += batchRows_ * stripeElements;
            }
        }
        if (outputPrefix != groupElements || sourceBase != sourceDataElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void GatherBlocks(const LocalTensor<uint32_t>& output,
                                        const LocalTensor<uint32_t>& source,
                                        const LocalTensor<uint32_t>& offsets,
                                        uint32_t totalBlocks)
    {
        const GatherRepeatParams gatherParams{1, GATHERB_BLOCKS_PER_REPEAT};
        for (uint32_t done = 0; done < totalBlocks; done += GATHERB_MAX_BLOCKS_PER_CALL) {
            const uint32_t currentBlocks = static_cast<uint32_t>(MinU64(
                GATHERB_MAX_BLOCKS_PER_CALL, totalBlocks - done));
            if (currentBlocks == 0 || currentBlocks % GATHERB_BLOCKS_PER_REPEAT != 0) {
                valid_ = false;
                return;
            }
            Gatherb<uint32_t>(output[done * GATHER_BLOCK_ELEMENTS], source, offsets[done],
                static_cast<uint8_t>(currentBlocks / GATHERB_BLOCKS_PER_REPEAT), gatherParams);
        }
    }

    __aicore__ inline void ProcessGroup(uint32_t inputStart, uint32_t inputEnd,
                                        uint32_t outputStart, uint32_t groupElements,
                                        uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t rowBlocks = (groupElements + GATHER_BLOCK_ELEMENTS - 1) /
            GATHER_BLOCK_ELEMENTS;
        const uint32_t rowBlockStride = AlignUpGatherBlocks(rowBlocks);
        uint32_t sourceDataElements = 0;
        if (rowBlocks == 0 || rowBlockStride < rowBlocks ||
            !GetSourceDataElements(inputStart, inputEnd, groupElements, sourceDataElements)) {
            valid_ = false;
            return;
        }
        const uint64_t sourceRequired = static_cast<uint64_t>(sourceDataElements) +
            static_cast<uint64_t>(batchRows_) * rowBlockStride * GATHER_BLOCK_ELEMENTS;
        const uint64_t outputRequired = static_cast<uint64_t>(batchRows_) * rowBlockStride *
            GATHER_BLOCK_ELEMENTS;
        const uint64_t offsetRequired = static_cast<uint64_t>(batchRows_) * rowBlockStride;
        if (sourceRequired > groupSourceElements || sourceRequired > sourceBufferBytes_ / sizeof(int32_t) ||
            outputRequired > outputBufferBytes_ / sizeof(int32_t) ||
            offsetRequired > offsetBufferBytes_ / sizeof(int32_t) || rowBlockStride >
            MAX_UINT16_VALUE) {
            valid_ = false;
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> outputI32 = outputBuffer_.Get<int32_t>();
        LocalTensor<int32_t> offsetsI32 = offsetBuffer_.Get<int32_t>();
        source.SetSize(sourceBufferBytes_ / sizeof(int32_t));
        outputI32.SetSize(outputBufferBytes_ / sizeof(int32_t));
        offsetsI32.SetSize(offsetBufferBytes_ / sizeof(int32_t));
        LocalTensor<uint32_t> sourceU32 = source.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> outputU32 = outputI32.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> offsetsU32 = offsetsI32.ReinterpretCast<uint32_t>();

        GenerateBlockOffsets(inputStart, inputEnd, groupElements, sourceDataElements, batchRows_,
            rowBlocks, rowBlockStride, offsetsI32);
        uint32_t generatedRows = batchRows_;
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            if (currentBatch != generatedRows) {
                GenerateBlockOffsets(inputStart, inputEnd, groupElements, sourceDataElements,
                    currentBatch, rowBlocks, rowBlockStride, offsetsI32);
                generatedRows = currentBatch;
            }
            if (!valid_) {
                return;
            }
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch, source);
            SyncMte2ToV();
            BuildBoundaryStash(inputStart, inputEnd, groupElements, sourceDataElements,
                rowBlockStride, currentBatch, source);
            if (!valid_) {
                return;
            }
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherBlocks(outputU32, sourceU32, offsetsU32, currentBatch * rowBlockStride);
            if (!valid_) {
                return;
            }
            SyncVToMte2();
            SyncVToMte3();
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)),
                rowBlockStride - rowBlocks,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], outputI32, copyOutParams);
            outputBusy = true;
        }
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatRowGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t planIndex_{0};
    uint32_t batchRows_{0};
    uint32_t groupCount_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint32_t offsetBufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
};

class KernelConcatRowGatherResident {
public:
    __aicore__ inline explicit KernelConcatRowGatherResident(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatRowGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        inputCount_ = tilingData_->inputCount;
        outputRowElements_ = tilingData_->outputRowElements;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        const uint32_t highCoreCount = tilingData_->highCoreCount;
        if (blockIdx_ < highCoreCount) {
            rowStart_ = static_cast<uint64_t>(blockIdx_) * tilingData_->rowsPerCore1;
            rowCount_ = tilingData_->rowsPerCore1;
            planIndex_ = 1;
        } else {
            rowStart_ = static_cast<uint64_t>(highCoreCount) * tilingData_->rowsPerCore1 +
                static_cast<uint64_t>(blockIdx_ - highCoreCount) * tilingData_->rowsPerCore0;
            rowCount_ = tilingData_->rowsPerCore0;
            planIndex_ = 0;
        }
        if (planIndex_ == 0) {
            batchRows_ = tilingData_->batchRows0;
            groupCount_ = tilingData_->groupCount0;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes0;
            outputBufferBytes_ = tilingData_->outputBufferBytes0;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes0;
        } else {
            batchRows_ = tilingData_->batchRows1;
            groupCount_ = tilingData_->groupCount1;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes1;
            outputBufferBytes_ = tilingData_->outputBufferBytes1;
            offsetBufferBytes_ = tilingData_->offsetBufferBytes1;
        }

        valid_ = highCoreCount <= blockNum_ && rowStart_ <= tilingData_->outer &&
                 rowCount_ <= tilingData_->outer - rowStart_ &&
                 tilingData_->dtypeSize == sizeof(int32_t) && rowCount_ != 0 &&
                 rowCount_ <= MAX_BLOCK_COUNT && batchRows_ != 0 &&
                 batchRows_ <= rowCount_ && groupCount_ != 0 &&
                 sourceBufferBytes_ != 0 && outputBufferBytes_ != 0 &&
                 offsetBufferBytes_ != 0;
        if (!valid_) {
            return;
        }
        // Host 已按三块 TBuf 的最大组合做 UB 预算；这里每个 group 只使用相应有效前缀。
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
        pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }

        uint32_t inputStart = 0;
        uint64_t outputStart = 0;
        bool outputBusy = false;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                groupSourceElements == 0 || outputStart + groupElements > outputRowElements_) {
                return;
            }

            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                         groupSourceElements, outputBusy);
            inputStart = inputEnd;
            outputStart += groupElements;
        }
    }

private:
    __aicore__ inline void FillConstantVector(const LocalTensor<int32_t>& tensor,
                                              uint32_t start, uint32_t count,
                                              int32_t value)
    {
        if (count == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Duplicate<int32_t>(tensor[start - lane], value, mask, 1, 1, 8);
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Duplicate<int32_t>(tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void AddScalarVector(const LocalTensor<int32_t>& tensor,
                                           uint32_t start, uint32_t count,
                                           int32_t value)
    {
        if (count == 0 || value == 0) {
            return;
        }
        const uint32_t lane = start & (GATHER_BLOCK_ELEMENTS - 1);
        if (lane != 0) {
            const uint32_t headCount = static_cast<uint32_t>(MinU64(
                count, GATHER_BLOCK_ELEMENTS - lane));
            uint64_t mask[1] = {((1ULL << headCount) - 1ULL) << lane};
            Adds<int32_t>(tensor[start - lane], tensor[start - lane], value,
                          mask, 1, {1, 1, 8, 8});
            start += headCount;
            count -= headCount;
        }
        if (count != 0) {
            Adds<int32_t>(tensor[start], tensor[start], value, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void GenerateOffsets(uint32_t inputStart, uint32_t inputEnd,
                                           uint32_t groupElements, uint64_t batchOffset)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
        LocalTensor<int32_t> strides = offsets[(batchRows_ - 1) * alignedElements];

        // 首行从连续元素偏移开始，随后为每个输入段加上它在常驻 source 中的行起点。
        CreateVecIndex(offsets, static_cast<int32_t>(0), alignedElements);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)),
             static_cast<int32_t>(alignedElements));

        uint32_t outputPrefix = 0;
        uint32_t sourceBase = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const int64_t sourceOffset = static_cast<int64_t>(sourceBase) +
                    static_cast<int64_t>(batchOffset) * inputElements;
                const int64_t adjustment = (sourceOffset - outputPrefix) *
                    static_cast<int64_t>(sizeof(int32_t));
                AddScalarVector(offsets, outputPrefix, inputElements,
                                static_cast<int32_t>(adjustment));
                if (batchRows_ > 1) {
                    FillConstantVector(strides, outputPrefix, inputElements,
                                       static_cast<int32_t>(inputElements * sizeof(int32_t)));
                }
            }
            outputPrefix += inputElements;
            const uint64_t inputPayload = rowCount_ * static_cast<uint64_t>(inputElements);
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(inputPayload));
        }

        // 行尾对齐位置不会写回 GM，固定读取 source[0] 以保证 Gather 地址合法。
        FillConstantVector(offsets, groupElements, alignedElements - groupElements, 0);
        if (batchRows_ > 1) {
            FillConstantVector(strides, groupElements, alignedElements - groupElements, 0);
        }
        for (uint32_t row = 1; row + 1 < batchRows_; ++row) {
            Add(offsets[row * alignedElements], offsets[(row - 1) * alignedElements],
                strides, static_cast<int32_t>(alignedElements));
        }
        if (batchRows_ > 1) {
            Add(strides, offsets[(batchRows_ - 2) * alignedElements], strides,
                static_cast<int32_t>(alignedElements));
        }
    }

    __aicore__ inline void ProcessGroup(uint32_t inputStart, uint32_t inputEnd,
                                        uint64_t outputStart, uint32_t groupElements,
                                        uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t outputElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        const uint64_t offsetElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        if (outputElements * sizeof(int32_t) > outputBufferBytes_ ||
            offsetElements * sizeof(int32_t) > offsetBufferBytes_ ||
            static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) > sourceBufferBytes_) {
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<int32_t>().ReinterpretCast<uint32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(static_cast<uint32_t>(outputElements));
        offsets.SetSize(static_cast<uint32_t>(offsetElements));

        // 每组 source 仅搬入一次；每个批次复用小 output UB，避免长跨行写回。
        CopyInputs(inputStart, inputEnd, source);
        SyncMte2ToV();
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            GenerateOffsets(inputStart, inputEnd, groupElements, batchOffset);
            // offset 重建不触碰 output，可与上批 MTE3 写回重叠；Gather 前才等待复用。
            if (outputBusy) {
                SyncMte3ToV();
            }
            GatherSegmented(output, source, offsets, currentBatch * alignedElements);

            SyncVToMte3();
            const uint32_t sourceStrideBlocks =
                (alignedElements - groupElements) / GATHER_BLOCK_ELEMENTS;
            const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)),
                sourceStrideBlocks,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }

        // source 将被下一组 MTE2 覆盖；最后一批 output 继续由 MTE3 异步写回。
        SyncVToMte2();
    }

    __aicore__ inline void GatherSegmented(const LocalTensor<int32_t>& output,
                                           const LocalTensor<int32_t>& source,
                                           const LocalTensor<uint32_t>& offsets,
                                           uint32_t count)
    {
        // count 重载在 A2 内部使用 uint8_t repeat，故每段限定在 255 * 64 个 FP32 元素内。
        for (uint32_t done = 0; done < count; done += GATHER_MAX_ELEMENTS_PER_CALL) {
            const uint32_t current = static_cast<uint32_t>(MinU64(
                GATHER_MAX_ELEMENTS_PER_CALL, count - done));
            Gather(output[done], source, offsets[done], 0, current);
        }
    }

    __aicore__ inline void CopyInputs(uint32_t inputStart, uint32_t inputEnd,
                                      const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t inputPayload = rowCount_ * static_cast<uint64_t>(inputElements);
            if (inputElements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(inputPayload * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(source[sourceBase], inputGm_[rowStart_ * inputElements],
                            copyInParams, padParams);
            }
            sourceBase += AlignUpGatherElements(static_cast<uint32_t>(inputPayload));
        }
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatRowGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t planIndex_{0};
    uint32_t batchRows_{0};
    uint32_t groupCount_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint32_t offsetBufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
};

template <bool MaskedRows = false>
class KernelConcatRowDirect {
public:
    __aicore__ inline explicit KernelConcatRowDirect(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatRowGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        inputCount_ = tilingData_->inputCount;
        outputRowElements_ = tilingData_->outputRowElements;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        const uint32_t highCoreCount = tilingData_->highCoreCount;
        if (blockIdx_ < highCoreCount) {
            rowStart_ = static_cast<uint64_t>(blockIdx_) * tilingData_->rowsPerCore1;
            rowCount_ = tilingData_->rowsPerCore1;
            planIndex_ = 1;
        } else {
            rowStart_ = static_cast<uint64_t>(highCoreCount) * tilingData_->rowsPerCore1 +
                static_cast<uint64_t>(blockIdx_ - highCoreCount) * tilingData_->rowsPerCore0;
            rowCount_ = tilingData_->rowsPerCore0;
            planIndex_ = 0;
        }
        if (planIndex_ == 0) {
            batchRows_ = tilingData_->batchRows0;
            groupCount_ = tilingData_->groupCount0;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes0;
            outputBufferBytes_ = tilingData_->outputBufferBytes0;
        } else {
            batchRows_ = tilingData_->batchRows1;
            groupCount_ = tilingData_->groupCount1;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes1;
            outputBufferBytes_ = tilingData_->outputBufferBytes1;
        }

        valid_ = highCoreCount <= blockNum_ && rowStart_ <= tilingData_->outer &&
                 rowCount_ <= tilingData_->outer - rowStart_ &&
                 tilingData_->dtypeSize == sizeof(int32_t) && rowCount_ != 0 &&
                 batchRows_ != 0 && groupCount_ != 0 && sourceBufferBytes_ != 0 &&
                 outputBufferBytes_ != 0;
        if (!valid_) {
            return;
        }
        // 复用 RowGather 的输入分段常驻布局：MTE2 只搬一次，后续用连续向量 Copy 代替全量 Gather。
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
        pipe_.InitBuffer(outputBuffer_, outputBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }

        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        bool outputBusy = false;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                groupSourceElements == 0 || outputStart + groupElements > outputRowElements_) {
                return;
            }
            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                         groupSourceElements, outputBusy);
            inputStart = inputEnd;
            outputStart += groupElements;
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t inputStart, uint32_t inputEnd,
                                        uint32_t outputStart, uint32_t groupElements,
                                        uint32_t groupSourceElements, bool& outputBusy)
    {
        const uint32_t alignedElements = AlignUpGatherElements(groupElements);
        const uint64_t outputElements = static_cast<uint64_t>(batchRows_) * alignedElements;
        if (outputElements * sizeof(int32_t) > outputBufferBytes_ ||
            static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) > sourceBufferBytes_) {
            return;
        }

        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> output = outputBuffer_.Get<int32_t>();
        source.SetSize(groupSourceElements);
        output.SetSize(static_cast<uint32_t>(outputElements));
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            CopyInputs(inputStart, inputEnd, rowStart_ + batchOffset, currentBatch, source);
            SyncMte2ToV();
            if (outputBusy) {
                SyncMte3ToV();
            }
            CopySegments(inputStart, inputEnd, currentBatch, alignedElements,
                         groupElements, source, output);
            // source 在下一批会被 MTE2 覆盖；output 则交给 MTE3 异步写回。
            SyncVToMte2();
            SyncVToMte3();
            const uint64_t outputOffset =
                (rowStart_ + batchOffset) * outputRowElements_ + outputStart;
            const DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(currentBatch),
                static_cast<uint32_t>(groupElements * sizeof(int32_t)),
                (alignedElements - groupElements) / GATHER_BLOCK_ELEMENTS,
                static_cast<uint32_t>((outputRowElements_ - groupElements) * sizeof(int32_t)), 0};
            DataCopyPad(outputGm_[outputOffset], output, copyOutParams);
            outputBusy = true;
        }
    }

    __aicore__ inline void CopyInputs(uint32_t inputStart, uint32_t inputEnd,
                                      uint64_t rowStart, uint32_t rows,
                                      const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        uint32_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                const uint32_t phase = outputPrefix & (GATHER_BLOCK_ELEMENTS - 1);
                const uint32_t sourceStrideElements = AlignUpGatherElements(phase + inputElements);
                if constexpr (MaskedRows) {
                    if (sourceStrideElements == 0 || sourceBase > source.GetSize() ||
                        sourceStrideElements > source.GetSize() - sourceBase) {
                        valid_ = false;
                        return;
                    }
                }
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                if constexpr (MaskedRows) {
                    const uint32_t rightPadding = sourceStrideElements - phase - inputElements;
                    const DataCopyExtParams copyInParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(inputElements * sizeof(int32_t)),
                        0, 0, 0};
                    const DataCopyPadExtParams<int32_t> padParams{true, static_cast<uint8_t>(phase),
                        static_cast<uint8_t>(rightPadding), 0};
                    DataCopyPad(source[sourceBase], inputGm_[rowStart * inputElements],
                                copyInParams, padParams);
                } else {
                    const DataCopyExtParams copyInParams{
                        1, static_cast<uint32_t>(rows * inputElements * sizeof(int32_t)), 0, 0, 0};
                    const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
                    DataCopyPad(source[sourceBase], inputGm_[rowStart * inputElements],
                                copyInParams, padParams);
                }
                if constexpr (MaskedRows) {
                    sourceBase += batchRows_ * sourceStrideElements;
                } else {
                    sourceBase += AlignUpGatherElements(batchRows_ * inputElements);
                }
            }
            outputPrefix += inputElements;
        }
    }

    __aicore__ inline void CopySegments(uint32_t inputStart, uint32_t inputEnd,
                                        uint32_t rows, uint32_t outputStrideElements,
                                        uint32_t groupElements,
                                        const LocalTensor<int32_t>& source,
                                        const LocalTensor<int32_t>& output)
    {
        uint32_t sourceBase = 0;
        uint32_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            if (inputElements != 0) {
                if (outputPrefix + inputElements > groupElements) {
                    return;
                }
                if constexpr (MaskedRows) {
                    const uint32_t phase = outputPrefix & (GATHER_BLOCK_ELEMENTS - 1);
                    const uint32_t sourceStrideElements = AlignUpGatherElements(phase + inputElements);
                    CopyMaskedRows(output[outputPrefix - phase], source[sourceBase], rows,
                                   outputStrideElements, sourceStrideElements, phase, inputElements);
                    sourceBase += batchRows_ * sourceStrideElements;
                } else {
                    for (uint32_t row = 0; row < rows; ++row) {
                        CopyOneRow(output[row * outputStrideElements + outputPrefix],
                                   source[sourceBase + row * inputElements], inputElements);
                    }
                    sourceBase += AlignUpGatherElements(batchRows_ * inputElements);
                }
            }
            outputPrefix += inputElements;
        }
    }

    __aicore__ inline void CopyMaskedRows(const LocalTensor<int32_t>& dst,
                                          const LocalTensor<int32_t>& src, uint32_t rows,
                                          uint32_t dstRowElements, uint32_t srcRowElements,
                                          uint32_t phase, uint32_t elements)
    {
        constexpr uint32_t COPY_VECTOR_ELEMENTS = 64;
        if (rows == 0 || rows > 255 || dstRowElements == 0 || srcRowElements == 0 ||
            dstRowElements % GATHER_BLOCK_ELEMENTS != 0 ||
            srcRowElements % GATHER_BLOCK_ELEMENTS != 0 || phase >= GATHER_BLOCK_ELEMENTS ||
            elements > srcRowElements - phase ||
            dstRowElements / GATHER_BLOCK_ELEMENTS > MAX_UINT16_VALUE ||
            srcRowElements / GATHER_BLOCK_ELEMENTS > MAX_UINT16_VALUE) {
            valid_ = false;
            return;
        }
        const CopyRepeatParams copyParams{1, 1,
            static_cast<uint16_t>(dstRowElements / GATHER_BLOCK_ELEMENTS),
            static_cast<uint16_t>(srcRowElements / GATHER_BLOCK_ELEMENTS)};
        const uint32_t occupiedElements = phase + elements;
        for (uint32_t offset = 0; offset < occupiedElements; offset += COPY_VECTOR_ELEMENTS) {
            const uint32_t lowLane = offset == 0 ? phase : 0;
            const uint32_t highLane = static_cast<uint32_t>(MinU64(
                COPY_VECTOR_ELEMENTS, occupiedElements - offset));
            if (lowLane >= highLane) {
                continue;
            }
            const uint32_t laneCount = highLane - lowLane;
            const uint64_t laneMask = laneCount == COPY_VECTOR_ELEMENTS ? ~0ULL :
                ((1ULL << laneCount) - 1ULL) << lowLane;
            uint64_t mask[2] = {laneMask, 0};
            Copy<int32_t>(dst[offset], src[offset], mask, static_cast<uint8_t>(rows), copyParams);
        }
    }

    __aicore__ inline void CopyOneRow(const LocalTensor<int32_t>& dst,
                                      const LocalTensor<int32_t>& src, uint32_t elements)
    {
        constexpr uint32_t COPY_VECTOR_ELEMENTS = 64;
        const CopyRepeatParams copyParams{1, 1, 8, 8};
        for (uint32_t offset = 0; offset < elements; offset += COPY_VECTOR_ELEMENTS) {
            const uint32_t count = static_cast<uint32_t>(MinU64(
                COPY_VECTOR_ELEMENTS, elements - offset));
            Copy<int32_t>(dst[offset], src[offset], count, 1, copyParams);
        }
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_.FetchEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatRowGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t planIndex_{0};
    uint32_t batchRows_{0};
    uint32_t groupCount_{0};
    uint32_t sourceBufferBytes_{0};
    uint32_t outputBufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    TBuf<TPosition::VECIN> sourceBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
};

// 输入先按输入段连续搬入单个 UB source 槽，再直接经 MTE3 写入目标的对应列。
// 与 RowGather 相比，该路径完全省去 offset 建表、Gather 和 output UB；适合高输入数、
// 窄段且行切分已饱和的 FP32 拓扑。
class KernelConcatRowDirectMte {
public:
    __aicore__ inline explicit KernelConcatRowDirectMte(TPipe& pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatRowGatherTilingData* tilingData)
    {
        tilingData_ = tilingData;
        inputCount_ = tilingData_->inputCount;
        outputRowElements_ = tilingData_->outputRowElements;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));

        const uint32_t highCoreCount = tilingData_->highCoreCount;
        if (blockIdx_ < highCoreCount) {
            rowStart_ = static_cast<uint64_t>(blockIdx_) * tilingData_->rowsPerCore1;
            rowCount_ = tilingData_->rowsPerCore1;
            planIndex_ = 1;
        } else {
            rowStart_ = static_cast<uint64_t>(highCoreCount) * tilingData_->rowsPerCore1 +
                static_cast<uint64_t>(blockIdx_ - highCoreCount) * tilingData_->rowsPerCore0;
            rowCount_ = tilingData_->rowsPerCore0;
            planIndex_ = 0;
        }
        if (planIndex_ == 0) {
            batchRows_ = tilingData_->batchRows0;
            sourceBatchRows_ = tilingData_->sourceBatchRows0;
            groupCount_ = tilingData_->groupCount0;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes0;
        } else {
            batchRows_ = tilingData_->batchRows1;
            sourceBatchRows_ = tilingData_->sourceBatchRows1;
            groupCount_ = tilingData_->groupCount1;
            sourceBufferBytes_ = tilingData_->sourceBufferBytes1;
        }

        valid_ = highCoreCount <= blockNum_ && rowStart_ <= tilingData_->outer &&
                 rowCount_ <= tilingData_->outer - rowStart_ &&
                 tilingData_->dtypeSize == sizeof(int32_t) && rowCount_ != 0 &&
                 batchRows_ != 0 && sourceBatchRows_ == batchRows_ &&
                 batchRows_ <= MAX_BLOCK_COUNT && groupCount_ != 0 &&
                 sourceBufferBytes_ != 0 && sourceBufferBytes_ % UB_BLOCK_BYTES == 0;
        if (!valid_) {
            return;
        }
        // VECCALC UB 可以同时作为 MTE2 目的和 MTE3 源，不引入任何 Vector 执行。
        pipe_.InitBuffer(sourceBuffer_, sourceBufferBytes_);
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        uint32_t inputStart = 0;
        uint32_t outputStart = 0;
        const uint32_t planBase = planIndex_ * GATHER_PLAN_INPUT_CAPACITY;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            const uint32_t planGroup = planBase + group;
            const uint32_t inputEnd = tilingData_->groupInputEnd[planGroup];
            const uint32_t groupElements = tilingData_->groupElements[planGroup];
            const uint32_t groupSourceElements = tilingData_->groupSourceElements[planGroup];
            if (inputEnd <= inputStart || inputEnd > inputCount_ || groupElements == 0 ||
                groupSourceElements == 0 || outputStart + groupElements > outputRowElements_ ||
                static_cast<uint64_t>(groupSourceElements) * sizeof(int32_t) >
                    sourceBufferBytes_) {
                return;
            }
            ProcessGroup(inputStart, inputEnd, outputStart, groupElements,
                groupSourceElements);
            if (!valid_) {
                return;
            }
            inputStart = inputEnd;
            outputStart += groupElements;
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t inputStart, uint32_t inputEnd,
        uint32_t outputStart, uint32_t groupElements, uint32_t groupSourceElements)
    {
        LocalTensor<int32_t> source = sourceBuffer_.Get<int32_t>();
        source.SetSize(groupSourceElements);
        for (uint64_t batchOffset = 0; batchOffset < rowCount_; batchOffset += batchRows_) {
            const uint32_t currentBatch = static_cast<uint32_t>(MinU64(
                batchRows_, rowCount_ - batchOffset));
            const uint64_t batchStart = rowStart_ + batchOffset;
            CopyInputs(inputStart, inputEnd, batchStart, currentBatch,
                groupSourceElements, source);
            if (!valid_) {
                return;
            }
            SyncMte2ToMte3();
            CopyOutputs(inputStart, inputEnd, outputStart, groupElements, batchStart,
                currentBatch, groupSourceElements, source);
            if (!valid_) {
                return;
            }
            // source 会被下一批 MTE2 覆盖，必须等本批所有 MTE3 读取完成。
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void CopyInputs(uint32_t inputStart, uint32_t inputEnd,
        uint64_t batchStart, uint32_t rows, uint32_t groupSourceElements,
        const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t rawSlotElements = static_cast<uint64_t>(sourceBatchRows_) * inputElements;
            if (rawSlotElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            const uint64_t slotElements64 = AlignUpGatherElements(
                static_cast<uint32_t>(rawSlotElements));
            const uint64_t payloadBytes = static_cast<uint64_t>(rows) * inputElements *
                sizeof(int32_t);
            if (slotElements64 > MAX_UINT32 || sourceBase > groupSourceElements ||
                slotElements64 > groupSourceElements - sourceBase ||
                payloadBytes > MAX_UINT32) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int32_t>(input));
                const DataCopyExtParams copyInParams{
                    1, static_cast<uint32_t>(payloadBytes), 0, 0, 0};
                DataCopyPad(source[sourceBase], inputGm_[batchStart * inputElements],
                    copyInParams, padParams);
            }
            sourceBase += static_cast<uint32_t>(slotElements64);
        }
        if (sourceBase != groupSourceElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void CopyOutputs(uint32_t inputStart, uint32_t inputEnd,
        uint32_t outputStart, uint32_t groupElements, uint64_t batchStart,
        uint32_t rows, uint32_t groupSourceElements, const LocalTensor<int32_t>& source)
    {
        uint32_t sourceBase = 0;
        uint32_t outputPrefix = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            const uint32_t inputElements = tilingData_->inputRowElements[input];
            const uint64_t rawSlotElements = static_cast<uint64_t>(sourceBatchRows_) * inputElements;
            if (rawSlotElements > MAX_UINT32) {
                valid_ = false;
                return;
            }
            const uint64_t slotElements64 = AlignUpGatherElements(
                static_cast<uint32_t>(rawSlotElements));
            if (slotElements64 > MAX_UINT32 || sourceBase > groupSourceElements ||
                slotElements64 > groupSourceElements - sourceBase ||
                outputPrefix > groupElements || inputElements > groupElements - outputPrefix) {
                valid_ = false;
                return;
            }
            if (inputElements != 0) {
                const uint64_t outputOffset = batchStart * outputRowElements_ + outputStart +
                    outputPrefix;
                const uint64_t blockLenBytes = static_cast<uint64_t>(inputElements) *
                    sizeof(int32_t);
                // source 每个输入段按 32B 对齐槽存放；非 32B 行宽时，二维 MTE3
                // 必须跳过每行尾部的 UB padding，stride 单位为 32B block。
                const uint64_t sourceStrideBlocks =
                    (slotElements64 - inputElements) / GATHER_BLOCK_ELEMENTS;
                const uint64_t dstStrideBytes = (outputRowElements_ - inputElements) *
                    sizeof(int32_t);
                if (blockLenBytes > MAX_UINT32 || sourceStrideBlocks > MAX_UINT32 ||
                    dstStrideBytes > MAX_UINT32) {
                    valid_ = false;
                    return;
                }
                const DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLenBytes),
                    static_cast<uint32_t>(sourceStrideBlocks),
                    static_cast<uint32_t>(dstStrideBytes), 0};
                DataCopyPad(outputGm_[outputOffset], source[sourceBase], copyOutParams);
            }
            outputPrefix += inputElements;
            sourceBase += static_cast<uint32_t>(slotElements64);
        }
        if (outputPrefix != groupElements || sourceBase != groupSourceElements) {
            valid_ = false;
        }
    }

    __aicore__ inline void SyncMte2ToMte3()
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    TPipe& pipe_;
    const __gm__ ConcatRowGatherTilingData* tilingData_{nullptr};
    uint32_t inputCount_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{1};
    uint32_t planIndex_{0};
    uint32_t batchRows_{0};
    uint32_t sourceBatchRows_{0};
    uint32_t groupCount_{0};
    uint32_t sourceBufferBytes_{0};
    uint64_t rowStart_{0};
    uint64_t rowCount_{0};
    uint64_t outputRowElements_{0};
    bool valid_{false};
    ListTensorDesc inputList_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
};

class KernelConcatPreparedFlatStatic {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatPreparedFlatTwoCommandsTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileBytes = tilingData_->tileBytes;
        LocalTensor<int8_t> ping(TPosition::VECCALC, 0, tileBytes);
        LocalTensor<int8_t> pong(TPosition::VECCALC, tileBytes, tileBytes);
        const uint32_t command = blockIdx_ * 2;

        CopyIn(ping, command);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        CopyOut(ping, command);

        CopyIn(pong, command + 1);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
        CopyOut(pong, command + 1);

        // 两块 UB 在本次 Kernel 内均不再复用，仅需等待全部写回完成后退出。
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

private:
    __aicore__ inline void CopyIn(const LocalTensor<int8_t>& local, uint32_t command)
    {
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(tilingData_->commandInput[command]));
        inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const DataCopyExtParams params{1, tilingData_->commandBytes[command], 0, 0, 0};
        const DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, inputGm_[tilingData_->commandInputOffset[command]], params, padParams);
    }

    __aicore__ inline void CopyOut(const LocalTensor<int8_t>& local, uint32_t command)
    {
        const DataCopyExtParams params{1, tilingData_->commandBytes[command], 0, 0, 0};
        DataCopyPad(outputGm_[tilingData_->commandOutputOffset[command]], local, params);
    }

private:
    const __gm__ ConcatPreparedFlatTwoCommandsTilingData* tilingData_{nullptr};
    uint32_t blockIdx_{0};
    ListTensorDesc inputList_;
    GlobalTensor<int8_t> inputGm_;
    GlobalTensor<int8_t> outputGm_;
};

template <bool BypassInputL2>
class KernelConcatPreparedRectStatic {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const __gm__ ConcatPreparedRectTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int8_t> local(TPosition::VECCALC, 0, tilingData_->tileBytes);
        const uint32_t command = blockIdx_ * PREPARED_RECT_COMMANDS_PER_CORE;

        CopyCommand(local, command);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CopyCommand(local, command + 1);

        // 前两条命令由 Host 保证非空；仅扫描可能存在的尾命令，避免自然的
        // 两/三命令路径为每条命令额外付出空命令判断和完成计数开销。
        for (uint32_t commandOffset = 2;
             commandOffset < PREPARED_RECT_COMMANDS_PER_CORE; ++commandOffset) {
            const uint64_t header = tilingData_->commandData[
                (command + commandOffset) * PREPARED_RECT_WORDS_PER_COMMAND];
            if (((header >> 16) & 0xffffULL) == 0 || (header >> 32) == 0) {
                break;
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            CopyCommand(local, command + commandOffset);
        }

        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

private:
    __aicore__ inline void CopyCommand(const LocalTensor<int8_t>& local, uint32_t command)
    {
        const uint32_t word = command * PREPARED_RECT_WORDS_PER_COMMAND;
        const uint64_t header = tilingData_->commandData[word];
        const uint64_t strides = tilingData_->commandData[word + 3];
        inputGm_.SetGlobalBuffer(inputList_.GetDataPtr<int8_t>(header & 0xffffU));
        uint64_t blockLen = header >> 32;
        if constexpr (BypassInputL2) {
            inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            blockLen &= PREPARED_RECT_BLOCK_LEN_MASK;
        }
        const DataCopyExtParams copyInParams{
            static_cast<uint16_t>((header >> 16) & 0xffffU),
            static_cast<uint32_t>(blockLen),
            static_cast<uint32_t>(strides & MAX_UINT32), 0, 0};
        const DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, inputGm_[tilingData_->commandData[word + 1]],
                    copyInParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        const DataCopyExtParams copyOutParams{
            static_cast<uint16_t>((header >> 16) & 0xffffU),
            static_cast<uint32_t>(blockLen), 0,
            static_cast<uint32_t>(strides >> 32), 0};
        DataCopyPad(outputGm_[tilingData_->commandData[word + 2]], local, copyOutParams);
    }

private:
    const __gm__ ConcatPreparedRectTilingData* tilingData_{nullptr};
    uint32_t blockIdx_{0};
    ListTensorDesc inputList_;
    GlobalTensor<int8_t> inputGm_;
    GlobalTensor<int8_t> outputGm_;
};

} // namespace ConcatKernel

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(10)) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIV_ONLY);
        AscendC::InitSocState();
        const __gm__ ConcatPreparedFlatTwoCommandsTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedFlatTwoCommandsTilingData*>(tiling);
        ConcatKernel::KernelConcatPreparedFlatStatic staticOp;
        staticOp.Init(inputs, output, tilingData);
        staticOp.Process();
        return;
    }

    if (TILING_KEY_IS(12)) {
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_AIV_ONLY);
        AscendC::InitSocState();
        const __gm__ ConcatPreparedRectTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedRectTilingData*>(tiling);
        ConcatKernel::KernelConcatPreparedRectStatic<false> staticRectOp;
        staticRectOp.Init(inputs, output, tilingData);
        staticRectOp.Process();
        return;
    }

    AscendC::TPipe pipe;

#if defined(CONCAT_ENABLE_ROW_PACKED_UB_TILING_KEY)
    if (TILING_KEY_IS(65)) {
        KERNEL_TASK_TYPE(65, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatPackedRowTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPackedRowTilingData*>(tiling);
        ConcatKernel::KernelConcatPackedRow packedRowOp(pipe);
        packedRowOp.Init(inputs, output, tilingData);
        packedRowOp.Process();
        return;
    }
#endif

    if (TILING_KEY_IS(8)) {
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatGridSpanTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatGridSpanTilingData*>(tiling);
        ConcatKernel::KernelConcat<ConcatKernel::SINGLE_BUFFER_NUM> singleBufferOp(pipe);
        singleBufferOp.Init(inputs, output, tilingData->inputCount,
                            tilingData->dtypeSize, tilingData->tileBytes);
        singleBufferOp.ProcessGridSpan(tilingData);
        return;
    }

    if (TILING_KEY_IS(11)) {
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatPreparedRectTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedRectTilingData*>(tiling);
        ConcatKernel::KernelConcat<ConcatKernel::SINGLE_BUFFER_NUM> preparedRectOp(pipe);
        preparedRectOp.Init(inputs, output, tilingData->inputCount,
                            tilingData->dtypeSize, tilingData->tileBytes);
        preparedRectOp.ProcessPreparedRect(tilingData);
        return;
    }

    ConcatKernel::KernelConcat<ConcatKernel::DOUBLE_BUFFER_NUM> op(pipe);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ConcatFlatTilingData, tilingData, tiling);
        op.Init(inputs, output, tilingData.inputCount, tilingData.dtypeSize, tilingData.tileBytes);
        op.ProcessFlat(tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ConcatRowTilingData, tilingData, tiling);
        op.Init(inputs, output, tilingData.inputCount, tilingData.dtypeSize, tilingData.tileBytes);
        op.ProcessRows(tilingData.outer, tilingData.outputRowBytes);
    } else if (TILING_KEY_IS(29)) {
        KERNEL_TASK_TYPE(29, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ConcatRowTilingData, tilingData, tiling);
        ConcatKernel::KernelConcat<ConcatKernel::DOUBLE_BUFFER_NUM, true> bypassRowOp(pipe);
        bypassRowOp.Init(inputs, output, tilingData.inputCount, tilingData.dtypeSize,
            tilingData.tileBytes);
        bypassRowOp.ProcessRows(tilingData.outer, tilingData.outputRowBytes);
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ConcatGridTilingData, tilingData, tiling);
        op.Init(inputs, output, tilingData.inputCount, tilingData.dtypeSize, tilingData.tileBytes);
        op.ProcessGrid(tilingData);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ConcatInputLptTilingData, tilingData, tiling);
        op.Init(inputs, output, tilingData.inputCount, tilingData.dtypeSize, tilingData.tileBytes);
        op.ProcessInputLpt(tilingData);
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatGridSpanTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatGridSpanTilingData*>(tiling);
        op.Init(inputs, output, tilingData->inputCount, tilingData->dtypeSize, tilingData->tileBytes);
        op.ProcessGridSpan(tilingData);
    } else if (TILING_KEY_IS(6)) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatPreparedFlatTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedFlatTilingData*>(tiling);
        op.Init(inputs, output, tilingData->inputCount, tilingData->dtypeSize, tilingData->tileBytes);
        op.ProcessPreparedFlat(tilingData);
    } else if (TILING_KEY_IS(7)) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, true> gatherOp(pipe);
        gatherOp.Init(inputs, output, tilingData);
        gatherOp.Process();
    } else if (TILING_KEY_IS(42)) {
        KERNEL_TASK_TYPE(42, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherAdaptiveGroupBatch adaptiveGroupBatchOp(pipe);
        adaptiveGroupBatchOp.Init(inputs, output, tilingData);
        adaptiveGroupBatchOp.Process();
    } else if (TILING_KEY_IS(43)) {
        KERNEL_TASK_TYPE(43, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherOffsetOutputAlias offsetOutputAliasOp(pipe);
        offsetOutputAliasOp.Init(inputs, output, tilingData);
        offsetOutputAliasOp.Process();
    } else if (TILING_KEY_IS(45)) {
        KERNEL_TASK_TYPE(45, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherSourcePingPongOffsetOutputAlias combinedOp(pipe);
        combinedOp.Init(inputs, output, tilingData);
        combinedOp.Process();
    } else if (TILING_KEY_IS(46)) {
        KERNEL_TASK_TYPE(46, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatPairGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPairGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatPairGather pairGatherOp(pipe);
        pairGatherOp.Init(inputs, output, tilingData);
        pairGatherOp.Process();
    } else if (TILING_KEY_IS(47)) {
        KERNEL_TASK_TYPE(47, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherSuffixPrefetch suffixPrefetchOp(pipe);
        suffixPrefetchOp.Init(inputs, output, tilingData);
        suffixPrefetchOp.Process();
    } else if (TILING_KEY_IS(44)) {
        KERNEL_TASK_TYPE(44, KERNEL_TYPE_MIX_AIC_1_2);
        // 该分支只验证混合任务的启动和 AIV 子核映射。正式异构路径会让 AIC 领取不重叠
        // 的行块；在此之前 AIC 必须无副作用返回。
        if ASCEND_IS_AIC {
            return;
        }
        // 该探针不读取 workspace；不能把可选的空地址误当作 AIV 的启动失败。
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, true, false, false, false, false, false, false, false,
            false, false, false, true> mixedAivProbeOp(pipe);
        mixedAivProbeOp.Init(inputs, output, tilingData);
        mixedAivProbeOp.Process();
    } else if (TILING_KEY_IS(40)) {
        KERNEL_TASK_TYPE(40, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherTwoPlane twoPlaneOp(pipe);
        twoPlaneOp.Init(inputs, output, tilingData);
        twoPlaneOp.Process();
    } else if (TILING_KEY_IS(33)) {
        // key33 已退出常规调度。保持 AIV 元数据，避免未使用的实验分支把安全路径也
        // 升级为 MIX 调度；若未来重新启用 SyncAll，必须同步恢复 MIX 元数据并重新验收。
        KERNEL_TASK_TYPE(33, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        // SetScheduleMode(1) 会在 workspace 起始处保留系统同步区；共享 seed 必须从
        // 用户 workspace 起点访问，否则会与 SyncAll 的系统状态重叠。
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, false, false, true> sharedSeedOp(pipe);
        sharedSeedOp.Init(inputs, output, tilingData, userWorkspace);
        sharedSeedOp.Process();
    } else if (TILING_KEY_IS(34)) {
        KERNEL_TASK_TYPE(34, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, true, false, false, true> coreSwizzleOp(pipe);
        coreSwizzleOp.Init(inputs, output, tilingData);
        coreSwizzleOp.Process();
    } else if (TILING_KEY_IS(35)) {
        KERNEL_TASK_TYPE(35, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, true, false, false, false, true> groupPhaseOp(pipe);
        groupPhaseOp.Init(inputs, output, tilingData);
        groupPhaseOp.Process();
    } else if (TILING_KEY_IS(36)) {
        KERNEL_TASK_TYPE(36, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatBandGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatBandGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatBandGather bandGatherOp(pipe);
        bandGatherOp.Init(inputs, output, tilingData);
        bandGatherOp.Process();
    } else if (TILING_KEY_IS(38)) {
        KERNEL_TASK_TYPE(38, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, true> wavefrontOp(pipe);
        wavefrontOp.Init(inputs, output, tilingData);
        wavefrontOp.Process();
    } else if (TILING_KEY_IS(39)) {
        KERNEL_TASK_TYPE(39, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, true, true>
            directWavefrontOp(pipe);
        directWavefrontOp.Init(inputs, output, tilingData);
        directWavefrontOp.Process();
    } else if (TILING_KEY_IS(15)) {
        KERNEL_TASK_TYPE(15, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<true, false> outputPingPongOp(pipe);
        outputPingPongOp.Init(inputs, output, tilingData);
        outputPingPongOp.Process();
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, true> sourcePingPongOp(pipe);
        sourcePingPongOp.Init(inputs, output, tilingData);
        sourcePingPongOp.Process();
    } else if (TILING_KEY_IS(17)) {
        KERNEL_TASK_TYPE(17, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false> mergedBatchOp(pipe);
        mergedBatchOp.Init(inputs, output, tilingData);
        mergedBatchOp.Process();
    } else if (TILING_KEY_IS(18)) {
        KERNEL_TASK_TYPE(18, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true> stagedSourceOp(pipe);
        stagedSourceOp.Init(inputs, output, tilingData);
        stagedSourceOp.Process();
    } else if (TILING_KEY_IS(19)) {
        KERNEL_TASK_TYPE(19, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherBlock blockGatherOp(pipe);
        blockGatherOp.Init(inputs, output, tilingData);
        blockGatherOp.Process();
    } else if (TILING_KEY_IS(20)) {
        KERNEL_TASK_TYPE(20, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, true> rowMajorStagedOp(pipe);
        rowMajorStagedOp.Init(inputs, output, tilingData);
        rowMajorStagedOp.Process();
    } else if (TILING_KEY_IS(21)) {
        KERNEL_TASK_TYPE(21, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, true> splitOutputOp(pipe);
        splitOutputOp.Init(inputs, output, tilingData);
        splitOutputOp.Process();
    } else if (TILING_KEY_IS(23)) {
        KERNEL_TASK_TYPE(23, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true, false, false, false, true>
            fullStageOffsetOp(pipe);
        fullStageOffsetOp.Init(inputs, output, tilingData);
        fullStageOffsetOp.Process();
    } else if (TILING_KEY_IS(24)) {
        KERNEL_TASK_TYPE(24, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true, false, false, false, false,
            true> compactStageOffsetOp(pipe);
        compactStageOffsetOp.Init(inputs, output, tilingData);
        compactStageOffsetOp.Process();
    } else if (TILING_KEY_IS(25)) {
        KERNEL_TASK_TYPE(25, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, true> microRegisterOp(pipe);
        microRegisterOp.Init(inputs, output, tilingData);
        microRegisterOp.Process();
    } else if (TILING_KEY_IS(26)) {
        KERNEL_TASK_TYPE(26, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowDirectMte directMteOp(pipe);
        directMteOp.Init(inputs, output, tilingData);
        directMteOp.Process();
    } else if (TILING_KEY_IS(27)) {
        KERNEL_TASK_TYPE(27, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true, false, false, false, false,
            false, false, true> rollingStageOffsetOp(pipe);
        rollingStageOffsetOp.Init(inputs, output, tilingData);
        rollingStageOffsetOp.Process();
    } else if (TILING_KEY_IS(28)) {
        KERNEL_TASK_TYPE(28, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true, false, false, false, false,
            false, false, false, true> deltaStagedOffsetOp(pipe);
        deltaStagedOffsetOp.Init(inputs, output, tilingData);
        deltaStagedOffsetOp.Process();
    } else if (TILING_KEY_IS(22)) {
        KERNEL_TASK_TYPE(22, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, true, false, false, true>
            cachedOffsetStagedOp(pipe);
        cachedOffsetStagedOp.Init(inputs, output, tilingData);
        cachedOffsetStagedOp.Process();
    } else if (TILING_KEY_IS(13)) {
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGatherResident residentGatherOp(pipe);
        residentGatherOp.Init(inputs, output, tilingData);
        residentGatherOp.Process();
    } else if (TILING_KEY_IS(14)) {
        KERNEL_TASK_TYPE(14, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowDirect<> directOp(pipe);
        directOp.Init(inputs, output, tilingData);
        directOp.Process();
    } else if (TILING_KEY_IS(30)) {
        KERNEL_TASK_TYPE(30, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowDirect<true> maskedCopyOp(pipe);
        maskedCopyOp.Init(inputs, output, tilingData);
        maskedCopyOp.Process();
    } else if (TILING_KEY_IS(31)) {
        KERNEL_TASK_TYPE(31, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowGatherTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowGatherTilingData*>(tiling);
        ConcatKernel::KernelConcatRowGather<false, false, false, false, false, false, false,
            false, false, false, false, false, true> partitionPipelineOp(pipe);
        partitionPipelineOp.Init(inputs, output, tilingData);
        partitionPipelineOp.Process();
    } else if (TILING_KEY_IS(9)) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatPreparedFlatTwoCommandsTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedFlatTwoCommandsTilingData*>(tiling);
        op.Init(inputs, output, tilingData->inputCount, tilingData->dtypeSize, tilingData->tileBytes);
        op.ProcessPreparedFlatTwoCommands(tilingData);
    }

#if defined(CONCAT_ENABLE_PREPARED_RECT_L2_KEY)
    if (TILING_KEY_IS(48)) {
        KERNEL_TASK_TYPE(48, KERNEL_TYPE_AIV_ONLY);
        AscendC::InitSocState();
        const __gm__ ConcatPreparedRectTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatPreparedRectTilingData*>(tiling);
        ConcatKernel::KernelConcatPreparedRectStatic<true> staticRectOp;
        staticRectOp.Init(inputs, output, tilingData);
        staticRectOp.Process();
    }
#endif

#if defined(CONCAT_ENABLE_ROW_BYTES_TILING_KEY)
    if (TILING_KEY_IS(49)) {
        KERNEL_TASK_TYPE(49, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowPrecomputedTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowPrecomputedTilingData*>(tiling);
        ConcatKernel::KernelConcat<ConcatKernel::DOUBLE_BUFFER_NUM, true> precomputedRowOp(pipe);
        precomputedRowOp.Init(inputs, output, tilingData->inputCount, tilingData->dtypeSize,
            tilingData->tileBytes);
        precomputedRowOp.ProcessRowsPrecomputed(tilingData);
    }
#endif

#if defined(CONCAT_ENABLE_ROW_BYTES_SINGLE_BUFFER_TILING_KEY)
    if (TILING_KEY_IS(55)) {
        KERNEL_TASK_TYPE(55, KERNEL_TYPE_AIV_ONLY);
        const __gm__ ConcatRowPrecomputedTilingData* tilingData =
            reinterpret_cast<const __gm__ ConcatRowPrecomputedTilingData*>(tiling);
        ConcatKernel::KernelConcat<ConcatKernel::SINGLE_BUFFER_NUM, true> singleBufferRowOp(pipe);
        singleBufferRowOp.Init(inputs, output, tilingData->inputCount, tilingData->dtypeSize,
            tilingData->tileBytes);
        singleBufferRowOp.ProcessRowsPrecomputed(tilingData);
    }
#endif
}
