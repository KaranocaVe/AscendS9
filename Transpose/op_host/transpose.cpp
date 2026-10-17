#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVE_BYTES = 8 * 1024;
constexpr uint64_t COPY_BUFFER_FACTOR = 2;
constexpr uint64_t MATRIX_BUFFER_FACTOR = 2;
constexpr uint64_t DATA_COPY_BLOCK_COUNT_MAX = 4095;
constexpr uint64_t DATA_COPY_BLOCK_LEN_MAX = (1ULL << 21) - 1;
constexpr uint64_t VECTOR_REPEAT_MAX = std::numeric_limits<uint8_t>::max();
constexpr uint64_t B16_REPEAT_MAX_ELEMENTS = VECTOR_REPEAT_MAX * 16;
constexpr uint64_t B32_REPEAT_MAX_ELEMENTS = VECTOR_REPEAT_MAX * 8;
constexpr uint64_t GATHER_B16_REPEAT_MAX_ELEMENTS = VECTOR_REPEAT_MAX * 128;
constexpr uint64_t GATHER_B32_REPEAT_MAX_ELEMENTS = VECTOR_REPEAT_MAX * 64;
constexpr uint64_t MATRIX_ROW_GROUP = 16;
constexpr uint64_t TRANS_DATA_ADDRESS_COUNT = 16;
constexpr uint64_t STRIDED_MATRIX_MAX_X = std::numeric_limits<uint16_t>::max();
constexpr uint64_t RECORD_PACK_BUFFER_FACTOR = 2;
// 小于 4 KiB 的离散搬运通常由指令启动开销主导；Vector 转置按单个16地址组折算。
constexpr uint64_t DMA_CALL_EQUIVALENT_BYTES = 4 * 1024;
constexpr uint64_t TRANS_DATA_CALL_EQUIVALENT_BYTES = TRANS_DATA_ADDRESS_COUNT * BLOCK_BYTES;
constexpr uint64_t GATHER_INSTRUCTION_EQUIVALENT_BYTES = 8 * BLOCK_BYTES;
constexpr uint64_t INPUT_BLOCK_EQUIVALENT_BYTES = BLOCK_BYTES;
// int8 二维搬入的短 burst 更容易暴露每个 block 的 MTE2 启动开销，单独提高模型权重。
constexpr uint64_t DENSE_B8_INPUT_BLOCK_EQUIVALENT_BYTES = 56;
constexpr uint64_t OUTPUT_BLOCK_EQUIVALENT_BYTES = 2 * BLOCK_BYTES;
constexpr uint64_t ACTIVE_CORE_EQUIVALENT_BYTES = 8 * BLOCK_BYTES;
constexpr uint64_t SINGLE_TBUF_RESERVE_BYTES = 16 * 1024;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const uint64_t increment = alignment - remainder;
    return value > std::numeric_limits<uint64_t>::max() - increment ?
        std::numeric_limits<uint64_t>::max() : value + increment;
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

uint64_t GreatestCommonDivisor(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

bool ParsePermutation(const gert::Shape& shape, const gert::ContinuousVector* dimsAttr,
    std::array<uint32_t, optiling::TRANSPOSE_MAX_RANK>& perm)
{
    const size_t rankSize = shape.GetDimNum();
    perm.fill(0);
    if (rankSize > optiling::TRANSPOSE_MAX_RANK || dimsAttr == nullptr ||
        dimsAttr->GetSize() != rankSize) {
        return false;
    }
    const uint32_t rank = static_cast<uint32_t>(rankSize);

    std::array<bool, optiling::TRANSPOSE_MAX_RANK> used{};
    const auto* dims = static_cast<const int64_t*>(dimsAttr->GetData());
    if (rank != 0 && dims == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t axis = dims[i];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || used[static_cast<size_t>(axis)]) {
            return false;
        }
        perm[static_cast<size_t>(i)] = static_cast<uint32_t>(axis);
        used[static_cast<size_t>(axis)] = true;
    }
    return true;
}

struct MergedTopology {
    uint32_t rank{0};
    std::array<uint32_t, optiling::TRANSPOSE_MAX_RANK> perm{};
    std::array<uint64_t, optiling::TRANSPOSE_MAX_RANK> sourceDims{};
    std::vector<uint32_t> effectivePerm;
    std::vector<uint64_t> effectiveDims;
    bool isIdentity{true};
};

struct OutputRun {
    uint32_t sourceStart{0};
    uint64_t elements{1};
};

struct BlockSwapLayout {
    uint64_t batch{1};
    uint64_t leftOuter{1};
    uint64_t fixed{1};
    uint64_t leftInner{1};
    uint64_t right{1};
    uint64_t record{1};
};

bool BuildMergedTopology(const std::array<uint64_t, optiling::TRANSPOSE_MAX_RANK>& shape,
    const std::array<uint32_t, optiling::TRANSPOSE_MAX_RANK>& perm, uint32_t rank,
    MergedTopology& topology)
{
    std::array<int32_t, optiling::TRANSPOSE_MAX_RANK> activeIndex{};
    activeIndex.fill(-1);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] != 1) {
            activeIndex[axis] = static_cast<int32_t>(topology.effectiveDims.size());
            topology.effectiveDims.push_back(shape[axis]);
        }
    }
    for (uint32_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
        const int32_t compactAxis = activeIndex[perm[outputAxis]];
        if (compactAxis >= 0) {
            topology.effectivePerm.push_back(static_cast<uint32_t>(compactAxis));
        }
    }

    topology.isIdentity = topology.effectivePerm.size() == topology.effectiveDims.size();
    for (size_t i = 0; i < topology.effectivePerm.size(); ++i) {
        if (topology.effectivePerm[i] != i) {
            topology.isIdentity = false;
            break;
        }
    }
    if (topology.effectivePerm.empty()) {
        return true;
    }

    std::vector<OutputRun> runs;
    uint32_t previousSource = optiling::TRANSPOSE_MAX_RANK;
    for (uint32_t sourceAxis : topology.effectivePerm) {
        if (runs.empty() || sourceAxis != previousSource + 1) {
            OutputRun run;
            run.sourceStart = sourceAxis;
            run.elements = topology.effectiveDims[sourceAxis];
            runs.push_back(run);
        } else if (MulOverflow(runs.back().elements, topology.effectiveDims[sourceAxis],
            runs.back().elements)) {
            return false;
        }
        previousSource = sourceAxis;
    }
    if (runs.size() > optiling::TRANSPOSE_MAX_RANK) {
        return false;
    }

    std::vector<uint32_t> sourceOrder(runs.size());
    for (uint32_t i = 0; i < runs.size(); ++i) {
        sourceOrder[i] = i;
    }
    std::sort(sourceOrder.begin(), sourceOrder.end(), [&runs](uint32_t lhs, uint32_t rhs) {
        return runs[lhs].sourceStart < runs[rhs].sourceStart;
    });
    for (uint32_t sourceBlock = 0; sourceBlock < sourceOrder.size(); ++sourceBlock) {
        const uint32_t outputBlock = sourceOrder[sourceBlock];
        topology.sourceDims[sourceBlock] = runs[outputBlock].elements;
        topology.perm[outputBlock] = sourceBlock;
    }
    topology.rank = static_cast<uint32_t>(runs.size());
    return true;
}

bool ProductRange(const std::vector<uint64_t>& dims, size_t begin, size_t end, uint64_t& result)
{
    result = 1;
    for (size_t axis = begin; axis < end; ++axis) {
        if (MulOverflow(result, dims[axis], result)) {
            return false;
        }
    }
    return true;
}

bool MatchSourceRange(const std::vector<uint32_t>& perm, size_t& outputAxis, size_t begin, size_t end)
{
    for (size_t sourceAxis = begin; sourceAxis < end; ++sourceAxis, ++outputAxis) {
        if (outputAxis >= perm.size() || perm[outputAxis] != sourceAxis) {
            return false;
        }
    }
    return true;
}

bool BuildBlockSwapLayout(const MergedTopology& topology, BlockSwapLayout& layout)
{
    const size_t rank = topology.effectivePerm.size();
    if (rank < 2) {
        return false;
    }

    // 枚举连续轴组，使输入 [B, Lo, F, Li, R, S] 转换为输出 [B, R, F, Lo, Li, S]。
    for (size_t batchEnd = rank; batchEnd-- > 0;) {
        for (size_t recordStart = rank; recordStart > batchEnd; --recordStart) {
            for (size_t rightStart = batchEnd + 1; rightStart < recordStart; ++rightStart) {
                for (size_t fixedStart = batchEnd; fixedStart <= rightStart; ++fixedStart) {
                    for (size_t fixedEnd = fixedStart; fixedEnd <= rightStart; ++fixedEnd) {
                        const size_t leftAxes = (fixedStart - batchEnd) + (rightStart - fixedEnd);
                        if (leftAxes == 0) {
                            continue;
                        }
                        size_t outputAxis = 0;
                        const bool matches =
                            MatchSourceRange(topology.effectivePerm, outputAxis, 0, batchEnd) &&
                            MatchSourceRange(topology.effectivePerm, outputAxis, rightStart, recordStart) &&
                            MatchSourceRange(topology.effectivePerm, outputAxis, fixedStart, fixedEnd) &&
                            MatchSourceRange(topology.effectivePerm, outputAxis, batchEnd, fixedStart) &&
                            MatchSourceRange(topology.effectivePerm, outputAxis, fixedEnd, rightStart) &&
                            MatchSourceRange(topology.effectivePerm, outputAxis, recordStart, rank) &&
                            outputAxis == rank;
                        if (!matches) {
                            continue;
                        }

                        return ProductRange(topology.effectiveDims, 0, batchEnd, layout.batch) &&
                               ProductRange(topology.effectiveDims, batchEnd, fixedStart, layout.leftOuter) &&
                               ProductRange(topology.effectiveDims, fixedStart, fixedEnd, layout.fixed) &&
                               ProductRange(topology.effectiveDims, fixedEnd, rightStart, layout.leftInner) &&
                               ProductRange(topology.effectiveDims, rightStart, recordStart, layout.right) &&
                               ProductRange(topology.effectiveDims, recordStart, rank, layout.record);
                    }
                }
            }
        }
    }
    return false;
}

bool Product(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    return !MulOverflow(lhs, rhs, result);
}

uint64_t GetUsableUb(uint64_t ubSize)
{
    return ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : ubSize;
}

bool FitsByteStrideU32(uint64_t elements, uint32_t elementSize)
{
    return elements <= std::numeric_limits<uint32_t>::max() / elementSize;
}

uint64_t SaturatingMul(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0;
    return MulOverflow(lhs, rhs, result) ? std::numeric_limits<uint64_t>::max() : result;
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs)
{
    return rhs > std::numeric_limits<uint64_t>::max() - lhs ?
        std::numeric_limits<uint64_t>::max() : lhs + rhs;
}

uint64_t AddCost(uint64_t cost, uint64_t count, uint64_t unitCost)
{
    return SaturatingAdd(cost, SaturatingMul(count, unitCost));
}

uint64_t GetAlignedTiledExtent(uint64_t total, uint64_t tile, uint64_t alignment)
{
    const uint64_t tileCount = CeilDiv(total, tile);
    if (tileCount == 0) {
        return 0;
    }
    const uint64_t fullExtent = SaturatingMul(tileCount - 1, tile);
    if (fullExtent > total) {
        return std::numeric_limits<uint64_t>::max();
    }
    return SaturatingAdd(fullExtent, AlignUp(total - fullExtent, alignment));
}

uint64_t GetTiledPaddedBytes(uint64_t total, uint64_t tile, uint64_t bytesPerElement)
{
    const uint64_t tileCount = CeilDiv(total, tile);
    if (tileCount == 0) {
        return 0;
    }
    const uint64_t fullTileBytes = AlignUp(SaturatingMul(tile, bytesPerElement), BLOCK_BYTES);
    const uint64_t fullElements = SaturatingMul(tileCount - 1, tile);
    if (fullTileBytes == std::numeric_limits<uint64_t>::max() || fullElements > total) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t tailBytes = AlignUp(SaturatingMul(total - fullElements, bytesPerElement), BLOCK_BYTES);
    return SaturatingAdd(SaturatingMul(tileCount - 1, fullTileBytes), tailBytes);
}

uint64_t GetGatherDispatchCount(uint64_t count, uint32_t elementSize)
{
    const uint64_t elementsPerRepeat = elementSize == 2 ? 128 : 64;
    // 线性 repeat 成本已由有效字节计入，这里只补充完整主体与尾处理的发射开销。
    return static_cast<uint64_t>(count >= elementsPerRepeat) +
           static_cast<uint64_t>(count % elementsPerRepeat != 0);
}

bool SetCopyTiling(optiling::TransposeTilingData& tiling, uint64_t totalElements, uint32_t elementSize,
    uint32_t coreNum, uint64_t ubSize)
{
    const uint64_t alignElements = BLOCK_BYTES / elementSize;
    const uint64_t usableUb = GetUsableUb(ubSize);
    uint64_t maxTile = AlignDown(usableUb / (COPY_BUFFER_FACTOR * elementSize), alignElements);
    if (maxTile == 0) {
        return false;
    }
    const uint64_t targetTile = totalElements == 0 ? alignElements :
        AlignUp(CeilDiv(totalElements, coreNum), alignElements);
    const uint64_t tileElements = std::max<uint64_t>(1, std::min(maxTile, targetTile));
    const uint64_t taskCount = totalElements == 0 ? 1 : CeilDiv(totalElements, tileElements);

    tiling.set_batch(1);
    tiling.set_outer(1);
    tiling.set_rows(1);
    tiling.set_cols(totalElements);
    tiling.set_inner(1);
    tiling.set_record(1);
    tiling.set_tileRows(1);
    tiling.set_tileCols(tileElements);
    tiling.set_taskCount(taskCount);
    tiling.set_bufferNum(1);
    return true;
}

bool SetBlockTransposeTiling(optiling::TransposeTilingData& tiling, uint64_t outer, uint64_t rows,
    uint64_t cols, uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    if (!FitsByteStrideU32(rows, elementSize) || !FitsByteStrideU32(cols, elementSize)) {
        return false;
    }
    const uint64_t rowAlignment = elementSize == 1 ? BLOCK_BYTES : MATRIX_ROW_GROUP;
    const uint64_t colAlignment = BLOCK_BYTES / elementSize;
    const uint64_t usableUb = GetUsableUb(ubSize);
    const uint64_t repeatMaxElements = elementSize == 1 ? DATA_COPY_BLOCK_COUNT_MAX :
        (elementSize == 4 ? B32_REPEAT_MAX_ELEMENTS : B16_REPEAT_MAX_ELEMENTS);
    const uint64_t maxTileRows = AlignDown(
        std::min(AlignUp(rows, rowAlignment), DATA_COPY_BLOCK_COUNT_MAX), rowAlignment);
    uint64_t bestTileRows = 0;
    uint64_t bestTileCols = 0;
    uint64_t bestTaskCount = 0;
    uint64_t bestScore = std::numeric_limits<uint64_t>::max();
    // 同时枚举行列 tile，以实际尾块、DMA burst 和最慢核 wave 评估方案。
    for (uint64_t tileRows = rowAlignment; tileRows <= maxTileRows; tileRows += rowAlignment) {
        const uint64_t denominator = SaturatingMul(tileRows, elementSize * MATRIX_BUFFER_FACTOR);
        if (denominator == 0 || denominator == std::numeric_limits<uint64_t>::max()) {
            continue;
        }
        const uint64_t alignedCols = AlignUp(cols, colAlignment);
        const uint64_t maxTileCols = AlignDown(
            std::min({usableUb / denominator, repeatMaxElements, alignedCols}), colAlignment);
        if (maxTileCols == 0) {
            continue;
        }
        for (uint64_t tileCols = colAlignment; tileCols <= maxTileCols; tileCols += colAlignment) {
            const uint64_t rowTileCount = CeilDiv(rows, tileRows);
            const uint64_t colTileCount = CeilDiv(cols, tileCols);
            uint64_t taskCount = 0;
            uint64_t tilesPerBatch = 0;
            if (!Product(rowTileCount, colTileCount, tilesPerBatch) ||
                !Product(outer, tilesPerBatch, taskCount) || taskCount == 0) {
                continue;
            }

            const uint64_t inputBlocks = SaturatingMul(SaturatingMul(outer, colTileCount), rows);
            const uint64_t outputBlocks = SaturatingMul(SaturatingMul(outer, rowTileCount), cols);
            const uint64_t paddedRows = GetAlignedTiledExtent(rows, tileRows, rowAlignment);
            const uint64_t paddedCols = GetAlignedTiledExtent(cols, tileCols, colAlignment);
            uint64_t paddedElements = SaturatingMul(paddedRows, paddedCols);
            paddedElements = SaturatingMul(paddedElements, outer);
            uint64_t cost = SaturatingMul(paddedElements, elementSize);
            const uint64_t inputBlockCost = elementSize == 1 ?
                DENSE_B8_INPUT_BLOCK_EQUIVALENT_BYTES : INPUT_BLOCK_EQUIVALENT_BYTES;
            cost = AddCost(cost, inputBlocks, inputBlockCost);
            cost = AddCost(cost, outputBlocks, OUTPUT_BLOCK_EQUIVALENT_BYTES);
            cost = AddCost(cost, taskCount, BLOCK_BYTES * 8);
            const uint64_t waves = CeilDiv(taskCount, coreNum);
            uint64_t score = SaturatingMul(CeilDiv(cost, taskCount), waves);
            if (elementSize == 1 && waves == 1) {
                // 单 wave 的完成时间由最重 tile 决定，不能用全体 task 的平均成本掩盖长尾。
                uint64_t maxTaskCost = SaturatingMul(tileRows, tileCols);
                maxTaskCost = AddCost(maxTaskCost, tileRows, inputBlockCost);
                maxTaskCost = AddCost(maxTaskCost, tileCols, OUTPUT_BLOCK_EQUIVALENT_BYTES);
                score = AddCost(maxTaskCost, 1, BLOCK_BYTES * 8);
            }
            // 每个活跃 Vector 核都有固定任务分发成本，按一条256B向量事务折算。
            score = AddCost(score, std::min<uint64_t>(taskCount, coreNum),
                ACTIVE_CORE_EQUIVALENT_BYTES);
            if (score < bestScore ||
                (score == bestScore &&
                    (taskCount < bestTaskCount ||
                        (taskCount == bestTaskCount && tileCols > bestTileCols)))) {
                bestScore = score;
                bestTileRows = tileRows;
                bestTileCols = tileCols;
                bestTaskCount = taskCount;
            }
        }
    }
    if (bestTileRows == 0 || bestTileCols == 0 || bestTaskCount == 0) {
        return false;
    }

    tiling.set_batch(outer);
    tiling.set_outer(outer);
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_inner(1);
    tiling.set_tileRows(bestTileRows);
    tiling.set_tileCols(bestTileCols);
    tiling.set_taskCount(bestTaskCount);
    tiling.set_bufferNum(1);
    return true;
}

bool SetStridedMatrixTiling(optiling::TransposeTilingData& tiling, const BlockSwapLayout& layout,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    uint64_t x = 0;
    if (!Product(layout.leftOuter, layout.leftInner, x)) {
        return false;
    }
    const uint64_t outputRows = AlignUp(x, TRANS_DATA_ADDRESS_COUNT);
    uint64_t outputRowStride = 0;
    if (x > STRIDED_MATRIX_MAX_X || outputRows > STRIDED_MATRIX_MAX_X ||
        layout.leftInner > DATA_COPY_BLOCK_COUNT_MAX ||
        !Product(layout.fixed - 1, x, outputRowStride) ||
        !FitsByteStrideU32(layout.right, elementSize) ||
        !FitsByteStrideU32(outputRowStride, elementSize)) {
        return false;
    }

    const uint64_t usableUb = GetUsableUb(ubSize);
    const uint64_t yAlignment = BLOCK_BYTES / elementSize;
    const uint64_t repeatMaxElements = VECTOR_REPEAT_MAX * yAlignment;
    uint64_t fixedBatches = 0;
    if (!Product(layout.batch, layout.fixed, fixedBatches)) {
        return false;
    }

    uint64_t bestChunkRows = 0;
    uint64_t bestPanelRows = 0;
    uint64_t bestTileY = 0;
    uint64_t bestTaskCount = 0;
    uint64_t bestScore = std::numeric_limits<uint64_t>::max();
    uint64_t bestInputCalls = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t inputBufferNum = 2;
    const uint64_t maxChunkRows = layout.fixed == 1 ?
        std::min(outputRows, AlignDown(DATA_COPY_BLOCK_COUNT_MAX, TRANS_DATA_ADDRESS_COUNT)) : outputRows;
    uint64_t segmentedOutputStride = 0;
    const bool supportsSegmentedOutput = Product(layout.fixed, x, segmentedOutputStride) &&
        FitsByteStrideU32(segmentedOutputStride, elementSize);
    for (uint64_t chunkRows = TRANS_DATA_ADDRESS_COUNT;
         chunkRows <= maxChunkRows; chunkRows += TRANS_DATA_ADDRESS_COUNT) {
        // 双缓冲输入时让输出覆盖不超过八个输入块，保持 MTE2 与 MTE3 的细粒度交叠。
        const uint64_t panelRows = std::min(outputRows,
            SaturatingMul(chunkRows, static_cast<uint64_t>(8)));
        if (panelRows < outputRows && !supportsSegmentedOutput) {
            continue;
        }
        uint64_t rowsPerY = 0;
        uint64_t bytesPerY = 0;
        if (panelRows == 0 ||
            !Product(panelRows + SaturatingMul(chunkRows, inputBufferNum), 1, rowsPerY) ||
            !Product(rowsPerY, elementSize, bytesPerY) || bytesPerY == 0) {
            continue;
        }
        const uint64_t maxTileY = AlignDown(
            std::min({usableUb / bytesPerY, repeatMaxElements, AlignUp(layout.right, yAlignment)}),
            yAlignment);
        if (maxTileY == 0) {
            continue;
        }

        uint64_t sourceSegments = CeilDiv(x, chunkRows);
        if (layout.fixed != 1) {
            sourceSegments = 0;
            for (uint64_t rowStart = 0; rowStart < x; rowStart += chunkRows) {
                const uint64_t rowEnd = std::min(x, rowStart + chunkRows);
                sourceSegments = SaturatingAdd(sourceSegments,
                    CeilDiv(rowEnd, layout.leftInner) - rowStart / layout.leftInner);
            }
        }
        uint64_t maxTileTaskCount = 0;
        if (!Product(fixedBatches, CeilDiv(layout.right, maxTileY), maxTileTaskCount) ||
            maxTileTaskCount == 0) {
            continue;
        }
        // 只有最大列块仍填不满 Vector 核时才展开更小 tile：这类欠并行矩阵可用
        // 额外 DMA 发射换取核间并行；已饱和的长矩阵保持既有最大列块与数据流。
        const uint64_t minTileY = maxTileTaskCount < coreNum ? yAlignment : maxTileY;
        for (uint64_t tileY = minTileY; tileY <= maxTileY; tileY += yAlignment) {
        const uint64_t colTileCount = CeilDiv(layout.right, tileY);
        const uint64_t panelCount = CeilDiv(x, panelRows);
        uint64_t taskCount = 0;
        if (!Product(fixedBatches, colTileCount, taskCount) || taskCount == 0) {
            continue;
        }

        const uint64_t paddedRightBytes = GetTiledPaddedBytes(layout.right, tileY, elementSize);
        const uint64_t inputCalls = SaturatingMul(taskCount, sourceSegments);
        const uint64_t inputBlocks = SaturatingMul(taskCount, x);
        uint64_t mte2Cost = SaturatingMul(SaturatingMul(fixedBatches, x), paddedRightBytes);
        mte2Cost = AddCost(mte2Cost, inputBlocks, INPUT_BLOCK_EQUIVALENT_BYTES);
        mte2Cost = AddCost(mte2Cost, inputCalls, DMA_CALL_EQUIVALENT_BYTES);

        const uint64_t outputBlocks = SaturatingMul(
            SaturatingMul(fixedBatches, layout.right), panelCount);
        uint64_t mte3Cost = SaturatingMul(
            SaturatingMul(fixedBatches, layout.right),
            GetTiledPaddedBytes(x, panelRows, elementSize));
        mte3Cost = AddCost(mte3Cost, outputBlocks, OUTPUT_BLOCK_EQUIVALENT_BYTES);
        mte3Cost = AddCost(mte3Cost, SaturatingMul(taskCount, panelCount),
            DMA_CALL_EQUIVALENT_BYTES);

        const uint64_t transCalls = SaturatingMul(taskCount,
            CeilDiv(x, TRANS_DATA_ADDRESS_COUNT));
        uint64_t vectorCost = SaturatingMul(
            SaturatingMul(fixedBatches,
                GetAlignedTiledExtent(x, panelRows, TRANS_DATA_ADDRESS_COUNT)), paddedRightBytes);
        vectorCost = AddCost(vectorCost, transCalls, TRANS_DATA_CALL_EQUIVALENT_BYTES);

        const uint64_t waves = CeilDiv(taskCount, coreNum);
        const uint64_t mte3PerTask = CeilDiv(mte3Cost, taskCount);
        uint64_t score = std::max(CeilDiv(SaturatingAdd(mte2Cost, vectorCost), taskCount),
            mte3PerTask);
        score = SaturatingMul(score, waves);
        // 双流水稳态之外，最后一个任务仍需等待一次 MTE3 写回排空。
        score = SaturatingAdd(score, mte3PerTask);
        if (score < bestScore ||
            (score == bestScore && inputCalls < bestInputCalls) ||
            (score == bestScore && inputCalls == bestInputCalls && taskCount < bestTaskCount)) {
            bestScore = score;
            bestInputCalls = inputCalls;
            bestChunkRows = chunkRows;
            bestPanelRows = panelRows;
            bestTileY = tileY;
            bestTaskCount = taskCount;
        }
        }
    }
    if (bestChunkRows == 0 || bestPanelRows == 0 || bestTileY == 0 || bestTaskCount == 0) {
        return false;
    }

    tiling.set_tileRows(bestChunkRows);
    tiling.set_panelRows(bestPanelRows);
    tiling.set_tileCols(bestTileY);
    // 当完整输出 panel 能一次写回，且矩阵批次数足以覆盖全部 Vector 核时，
    // 将同一矩阵的多个相邻列 tile 交给同一个任务顺序处理。输入/输出 UB
    // 缓冲仍复用单 tile 容量，因此不改变 MTE2/MTE3 的地址契约；这里只减少
    // 任务边界、队列生命周期和重复索引计算。批次数过少时保持原始任务粒度，
    // 避免降低核间并行度。
    uint64_t colTilesPerTask = 1;
    const uint64_t colTileCount = CeilDiv(layout.right, bestTileY);
    const uint64_t fullPanelRows = AlignUp(x, TRANS_DATA_ADDRESS_COUNT);
    if (layout.fixed == 1 && layout.batch >= static_cast<uint64_t>(coreNum) * 2 &&
        bestPanelRows == fullPanelRows && colTileCount > 1) {
        const uint64_t groupedTaskCount = fixedBatches;
        if (groupedTaskCount >= coreNum) {
            colTilesPerTask = colTileCount;
            bestTaskCount = groupedTaskCount;
        }
    }
    tiling.set_taskCount(bestTaskCount);
    tiling.set_bufferNum(static_cast<uint32_t>(inputBufferNum));
    tiling.set_colTilesPerTask(static_cast<uint32_t>(colTilesPerTask));
    return true;
}

bool SetGatherRecordTiling(optiling::TransposeTilingData& tiling,
    const BlockSwapLayout& layout, uint64_t left, uint32_t elementSize, uint32_t coreNum, uint64_t usableUb)
{
    if (elementSize != 2 && elementSize != 4) {
        return false;
    }
    uint64_t recordBytes = 0;
    uint64_t fullRightBytes = 0;
    if (!Product(layout.record, elementSize, recordBytes) || recordBytes % BLOCK_BYTES == 0 ||
        !Product(layout.right, recordBytes, fullRightBytes) ||
        fullRightBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint64_t sourceRowBytes = 0;
    const bool supportsTwoDimensionalInput = layout.leftInner == 1 &&
        Product(layout.fixed, fullRightBytes, sourceRowBytes) &&
        sourceRowBytes <= std::numeric_limits<uint32_t>::max();

    const uint64_t gatherMaxElements = elementSize == 2 ?
        GATHER_B16_REPEAT_MAX_ELEMENTS : GATHER_B32_REPEAT_MAX_ELEMENTS;
    if (layout.record > gatherMaxElements) {
        return false;
    }
    const uint64_t maxLeftTile = std::min({left, DATA_COPY_BLOCK_COUNT_MAX,
        gatherMaxElements / layout.record});
    const uint64_t maxRightTile = std::min(layout.right, DATA_COPY_BLOCK_COUNT_MAX);
    if (maxLeftTile == 0 || maxRightTile == 0) {
        return false;
    }

    uint64_t fixedBatches = 0;
    uint64_t validBytes = 0;
    if (!Product(layout.batch, layout.fixed, fixedBatches) ||
        !Product(fixedBatches, left, validBytes) ||
        !Product(validBytes, fullRightBytes, validBytes)) {
        return false;
    }

    uint64_t bestLeftTile = 0;
    uint64_t bestRightTile = 0;
    uint64_t bestPanelCols = 0;
    uint64_t bestTaskCount = 0;
    uint64_t bestInputCalls = std::numeric_limits<uint64_t>::max();
    uint32_t bestInputBufferNum = 0;
    uint64_t bestScore = std::numeric_limits<uint64_t>::max();
    for (uint64_t leftTile = 1; leftTile <= maxLeftTile; ++leftTile) {
        const uint64_t gatherCount = SaturatingMul(leftTile, layout.record);
        const uint64_t gatherRepeatElements = 8 * BLOCK_BYTES / elementSize;
        // 完整行 tile 的尾 repeat 可补齐 offset 后并入主体指令；尾行 tile 仍保持精确 Gather。
        const bool padFullGather = gatherCount > gatherRepeatElements &&
            gatherCount % gatherRepeatElements != 0;
        const uint64_t storedGatherCount = padFullGather ?
            AlignUp(gatherCount, gatherRepeatElements) : gatherCount;
        const uint64_t outputPitchBytes =
            AlignUp(SaturatingMul(storedGatherCount, elementSize), BLOCK_BYTES);
        const uint64_t offsetBytes =
            AlignUp(SaturatingMul(storedGatherCount, sizeof(uint32_t)), BLOCK_BYTES);
        if (gatherCount > gatherMaxElements || outputPitchBytes == std::numeric_limits<uint64_t>::max() ||
            outputPitchBytes > DATA_COPY_BLOCK_LEN_MAX || offsetBytes == std::numeric_limits<uint64_t>::max()) {
            continue;
        }
        const uint64_t leftTileCount = CeilDiv(left, leftTile);
        const uint64_t leftTail = left - (leftTileCount - 1) * leftTile;
        const uint64_t fullGatherInstructions =
            padFullGather ? 1 : GetGatherDispatchCount(gatherCount, elementSize);
        const uint64_t tailGatherInstructions = leftTail == leftTile ?
            fullGatherInstructions :
            GetGatherDispatchCount(SaturatingMul(leftTail, layout.record), elementSize);

        for (uint64_t rightTile = 1; rightTile <= maxRightTile; ++rightTile) {
            const uint64_t outputBufferBytes = SaturatingMul(rightTile, outputPitchBytes);
            const uint64_t outputAndOffsetBytes = SaturatingAdd(outputBufferBytes, offsetBytes);
            if (outputAndOffsetBytes >= usableUb) {
                break;
            }

            const uint64_t rightTileCount = CeilDiv(layout.right, rightTile);
            uint64_t tasksPerFixed = 0;
            uint64_t taskCount = 0;
            if (!Product(leftTileCount, rightTileCount, tasksPerFixed) ||
                !Product(fixedBatches, tasksPerFixed, taskCount) || taskCount == 0) {
                continue;
            }

            const uint64_t outputCalls = taskCount;
            const uint64_t outputBlocks = SaturatingMul(
                SaturatingMul(fixedBatches, leftTileCount), layout.right);

            const uint64_t paddedLeftBytes = GetTiledPaddedBytes(left, leftTile, recordBytes);
            uint64_t mte3Cost = SaturatingMul(
                SaturatingMul(fixedBatches, layout.right), paddedLeftBytes);
            mte3Cost = AddCost(mte3Cost, outputBlocks, OUTPUT_BLOCK_EQUIVALENT_BYTES);
            mte3Cost = AddCost(mte3Cost, outputCalls, DMA_CALL_EQUIVALENT_BYTES);

            uint64_t gatherInstructions = SaturatingMul(leftTileCount - 1, fullGatherInstructions);
            gatherInstructions = SaturatingAdd(gatherInstructions, tailGatherInstructions);
            gatherInstructions = SaturatingMul(
                SaturatingMul(fixedBatches, layout.right), gatherInstructions);
            const uint64_t fullTileCount = leftTileCount -
                static_cast<uint64_t>(leftTail != leftTile);
            uint64_t paddedGatherElements = SaturatingMul(fixedBatches, layout.right);
            paddedGatherElements = SaturatingMul(paddedGatherElements, fullTileCount);
            paddedGatherElements = SaturatingMul(
                paddedGatherElements, storedGatherCount - gatherCount);
            // 合并尾 repeat 会执行补齐 lane，模型必须计入额外 Vector 流量，避免偏爱高 padding tile。
            uint64_t vectorCost = SaturatingAdd(
                validBytes, SaturatingMul(paddedGatherElements, elementSize));
            vectorCost = AddCost(vectorCost, gatherInstructions,
                GATHER_INSTRUCTION_EQUIVALENT_BYTES);

            auto evaluatePanel = [&](uint64_t panelCols, uint32_t inputBufferNum) {
                const uint64_t panelBlockBytes = SaturatingMul(panelCols, recordBytes);
                const uint64_t inputPitchBytes = AlignUp(panelBlockBytes, BLOCK_BYTES);
                if (panelCols == 0 || panelBlockBytes > DATA_COPY_BLOCK_LEN_MAX ||
                    inputPitchBytes == std::numeric_limits<uint64_t>::max()) {
                    return;
                }
                const uint64_t inputBufferBytes = SaturatingMul(leftTile, inputPitchBytes);
                uint64_t usedUb = SaturatingMul(inputBufferBytes, inputBufferNum);
                usedUb = SaturatingAdd(usedUb, outputAndOffsetBytes);
                const uint64_t maxOffset = SaturatingAdd(
                    SaturatingMul(leftTile - 1, inputPitchBytes),
                    SaturatingMul(layout.record - 1, elementSize));
                const uint64_t maxBase = SaturatingMul(panelCols - 1, recordBytes);
                const uint64_t sourceEnd = SaturatingAdd(
                    SaturatingAdd(maxOffset, maxBase), elementSize);
                if (usedUb > usableUb || maxOffset > std::numeric_limits<uint32_t>::max() ||
                    maxBase > std::numeric_limits<uint32_t>::max() || sourceEnd > inputBufferBytes) {
                    return;
                }

                auto getExtentStats = [&](uint64_t extent, uint64_t& panelCount,
                                          uint64_t& paddedBytes) {
                    panelCount = CeilDiv(extent, panelCols);
                    const uint64_t tail = extent - (panelCount - 1) * panelCols;
                    paddedBytes = SaturatingMul(panelCount - 1, inputPitchBytes);
                    paddedBytes = SaturatingAdd(paddedBytes,
                        AlignUp(SaturatingMul(tail, recordBytes), BLOCK_BYTES));
                };
                uint64_t fullPanelCount = 0;
                uint64_t fullPaddedBytes = 0;
                uint64_t tailPanelCount = 0;
                uint64_t tailPaddedBytes = 0;
                getExtentStats(rightTile, fullPanelCount, fullPaddedBytes);
                const uint64_t rightTail = layout.right - (rightTileCount - 1) * rightTile;
                getExtentStats(rightTail, tailPanelCount, tailPaddedBytes);
                uint64_t panelCount = SaturatingMul(rightTileCount - 1, fullPanelCount);
                panelCount = SaturatingAdd(panelCount, tailPanelCount);
                uint64_t paddedRightBytes = SaturatingMul(rightTileCount - 1, fullPaddedBytes);
                paddedRightBytes = SaturatingAdd(paddedRightBytes, tailPaddedBytes);

                uint64_t inputBlocks = SaturatingMul(fixedBatches, left);
                inputBlocks = SaturatingMul(inputBlocks, panelCount);
                uint64_t inputCalls = inputBlocks;
                if (supportsTwoDimensionalInput) {
                    inputCalls = SaturatingMul(fixedBatches, leftTileCount);
                    inputCalls = SaturatingMul(inputCalls, panelCount);
                }
                uint64_t mte2Cost = SaturatingMul(
                    SaturatingMul(fixedBatches, left), paddedRightBytes);
                mte2Cost = AddCost(mte2Cost, inputBlocks, INPUT_BLOCK_EQUIVALENT_BYTES);
                mte2Cost = AddCost(mte2Cost, inputCalls, DMA_CALL_EQUIVALENT_BYTES);

                const uint64_t waves = CeilDiv(taskCount, coreNum);
                uint64_t perTaskCost = 0;
                if (inputBufferNum == 1) {
                    perTaskCost = std::max(CeilDiv(SaturatingAdd(mte2Cost, vectorCost), taskCount),
                        CeilDiv(mte3Cost, taskCount));
                } else {
                    perTaskCost = std::max({CeilDiv(mte2Cost, taskCount),
                        CeilDiv(vectorCost, taskCount), CeilDiv(mte3Cost, taskCount)});
                }
                uint64_t score = SaturatingMul(perTaskCost, waves);
                // offset 表每核仅初始化一次，将标量写表开销计入关键路径。
                score = AddCost(score, storedGatherCount, BLOCK_BYTES);
                if (score < bestScore ||
                    (score == bestScore && inputCalls < bestInputCalls) ||
                    (score == bestScore && inputCalls == bestInputCalls && taskCount < bestTaskCount)) {
                    bestScore = score;
                    bestInputCalls = inputCalls;
                    bestLeftTile = leftTile;
                    bestRightTile = rightTile;
                    bestPanelCols = panelCols;
                    bestTaskCount = taskCount;
                    bestInputBufferNum = inputBufferNum;
                }
            };

            evaluatePanel(rightTile, 1);
            if (rightTile > 1 && outputAndOffsetBytes < usableUb) {
                const uint64_t panelPitchLimit =
                    (usableUb - outputAndOffsetBytes) / (2 * leftTile);
                uint64_t maxPanelCols = std::min(rightTile - 1, panelPitchLimit / recordBytes);
                while (maxPanelCols != 0 &&
                       AlignUp(SaturatingMul(maxPanelCols, recordBytes), BLOCK_BYTES) > panelPitchLimit) {
                    --maxPanelCols;
                }
                if (maxPanelCols != 0) {
                    const uint64_t balancedPanelCols =
                        std::min(maxPanelCols, CeilDiv(rightTile, 2));
                    evaluatePanel(balancedPanelCols, 2);
                    // 相邻的非对称拆分可能减少两段合计的 32B padding，补充一次局部搜索。
                    if (balancedPanelCols < maxPanelCols) {
                        evaluatePanel(balancedPanelCols + 1, 2);
                    }
                    // 对非对齐 record，面板列数可使单行输入 burst 恰好落在 32B 边界。
                    // 这会消除每行 DataCopyPad 的尾部补齐；将其作为通用候选与均分
                    // 面板共同参与代价模型，避免固定偏向更大或更小的面板。
                    const uint64_t panelAlignment = BLOCK_BYTES /
                        GreatestCommonDivisor(recordBytes, BLOCK_BYTES);
                    const uint64_t alignedPanelCols = AlignDown(maxPanelCols, panelAlignment);
                    if (alignedPanelCols != 0 && alignedPanelCols != balancedPanelCols &&
                        alignedPanelCols != balancedPanelCols + 1) {
                        evaluatePanel(alignedPanelCols, 2);
                    }
                }
            }
        }
    }

    if (bestLeftTile == 0 || bestRightTile == 0 || bestPanelCols == 0 ||
        bestTaskCount == 0 || bestInputBufferNum == 0) {
        return false;
    }

    tiling.set_tileRows(bestLeftTile);
    tiling.set_tileCols(bestRightTile);
    tiling.set_panelCols(bestPanelCols);
    tiling.set_taskCount(bestTaskCount);
    tiling.set_bufferNum(bestInputBufferNum);
    tiling.set_microKernel(static_cast<uint32_t>(optiling::TransposeBlockSwapKernel::RECORD_GATHER));
    return true;
}

bool SetRecordBlockSwapTiling(optiling::TransposeTilingData& tiling, const BlockSwapLayout& layout,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    uint64_t left = 0;
    uint64_t outputRecordElements = 0;
    uint64_t outputColumnElements = 0;
    if (!Product(layout.leftOuter, layout.leftInner, left) ||
        !Product(left, layout.record, outputRecordElements) ||
        !Product(layout.fixed, outputRecordElements, outputColumnElements) ||
        !FitsByteStrideU32(outputRecordElements, elementSize) ||
        !FitsByteStrideU32(outputColumnElements, elementSize) ||
        !FitsByteStrideU32(layout.record, elementSize)) {
        return false;
    }
    const uint64_t blockElements = BLOCK_BYTES / elementSize;
    const uint64_t alignedRecord = AlignUp(layout.record, blockElements);
    const bool alignedMove = alignedRecord == layout.record;
    const uint64_t usableUb = GetUsableUb(ubSize);
    if (!alignedMove &&
        SetGatherRecordTiling(tiling, layout, left, elementSize, coreNum, usableUb)) {
        return true;
    }
    if (!alignedMove && elementSize != 2) {
        return false;
    }

    const uint64_t bufferFactor = alignedMove ? 1 : RECORD_PACK_BUFFER_FACTOR;
    const uint64_t repeatLeftMax = alignedMove ? left : VECTOR_REPEAT_MAX * blockElements / alignedRecord;
    if (repeatLeftMax == 0) {
        return false;
    }

    uint64_t bytesPerRight = 0;
    if (!Product(alignedRecord, elementSize * bufferFactor, bytesPerRight) || bytesPerRight == 0) {
        return false;
    }
    uint64_t maxRightTile = std::min(AlignUp(layout.right, TRANS_DATA_ADDRESS_COUNT),
        AlignDown(DATA_COPY_BLOCK_COUNT_MAX, TRANS_DATA_ADDRESS_COUNT));
    maxRightTile = std::min(maxRightTile,
        AlignDown(usableUb / bytesPerRight, TRANS_DATA_ADDRESS_COUNT));
    if (maxRightTile == 0) {
        return false;
    }

    uint64_t bestLeftTile = 0;
    uint64_t bestRightTile = 0;
    uint64_t bestTaskCount = 0;
    uint64_t bestScore = std::numeric_limits<uint64_t>::max();
    // 遍历所有硬件地址组粒度的 N tile，用统一命令成本和 padding 成本选择方案。
    for (uint64_t rightTile = TRANS_DATA_ADDRESS_COUNT; rightTile <= maxRightTile;
         rightTile += TRANS_DATA_ADDRESS_COUNT) {
        uint64_t bytesPerLeft = 0;
        if (!Product(rightTile, bytesPerRight, bytesPerLeft) || bytesPerLeft == 0) {
            continue;
        }
        const uint64_t leftTile = std::min({left, usableUb / bytesPerLeft, repeatLeftMax});
        if (leftTile == 0) {
            continue;
        }
        const uint64_t rightTileCount = CeilDiv(layout.right, rightTile);
        const uint64_t leftTileCount = CeilDiv(left, leftTile);
        uint64_t tasksPerFixed = 0;
        uint64_t tasksPerBatch = 0;
        uint64_t taskCount = 0;
        if (!Product(rightTileCount, leftTileCount, tasksPerFixed) ||
            !Product(layout.fixed, tasksPerFixed, tasksPerBatch) ||
            !Product(layout.batch, tasksPerBatch, taskCount)) {
            continue;
        }

        const uint64_t fixedBatches = SaturatingMul(layout.batch, layout.fixed);
        const uint64_t inputCalls = SaturatingMul(SaturatingMul(fixedBatches, rightTileCount), left);
        const uint64_t outputCalls = taskCount;
        const uint64_t transCalls = alignedMove ? 0 :
            SaturatingMul(SaturatingMul(taskCount, rightTile / TRANS_DATA_ADDRESS_COUNT), 2);
        uint64_t paddedElements = SaturatingMul(rightTileCount, rightTile);
        paddedElements = SaturatingMul(paddedElements, SaturatingMul(leftTileCount, leftTile));
        paddedElements = SaturatingMul(paddedElements, alignedRecord);
        paddedElements = SaturatingMul(paddedElements, fixedBatches);
        uint64_t cost = SaturatingMul(paddedElements, elementSize * bufferFactor);
        cost = AddCost(cost, inputCalls, DMA_CALL_EQUIVALENT_BYTES);
        cost = AddCost(cost, outputCalls, DMA_CALL_EQUIVALENT_BYTES);
        cost = AddCost(cost, transCalls, TRANS_DATA_CALL_EQUIVALENT_BYTES);
        const uint64_t activeCores = std::min<uint64_t>(taskCount, coreNum);
        const uint64_t score = activeCores == 0 ? std::numeric_limits<uint64_t>::max() :
            CeilDiv(cost, activeCores);
        if (score < bestScore || (score == bestScore && taskCount < bestTaskCount)) {
            bestScore = score;
            bestLeftTile = leftTile;
            bestRightTile = rightTile;
            bestTaskCount = taskCount;
        }
    }
    if (bestLeftTile == 0 || bestRightTile == 0 || bestTaskCount == 0) {
        return false;
    }

    tiling.set_tileRows(bestLeftTile);
    tiling.set_tileCols(bestRightTile);
    tiling.set_taskCount(bestTaskCount);
    tiling.set_bufferNum(1);
    tiling.set_microKernel(
        static_cast<uint32_t>(optiling::TransposeBlockSwapKernel::RECORD_TRANS_DATA));
    return true;
}

void SetBlockSwapLayoutFields(optiling::TransposeTilingData& tiling, const BlockSwapLayout& layout)
{
    tiling.set_batch(layout.batch);
    tiling.set_outer(layout.leftOuter);
    tiling.set_rows(layout.fixed);
    tiling.set_cols(layout.leftInner);
    tiling.set_inner(layout.right);
    tiling.set_record(layout.record);
}

bool SetBlockSwapTiling(optiling::TransposeTilingData& tiling, const BlockSwapLayout& layout,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    uint64_t left = 0;
    if (!Product(layout.leftOuter, layout.leftInner, left)) {
        return false;
    }

    bool success = false;
    if (layout.record == 1 && elementSize != 1) {
        success = SetStridedMatrixTiling(tiling, layout, elementSize, coreNum, ubSize);
        if (success) {
            const uint64_t colTileCount = CeilDiv(layout.right, tiling.get_tileCols());
            // 单矩阵欠并行时每个活跃核只承担一个列 tile，避免进入通用任务解码中的
            // 矩阵/列商余分解；数据搬运、UB 布局和连续写回均与 STRIDED_MATRIX 相同。
            const bool singleMatrixFastPath = layout.batch == 1 && layout.fixed == 1 &&
                tiling.get_colTilesPerTask() == 1 && tiling.get_taskCount() == colTileCount &&
                tiling.get_taskCount() <= coreNum;
            tiling.set_microKernel(static_cast<uint32_t>(singleMatrixFastPath ?
                optiling::TransposeBlockSwapKernel::STRIDED_SINGLE_MATRIX :
                optiling::TransposeBlockSwapKernel::STRIDED_MATRIX));
        }
    }
    if (!success && layout.record == 1 && layout.fixed == 1) {
        success = SetBlockTransposeTiling(
            tiling, layout.batch, left, layout.right, elementSize, coreNum, ubSize);
        if (success) {
            tiling.set_microKernel(static_cast<uint32_t>(optiling::TransposeBlockSwapKernel::DENSE_MATRIX));
        }
    } else if (!success && layout.record > 1) {
        success = SetRecordBlockSwapTiling(tiling, layout, elementSize, coreNum, ubSize);
    }
    if (success) {
        SetBlockSwapLayoutFields(tiling, layout);
    }
    return success;
}

bool CanUseSingleMatrixTBufKey(optiling::TransposeTilingData& tiling, uint32_t elementSize)
{
    if (tiling.get_microKernel() != static_cast<uint32_t>(
            optiling::TransposeBlockSwapKernel::STRIDED_SINGLE_MATRIX)) {
        return false;
    }
    uint64_t matrixRows = 0;
    uint64_t stripeElements = 0;
    if (!Product(tiling.get_outer(), tiling.get_cols(), matrixRows) ||
        !Product(matrixRows, tiling.get_tileCols(), stripeElements)) {
        return false;
    }
    const uint64_t alignElements = BLOCK_BYTES / elementSize;
    uint64_t stripeBytes = 0;
    if (alignElements == 0 || matrixRows == 0 || matrixRows % TRANS_DATA_ADDRESS_COUNT != 0 ||
        tiling.get_tileCols() == 0 || tiling.get_tileCols() % alignElements != 0 ||
        tiling.get_inner() % tiling.get_tileCols() != 0 ||
        tiling.get_taskCount() != CeilDiv(tiling.get_inner(), tiling.get_tileCols()) ||
        tiling.get_taskCount() == 0 ||
        !Product(stripeElements, elementSize, stripeBytes) ||
        matrixRows > DATA_COPY_BLOCK_COUNT_MAX ||
        stripeBytes > (tiling.get_ubSize() > SINGLE_TBUF_RESERVE_BYTES ?
            (tiling.get_ubSize() - SINGLE_TBUF_RESERVE_BYTES) / 2 : 0)) {
        return false;
    }
    return tiling.get_tileCols() / alignElements <= VECTOR_REPEAT_MAX;
}

bool CanUsePhaseCompactKey(optiling::TransposeTilingData& tiling, uint32_t elementSize)
{
    // 该路径按 FP32 的 8 元素/32B 相位，把非对齐行首向左扩到块边界，随后在 UB
    // 内压缩回紧凑行。条件仅依赖 dtype、DMA 对齐、tiling 与 UB 资源，不绑定 shape。
    if (elementSize != sizeof(float) || tiling.get_rows() != 1 || tiling.get_inner() == 0 ||
        tiling.get_tileRows() == 0 || tiling.get_tileCols() == 0 || tiling.get_panelRows() == 0) {
        return false;
    }
    const uint64_t blockElements = BLOCK_BYTES / elementSize;
    const uint64_t tileColsAlign = AlignUp(tiling.get_tileCols(), blockElements);
    if (blockElements == 0 || tiling.get_inner() % blockElements != 1 ||
        tiling.get_tileRows() % blockElements != 0 || tileColsAlign == 0 ||
        tileColsAlign % blockElements != 0 ||
        tileColsAlign / blockElements > VECTOR_REPEAT_MAX) {
        return false;
    }
    const uint64_t rawPitch = SaturatingAdd(tileColsAlign, blockElements);
    if (rawPitch == std::numeric_limits<uint64_t>::max() || tiling.get_inner() <= rawPitch) {
        return false;
    }
    // phaseMaskBuffer_ 为每个相位预留 blockElements 个 uint32 mask word；mask 覆盖范围
    // 必须在此容量内，不能仅因当前 C5 的窄 tile 恰好安全就放宽到越界布局。
    const uint64_t maskWords = CeilDiv(rawPitch, sizeof(uint32_t) * 8);
    if (maskWords == 0 || maskWords > blockElements) {
        return false;
    }
    const uint64_t sourceGapBlocks = (tiling.get_inner() - 1 - rawPitch) / blockElements;
    if (tiling.get_tileRows() > STRIDED_MATRIX_MAX_X ||
        rawPitch / blockElements > STRIDED_MATRIX_MAX_X ||
        sourceGapBlocks > STRIDED_MATRIX_MAX_X) {
        return false;
    }

    // 单份扩宽输入 + 单份压缩输入 + 32B 对齐的八相位 mask + 输出 panel。
    const uint64_t rawInputBytes = SaturatingMul(
        SaturatingMul(tiling.get_tileRows(), rawPitch), elementSize);
    const uint64_t compactInputBytes = SaturatingMul(
        SaturatingMul(tiling.get_tileRows(), tileColsAlign), elementSize);
    const uint64_t maskBytes = SaturatingMul(
        SaturatingMul(blockElements, blockElements), sizeof(uint32_t));
    const uint64_t outputBytes = SaturatingMul(
        SaturatingMul(tiling.get_panelRows(), tileColsAlign), elementSize);
    const uint64_t requiredBytes = SaturatingAdd(
        SaturatingAdd(rawInputBytes, compactInputBytes), SaturatingAdd(maskBytes, outputBytes));
    return requiredBytes <= GetUsableUb(tiling.get_ubSize());
}

bool CanUsePhaseMajorCompactKey(optiling::TransposeTilingData& tiling, uint32_t elementSize)
{
    // 单次串流 GatherMask 要为每个 raw 行提供一个完整 32B mask 槽；只有完整的
    // 8 行相位周期才能把源行按相位重排而不改变尾块的既有回退语义。
    if (!CanUsePhaseCompactKey(tiling, elementSize)) {
        return false;
    }
    const uint64_t blockElements = BLOCK_BYTES / elementSize;
    const uint64_t tileColsAlign = AlignUp(tiling.get_tileCols(), blockElements);
    const uint64_t rawPitch = SaturatingAdd(tileColsAlign, blockElements);
    if (blockElements == 0 || rawPitch == std::numeric_limits<uint64_t>::max() ||
        tiling.get_tileRows() % blockElements != 0 ||
        tiling.get_tileRows() > STRIDED_MATRIX_MAX_X ||
        rawPitch > std::numeric_limits<uint16_t>::max() * blockElements) {
        return false;
    }
    const uint64_t sourceSpan = SaturatingMul(tiling.get_inner(), blockElements);
    if (sourceSpan == std::numeric_limits<uint64_t>::max() || sourceSpan < rawPitch ||
        (sourceSpan - rawPitch) % blockElements != 0 ||
        (sourceSpan - rawPitch) / blockElements > STRIDED_MATRIX_MAX_X) {
        return false;
    }

    const uint64_t rawInputBytes = SaturatingMul(
        SaturatingMul(tiling.get_tileRows(), rawPitch), elementSize);
    const uint64_t compactInputBytes = SaturatingMul(
        SaturatingMul(tiling.get_tileRows(), tileColsAlign), elementSize);
    // src1RepeatStride=1 以 32B datablock 递进，因此每个 raw 行占一个 32B 槽位。
    const uint64_t maskBytes = SaturatingMul(
        SaturatingMul(tiling.get_tileRows(), blockElements), sizeof(uint32_t));
    const uint64_t outputBytes = SaturatingMul(
        SaturatingMul(tiling.get_panelRows(), tileColsAlign), elementSize);
    const uint64_t requiredBytes = SaturatingAdd(
        SaturatingAdd(rawInputBytes, compactInputBytes), SaturatingAdd(maskBytes, outputBytes));
    return requiredBytes <= GetUsableUb(tiling.get_ubSize());
}

bool TryUsePhaseMajorTwoSlotKey(optiling::TransposeTilingData& tiling, uint32_t elementSize)
{
    // 双槽只在原 phase-major 数据契约成立时尝试；候选搜索不读取公开 shape、case 或
    // seed，而是按当前 UB、DMA 几何和 panel/chunk 拓扑选择仍能形成流水的最大 chunk。
    if (!CanUsePhaseMajorCompactKey(tiling, elementSize)) {
        return false;
    }
    const uint64_t blockElements = BLOCK_BYTES / elementSize;
    const uint64_t tileColsAlign = AlignUp(tiling.get_tileCols(), blockElements);
    const uint64_t rawPitch = SaturatingAdd(tileColsAlign, blockElements);
    const uint64_t panelRows = tiling.get_panelRows();
    const uint64_t originalChunkRows = tiling.get_tileRows();
    if (blockElements == 0 || rawPitch == std::numeric_limits<uint64_t>::max() ||
        panelRows == 0 || originalChunkRows < blockElements || panelRows <= blockElements) {
        return false;
    }

    const uint64_t maxChunkRows = AlignDown(
        std::min(originalChunkRows, panelRows / 2), blockElements);
    if (maxChunkRows == 0) {
        return false;
    }

    uint64_t selectedChunkRows = 0;
    for (uint64_t chunkRows = maxChunkRows;; chunkRows -= blockElements) {
        const uint64_t rawInputBytes = SaturatingMul(
            SaturatingMul(chunkRows, rawPitch), elementSize);
        const uint64_t compactInputBytes = SaturatingMul(
            SaturatingMul(chunkRows, tileColsAlign), elementSize);
        const uint64_t maskBytes = SaturatingMul(
            SaturatingMul(chunkRows, blockElements), sizeof(uint32_t));
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(panelRows, tileColsAlign), elementSize);
        uint64_t requiredBytes = SaturatingMul(rawInputBytes, 2);
        requiredBytes = SaturatingAdd(requiredBytes, compactInputBytes);
        requiredBytes = SaturatingAdd(requiredBytes, maskBytes);
        requiredBytes = SaturatingAdd(requiredBytes, outputBytes);
        if (requiredBytes <= GetUsableUb(tiling.get_ubSize())) {
            selectedChunkRows = chunkRows;
            break;
        }
        if (chunkRows == blockElements) {
            break;
        }
    }
    if (selectedChunkRows == 0) {
        return false;
    }
    tiling.set_tileRows(selectedChunkRows);
    return true;
}

uint64_t GetContiguousOutputSuffix(const MergedTopology& topology)
{
    if (topology.effectivePerm.empty()) {
        return 1;
    }
    int64_t expected = static_cast<int64_t>(topology.effectiveDims.size()) - 1;
    uint64_t elements = 1;
    for (auto it = topology.effectivePerm.rbegin(); it != topology.effectivePerm.rend(); ++it) {
        if (static_cast<int64_t>(*it) != expected) {
            break;
        }
        uint64_t next = 0;
        if (MulOverflow(elements, topology.effectiveDims[*it], next)) {
            return 1;
        }
        elements = next;
        --expected;
    }
    return elements;
}

bool SetGenericTiling(optiling::TransposeTilingData& tiling, uint64_t totalElements,
    const MergedTopology& topology, uint32_t elementSize, uint64_t ubSize)
{
    const uint64_t inner = GetContiguousOutputSuffix(topology);
    const uint64_t outer = totalElements == 0 ? 1 : totalElements / inner;
    const uint64_t alignElements = BLOCK_BYTES / elementSize;
    const uint64_t usableUb = GetUsableUb(ubSize);
    uint64_t maxTileElements = AlignDown(usableUb / (2 * elementSize), alignElements);
    if (maxTileElements == 0) {
        return false;
    }
    const uint64_t tileCols = std::min(inner, maxTileElements);
    const uint64_t innerTiles = CeilDiv(inner, tileCols);
    uint64_t tileRows = innerTiles == 1 ? std::max<uint64_t>(1, maxTileElements / tileCols) : 1;
    tileRows = std::min(tileRows, outer);
    uint64_t taskCount = 0;
    if (!Product(CeilDiv(outer, tileRows), innerTiles, taskCount)) {
        return false;
    }

    tiling.set_batch(1);
    tiling.set_outer(outer);
    tiling.set_rows(outer);
    tiling.set_cols(inner);
    tiling.set_inner(inner);
    tiling.set_record(inner);
    tiling.set_tileRows(tileRows);
    tiling.set_tileCols(tileCols);
    tiling.set_taskCount(totalElements == 0 ? 1 : taskCount);
    tiling.set_bufferNum(1);
    return true;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& inputShape = context->GetInputShape(0)->GetStorageShape();
    const size_t rankSize = inputShape.GetDimNum();
    if (rankSize > TRANSPOSE_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t rank = static_cast<uint32_t>(rankSize);
    const auto* dimsAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    std::array<uint32_t, TRANSPOSE_MAX_RANK> perm{};
    if (!ParsePermutation(inputShape, dimsAttr, perm)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t elementSizeSigned = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (elementSizeSigned != 1 && elementSizeSigned != 2 && elementSizeSigned != 4) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t elementSize = static_cast<uint32_t>(elementSizeSigned);

    std::array<uint64_t, TRANSPOSE_MAX_RANK> shape{};
    std::array<uint64_t, TRANSPOSE_MAX_RANK> inputStride{};
    std::array<uint64_t, TRANSPOSE_MAX_RANK> outputStride{};
    uint64_t totalElements = 1;
    for (uint32_t axis = rank; axis > 0;) {
        --axis;
        const int64_t dim = inputShape.GetDim(axis);
        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }
        shape[static_cast<size_t>(axis)] = static_cast<uint64_t>(dim);
        inputStride[static_cast<size_t>(axis)] = totalElements;
        if (MulOverflow(totalElements, static_cast<uint64_t>(dim), totalElements)) {
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t outputElements = 1;
    for (uint32_t outputAxis = rank; outputAxis > 0;) {
        --outputAxis;
        outputStride[outputAxis] = outputElements;
        if (MulOverflow(outputElements, shape[perm[outputAxis]], outputElements)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (outputElements != totalElements) {
        return ge::GRAPH_FAILED;
    }

    MergedTopology topology;
    if (!BuildMergedTopology(shape, perm, rank, topology)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || ubSize < BLOCK_BYTES) {
        return ge::GRAPH_FAILED;
    }

    TransposeTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_rank(rank);
    tiling.set_mergedRank(topology.rank);
    tiling.set_elementSize(elementSize);
    tiling.set_ubSize(ubSize);
    tiling.set_shape(shape.data());
    tiling.set_inputStride(inputStride.data());
    tiling.set_outputStride(outputStride.data());
    tiling.set_perm(perm.data());
    tiling.set_panelRows(0);
    tiling.set_panelCols(0);
    tiling.set_colTilesPerTask(1);

    TransposePlanKind planKind = TransposePlanKind::GENERIC_AFFINE;
    uint64_t tilingKey = TRANSPOSE_KEY_GENERIC_AFFINE;
    bool tilingSuccess = false;
    tiling.set_microKernel(0);
    if (totalElements == 0 || topology.isIdentity) {
        planKind = TransposePlanKind::COPY;
        tilingKey = TRANSPOSE_KEY_COPY;
        tilingSuccess = SetCopyTiling(tiling, totalElements, elementSize, coreNum, ubSize);
    } else {
        BlockSwapLayout layout;
        if (BuildBlockSwapLayout(topology, layout)) {
            tilingSuccess = SetBlockSwapTiling(tiling, layout, elementSize, coreNum, ubSize);
        }
        if (tilingSuccess) {
            planKind = TransposePlanKind::BLOCK_SWAP;
            tilingKey = TRANSPOSE_KEY_BLOCK_SWAP;
            if (tiling.get_microKernel() == static_cast<uint32_t>(
                TransposeBlockSwapKernel::STRIDED_SINGLE_MATRIX)) {
                tilingKey = CanUseSingleMatrixTBufKey(tiling, elementSize) ?
                    TRANSPOSE_KEY_STRIDED_SINGLE_TBUF : TRANSPOSE_KEY_STRIDED_SINGLE_MATRIX;
            } else if (tiling.get_microKernel() == static_cast<uint32_t>(
                TransposeBlockSwapKernel::DENSE_MATRIX)) {
                tilingKey = TRANSPOSE_KEY_DENSE_MATRIX;
            } else if (tiling.get_microKernel() == static_cast<uint32_t>(
                TransposeBlockSwapKernel::STRIDED_MATRIX)) {
                tilingKey = tiling.get_rows() != 1 ? TRANSPOSE_KEY_STRIDED_NONFIXED_AXIS :
                    (TryUsePhaseMajorTwoSlotKey(tiling, elementSize) ?
                        TRANSPOSE_KEY_STRIDED_FIXED_PHASE_MAJOR_COMPACT :
                        // 双槽预算不成立时回到单相位压缩路径；不让单槽入口携带
                        // 双槽模板实例，从而避免无关 dtype 的设备代码膨胀。
                        (CanUsePhaseCompactKey(tiling, elementSize) ?
                            TRANSPOSE_KEY_STRIDED_FIXED_PHASE_COMPACT :
                            TRANSPOSE_KEY_STRIDED_FIXED_COMPACT));
            }
        }
    }
    if (!tilingSuccess) {
        planKind = TransposePlanKind::GENERIC_AFFINE;
        tilingKey = TRANSPOSE_KEY_GENERIC_AFFINE;
        tilingSuccess = SetGenericTiling(tiling, totalElements, topology, elementSize, ubSize);
    }
    if (!tilingSuccess || tiling.get_taskCount() == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t launchTaskCount = std::min<uint64_t>(coreNum, tiling.get_taskCount());
    // 16-bit 单矩阵 TBuf 条带每个任务都包含完整二维搬入、VA 转置与连续写回。对欠并行且
    // 至少有 16 条带的场景，让一个 AIV 顺序复用同一对 TBuf 处理两条带，降低短核
    // 的启动与跨核资源开销；其它 dtype 和 tiling key 保持一任务一核的既有策略。
    if (tilingKey == TRANSPOSE_KEY_STRIDED_SINGLE_TBUF &&
        elementSize == 2 && tiling.get_taskCount() >= TRANS_DATA_ADDRESS_COUNT &&
        tiling.get_taskCount() <= coreNum) {
        launchTaskCount = CeilDiv(tiling.get_taskCount(), static_cast<uint64_t>(2));
    }
    const uint32_t blockDim = static_cast<uint32_t>(launchTaskCount);
    tiling.set_planKind(static_cast<uint32_t>(planKind));
    tiling.set_blockDim(blockDim);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKey));

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    auto* rawTiling = context->GetRawTilingData();
    if (workspaceSize == nullptr || rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    workspaceSize[0] = 0;
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    std::array<uint32_t, optiling::TRANSPOSE_MAX_RANK> perm{};
    const auto* dimsAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    if (!ParsePermutation(*inputShape, dimsAttr, perm)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rank = static_cast<uint32_t>(inputShape->GetDimNum());
    outputShape->SetDimNum(0);
    for (uint32_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
        outputShape->AppendDim(inputShape->GetDim(perm[outputAxis]));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dims").ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
}
