
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>


namespace optiling {
namespace {
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t VECTOR_BYTES = 256;
// resident/axis 路径保留四个 int32 index DMA 块的最小工作集门槛。
constexpr uint64_t INT32_INDEX_ELEMENTS_PER_BLOCK = BLOCK_BYTES / sizeof(int32_t);
constexpr uint64_t MIN_RESIDENT_INDEX_BLOCKS = 4;
constexpr uint64_t MIN_RESIDENT_INDEX_ELEMENTS =
    INT32_INDEX_ELEMENTS_PER_BLOCK * MIN_RESIDENT_INDEX_BLOCKS;
constexpr uint64_t MIN_INT8_INNER_ELEMENTS = BLOCK_BYTES / sizeof(int8_t);
constexpr uint64_t MIN_AXIS_TARGET_ROWS = 2;
constexpr uint64_t TARGET_TASK_BYTES = 512 * 1024;
constexpr uint64_t UB_USE_NUMERATOR = 9;
constexpr uint64_t UB_USE_DENOMINATOR = 10;
constexpr uint64_t GATHER_UB_RESERVE_BYTES = 4 * 1024;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
constexpr uint64_t COOP_RAW_SLOT_BYTES = 32;
constexpr uint64_t COOP_LAYER_CAP_FACTOR = 4;
constexpr uint32_t MTE_BLOCK_COUNT_LIMIT = 4095;
constexpr uint32_t GATHER_FP16_MAX_COUNT = 32767;
constexpr uint32_t GATHER_FP32_MAX_COUNT = 16383;
constexpr uint32_t STRATEGY_ATOMIC = 0;
constexpr uint32_t STRATEGY_UB = 1;
constexpr uint32_t STRATEGY_NARROW_UB = 2;
constexpr uint32_t STRATEGY_GATHER_NARROW = 3;
constexpr uint32_t STRATEGY_GATHER_BATCH = 4;
constexpr uint32_t STRATEGY_BF16_RESIDENT = 5;
constexpr uint32_t STRATEGY_INT8_RESIDENT = 6;
constexpr uint32_t STRATEGY_INT8_AXIS = 7;
constexpr uint32_t STRATEGY_BF16_SOURCE_DB = 8;
constexpr uint32_t STRATEGY_GATHER_NARROW_DB = 9;
constexpr uint32_t STRATEGY_UB_MIXED = 10;
constexpr uint32_t STRATEGY_UB_MIXED_DB = 12;
// BF16 独立输出行窗口最多并行保留四条 FP32 累加工作行。
constexpr uint32_t BF16_INDEPENDENT_WORK_ROWS = 4;
// 两个 Gather repeat 合并为一个连续的 256-lane payload 行。该策略仅在 UB
// 预算、repeat 上限和通用指令数模型都成立时由 host 选择。
constexpr uint32_t STRATEGY_GATHER_BATCH_WIDE = 11;
constexpr uint64_t COOP_TILING_KEY = 1;
constexpr uint64_t SOURCE_L2_BYPASS_TILING_KEY = 2;
constexpr uint64_t MAX_SEARCH_TASKS_PER_CORE = 4;
constexpr uint64_t GATHER_PIPELINE_SOURCE_SKEW_BYTES = 64;
constexpr uint64_t GATHER_PIPELINE_DENSE_SKEW_BYTES = 128;
// FP32 的一条 Vector 指令覆盖 8 个 32B block；额外错开一个完整 Vector 宽度，
// 可将累加器与 dense 操作数映射到互补的 DAV_2201 bank group 半区。
constexpr uint64_t GATHER_PIPELINE_FP32_DENSE_SKEW_BYTES = VECTOR_BYTES;
constexpr uint64_t GATHER_PIPELINE_TAIL_BYTES = 256;
// FP32 转置块路径以 16 行作为 TransDataTo5HD 的物理矩阵宽度；一批同时承载
// 16 条有效 outer 行，Gatherb 分别消费低、高两个 8 行半区。尾批仍用零填充，
// 只将真实 outer 行写回。
constexpr uint64_t FP32_TRANSPOSE_ROWS = 16;
constexpr uint64_t FP32_TRANSPOSE_ACTIVE_ROWS = FP32_TRANSPOSE_ROWS;
constexpr uint64_t FP32_TRANSPOSE_COL_ALIGNMENT = BLOCK_BYTES / sizeof(float);
// FP32 转置后的低、高 half 分别存为独立的 8-float record，Gatherb 的地址
// 步长因此为一个 32B datablock，而不是把两个 half 合并成 64B record。
constexpr uint64_t FP32_TRANSPOSE_OFFSET_SCALE_BYTES =
    (FP32_TRANSPOSE_ROWS / 2) * sizeof(float);
// 一个 FP32 Vector 指令覆盖 8 个 32B datablock。局部表以这一硬件粒度为上限，
// 将其余 UB 让给更大的连续 source chunk；高冲突分布仍通过 header 哨兵保序回退。
constexpr uint64_t FP32_TRANSPOSE_LAYER_CAP = VECTOR_BYTES / BLOCK_BYTES;
// TransDataTo5HD 的 repeat 参数为 uint8_t。source chunk 以一个 32B FP32
// datablock（8 列）为单位增长，最大值由该硬件上限而非经验常数限定。
constexpr uint64_t FP32_TRANSPOSE_MAX_REPEAT = 255;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    // 所有运行时尺寸均来自 shape/tiling；对异常的零除数做保守处理，
    // 避免非法输入在 Host tiling 阶段触发未定义行为。
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

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

bool AddOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return true;
    }
    result = lhs + rhs;
    return false;
}

// resident 路径将 source 的每个逻辑元素顺序消费一次；超过实际 L2 容量时，
// 只绕过这条无复用流，避免其替换 self、index 等后续仍可能复用的数据。
bool ShouldBypassResidentSourceL2(uint32_t strategy, uint64_t outerSize, uint64_t indexSize,
    uint64_t innerSize, uint32_t elementSize, uint64_t l2Size)
{
    if ((strategy != STRATEGY_UB && strategy != STRATEGY_UB_MIXED && strategy != STRATEGY_UB_MIXED_DB) ||
        l2Size == 0 ||
        elementSize == 0) {
        return false;
    }

    uint64_t outerIndexElements = 0;
    uint64_t sourceElements = 0;
    uint64_t sourceBytes = 0;
    if (MulOverflow(outerSize, indexSize, outerIndexElements) ||
        MulOverflow(outerIndexElements, innerSize, sourceElements) ||
        MulOverflow(sourceElements, elementSize, sourceBytes)) {
        return false;
    }
    return sourceBytes > l2Size;
}

bool AppendAlignedRegion(uint64_t& offset, uint64_t bytes)
{
    if (offset > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    offset = AlignUp(offset, WORKSPACE_ALIGNMENT);
    return !AddOverflow(offset, bytes, offset);
}

// cooperative 窄 Gather 双槽 payload 的精确 UB 模型。host 与 kernel 共用同一对齐、
// skew 和尾部保留公式，避免选择到无法初始化的队列布局。
bool GetGatherDoubleBufferBytes(uint64_t indexBytes, uint64_t sourceRowBytes,
    uint64_t accumulatorRowBytes, uint64_t counterRowBytes, uint64_t compactBytes,
    uint64_t denseSkewBytes, uint64_t& slotBytes, uint64_t& usedBytes)
{
    uint64_t sourceBufferBytes = 0;
    uint64_t sourceOffsetBytes = 0;
    uint64_t sourceEndBytes = 0;
    uint64_t denseOffsetBytes = 0;
    uint64_t denseEndBytes = 0;
    uint64_t fixedBytes = 0;
    uint64_t doubleSlotBytes = 0;
    const bool overflow =
        AddOverflow(sourceRowBytes, BLOCK_BYTES, sourceBufferBytes) ||
        accumulatorRowBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1) ||
        AddOverflow(AlignUp(accumulatorRowBytes, WORKSPACE_ALIGNMENT),
            GATHER_PIPELINE_SOURCE_SKEW_BYTES, sourceOffsetBytes) ||
        AddOverflow(sourceOffsetBytes, sourceBufferBytes, sourceEndBytes) ||
        sourceEndBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1) ||
        AddOverflow(AlignUp(sourceEndBytes, WORKSPACE_ALIGNMENT), denseSkewBytes, denseOffsetBytes) ||
        AddOverflow(denseOffsetBytes, accumulatorRowBytes, denseEndBytes) ||
        denseEndBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1) ||
        AddOverflow(AlignUp(denseEndBytes, WORKSPACE_ALIGNMENT), GATHER_PIPELINE_TAIL_BYTES, slotBytes) ||
        AddOverflow(indexBytes, accumulatorRowBytes, fixedBytes) ||
        AddOverflow(fixedBytes, counterRowBytes, fixedBytes) ||
        AddOverflow(fixedBytes, compactBytes, fixedBytes) ||
        AddOverflow(fixedBytes, accumulatorRowBytes, fixedBytes) ||
        MulOverflow(2, slotBytes, doubleSlotBytes) ||
        AddOverflow(fixedBytes, doubleSlotBytes, usedBytes);
    return !overflow;
}

bool GetElementSize(ge::DataType dataType, uint32_t& elementSize)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            elementSize = 4;
            return true;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            elementSize = 2;
            return true;
        case ge::DT_INT8:
            elementSize = 1;
            return true;
        default:
            return false;
    }
}

bool GetLogicalShape(gert::TilingContext* context, uint64_t& outerSize, uint64_t& axisSize,
    uint64_t& innerSize, uint64_t& indexSize, ge::DataType& dataType, uint32_t& elementSize)
{
    const gert::StorageShape* selfStorageShape = context->GetInputShape(0);
    const gert::StorageShape* indexStorageShape = context->GetInputShape(1);
    const gert::StorageShape* sourceStorageShape = context->GetInputShape(2);
    const gert::CompileTimeTensorDesc* selfDesc = context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc* indexDesc = context->GetInputDesc(1);
    const gert::CompileTimeTensorDesc* sourceDesc = context->GetInputDesc(2);
    if (selfStorageShape == nullptr || indexStorageShape == nullptr || sourceStorageShape == nullptr ||
        selfDesc == nullptr || indexDesc == nullptr || sourceDesc == nullptr) {
        return false;
    }

    const gert::Shape& selfShape = selfStorageShape->GetStorageShape();
    const gert::Shape& indexShape = indexStorageShape->GetStorageShape();
    const gert::Shape& sourceShape = sourceStorageShape->GetStorageShape();
    const size_t rank = selfShape.GetDimNum();
    if (rank == 0 || indexShape.GetDimNum() != 1 || sourceShape.GetDimNum() != rank) {
        return false;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* dimAttr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(0);
    if (dimAttr == nullptr) {
        return false;
    }
    int64_t dim = *dimAttr;
    const int64_t signedRank = static_cast<int64_t>(rank);
    if (dim < 0) {
        dim += signedRank;
    }
    if (dim < 0 || dim >= signedRank) {
        return false;
    }

    dataType = selfDesc->GetDataType();
    if (!GetElementSize(dataType, elementSize) || sourceDesc->GetDataType() != dataType ||
        indexDesc->GetDataType() != ge::DT_INT32) {
        return false;
    }

    const int64_t signedIndexSize = indexShape.GetDim(0);
    if (signedIndexSize <= 0) {
        return false;
    }
    indexSize = static_cast<uint64_t>(signedIndexSize);
    outerSize = 1;
    innerSize = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t selfDim = selfShape.GetDim(i);
        const int64_t sourceDim = sourceShape.GetDim(i);
        if (selfDim <= 0 || sourceDim <= 0) {
            return false;
        }
        if (static_cast<int64_t>(i) == dim) {
            axisSize = static_cast<uint64_t>(selfDim);
            if (static_cast<uint64_t>(sourceDim) != indexSize) {
                return false;
            }
            continue;
        }
        if (sourceDim != selfDim) {
            return false;
        }
        uint64_t nextSize = 0;
        if (static_cast<int64_t>(i) < dim) {
            if (MulOverflow(outerSize, static_cast<uint64_t>(selfDim), nextSize)) {
                return false;
            }
            outerSize = nextSize;
        } else {
            if (MulOverflow(innerSize, static_cast<uint64_t>(selfDim), nextSize)) {
                return false;
            }
            innerSize = nextSize;
        }
    }
    return true;
}

bool IsScalarInnerTopology(uint64_t innerSize, uint64_t innerBytes, uint32_t elementSize)
{
    // Gather 窄路径处理的是每个 outer 只有一个连续元素的语义拓扑；将条件
    // 集中在这里，避免把轴或形状细节误当成调度依据。
    return elementSize != 0 && innerSize == 1 && innerBytes == elementSize;
}

void SetCoreTiling(IndexAddTilingData& tiling, uint64_t innerTile, uint64_t outerTile,
    uint64_t outerBatch, uint32_t strategy, uint64_t innerTileNum, uint64_t outerTileNum,
    uint64_t taskNum, uint64_t sourceChunk)
{
    tiling.set_innerTile(innerTile);
    tiling.set_outerTile(outerTile);
    tiling.set_outerBatch(outerBatch);
    tiling.set_strategy(strategy);
    tiling.set_innerTileNum(static_cast<uint32_t>(innerTileNum));
    tiling.set_outerTileNum(static_cast<uint32_t>(outerTileNum));
    tiling.set_taskNum(static_cast<uint32_t>(taskNum));
    tiling.set_sourceChunk(static_cast<uint32_t>(sourceChunk));
}

bool SetAtomicTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t innerSize, uint64_t indexSize,
    uint32_t elementSize, uint32_t coreNum, uint64_t usableUb)
{
    const uint64_t alignElements = BLOCK_BYTES / elementSize;
    const uint64_t targetOuterTileNum = std::min<uint64_t>(outerSize, coreNum);
    const uint64_t outerTile = CeilDiv(outerSize, targetOuterTileNum);
    const uint64_t outerTileNum = CeilDiv(outerSize, outerTile);

    uint64_t bytesPerInnerElement = 0;
    if (MulOverflow(outerTile, indexSize, bytesPerInnerElement) ||
        MulOverflow(bytesPerInnerElement, elementSize, bytesPerInnerElement)) {
        return false;
    }
    uint64_t targetInner = TARGET_TASK_BYTES / bytesPerInnerElement;
    targetInner = std::max<uint64_t>(targetInner, alignElements);
    targetInner = targetInner / alignElements * alignElements;
    uint64_t indexBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) || indexBytes > usableUb) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);

    if (indexBytes >= usableUb) {
        return false;
    }

    const uint64_t maxTileBytes = (usableUb - indexBytes) / BLOCK_BYTES * BLOCK_BYTES;
    const uint64_t maxInnerTile = maxTileBytes / elementSize;
    uint64_t innerTile = std::min<uint64_t>({innerSize, targetInner, maxInnerTile});
    if (innerTile != innerSize) {
        innerTile = innerTile / alignElements * alignElements;
    }
    if (innerTile == 0) {
        return false;
    }
    const uint64_t alignedTileBytes = AlignUp(innerTile * elementSize, BLOCK_BYTES);
    const uint64_t sourceChunk =
        std::min<uint64_t>((usableUb - indexBytes) / alignedTileBytes, MTE_BLOCK_COUNT_LIMIT);
    const uint64_t outerBatch = std::min<uint64_t>({outerTile, sourceChunk, MTE_BLOCK_COUNT_LIMIT});
    const uint64_t innerTileNum = CeilDiv(innerSize, innerTile);
    if (sourceChunk == 0 || outerBatch == 0 || outerTileNum > std::numeric_limits<uint32_t>::max() ||
        innerTileNum > std::numeric_limits<uint32_t>::max() ||
        outerTileNum * innerTileNum > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    SetCoreTiling(tiling, innerTile, outerTile, outerBatch, STRATEGY_ATOMIC,
        innerTileNum, outerTileNum, outerTileNum * innerTileNum, sourceChunk);
    return true;
}

bool SetUbTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize, uint64_t innerSize,
    uint64_t indexSize, uint32_t elementSize, uint32_t coreNum, uint64_t usableUb)
{
    uint64_t indexBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes)) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    if (indexBytes >= usableUb || axisSize == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    // 至少预留一个 source panel，剩余空间用于 axis 个累加器行。
    const uint64_t maxTileBytes = (usableUb - indexBytes) / (axisSize + 1) / BLOCK_BYTES * BLOCK_BYTES;
    const uint64_t vectorElements = VECTOR_BYTES / elementSize;
    if (maxTileBytes < VECTOR_BYTES) {
        return false;
    }

    const uint64_t requiredInnerTasks = std::max<uint64_t>(1, CeilDiv(coreNum, outerSize));
    uint64_t parallelTile = innerSize;
    if (requiredInnerTasks > 1) {
        parallelTile = CeilDiv(innerSize, requiredInnerTasks) / vectorElements * vectorElements;
        parallelTile = std::max(parallelTile, vectorElements);
    }
    const uint64_t maxInnerTile = maxTileBytes / elementSize;
    uint64_t innerTile = std::min(parallelTile, maxInnerTile);
    if (innerTile != innerSize) {
        innerTile = innerTile / vectorElements * vectorElements;
    }
    if (innerTile == 0) {
        return false;
    }

    uint64_t alignedTileBytes = AlignUp(innerTile * elementSize, BLOCK_BYTES);
    uint64_t accumulatorBytes = 0;
    if (MulOverflow(axisSize, alignedTileBytes, accumulatorBytes) ||
        accumulatorBytes > usableUb - indexBytes || alignedTileBytes > usableUb - indexBytes - accumulatorBytes) {
        return false;
    }
    uint64_t sourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
        (usableUb - indexBytes - accumulatorBytes) / alignedTileBytes});
    uint64_t innerTileNum = CeilDiv(innerSize, innerTile);
    uint64_t taskNum = 0;
    if (MulOverflow(outerSize, innerTileNum, taskNum)) {
        return false;
    }

    // 在不增加 tile 数的前提下搜索均衡切分。搜索规模受核数约束，极大 shape 直接保留基线。
    uint64_t searchTaskLimit = 0;
    if (!MulOverflow(coreNum, coreNum, searchTaskLimit) &&
        !MulOverflow(searchTaskLimit, MAX_SEARCH_TASKS_PER_CORE, searchTaskLimit) &&
        taskNum <= searchTaskLimit && innerTileNum <= coreNum) {
        uint64_t bestTile = innerTile;
        uint64_t bestTileNum = innerTileNum;
        uint64_t bestRowBytes = alignedTileBytes;
        uint64_t bestSourceChunk = sourceChunk;
        uint64_t bestTaskNum = taskNum;
        uint64_t bestMaxCoreBytes = std::numeric_limits<uint64_t>::max();
        uint64_t bestPanelCalls = std::numeric_limits<uint64_t>::max();
        uint64_t bestVectorPenalty = std::numeric_limits<uint64_t>::max();
        const uint64_t alignElements = BLOCK_BYTES / elementSize;
        for (uint64_t tileCount = 1; tileCount <= innerTileNum; ++tileCount) {
            uint64_t candidate = AlignUp(CeilDiv(innerSize, tileCount), alignElements);
            if (candidate > maxInnerTile) {
                continue;
            }
            const uint64_t candidateTileNum = CeilDiv(innerSize, candidate);
            if (candidateTileNum == 0 || candidateTileNum > innerTileNum) {
                continue;
            }
            const uint64_t rowBytes = AlignUp(candidate * elementSize, BLOCK_BYTES);
            uint64_t candidateAccumulatorBytes = 0;
            if (MulOverflow(axisSize, rowBytes, candidateAccumulatorBytes) ||
                candidateAccumulatorBytes > usableUb - indexBytes ||
                rowBytes > usableUb - indexBytes - candidateAccumulatorBytes) {
                continue;
            }
            const uint64_t candidateSourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
                (usableUb - indexBytes - candidateAccumulatorBytes) / rowBytes});
            uint64_t candidateTaskNum = 0;
            if (candidateSourceChunk == 0 || MulOverflow(outerSize, candidateTileNum, candidateTaskNum) ||
                candidateTaskNum == 0) {
                continue;
            }
            const uint64_t blockDim = std::min<uint64_t>(coreNum, candidateTaskNum);
            uint64_t maxCoreBytes = 0;
            for (uint64_t core = 0; core < blockDim; ++core) {
                uint64_t coreBytes = 0;
                for (uint64_t task = core; task < candidateTaskNum; task += blockDim) {
                    const uint64_t innerGroup = task % candidateTileNum;
                    const uint64_t innerStart = innerGroup * candidate;
                    const uint64_t innerCount = std::min(candidate, innerSize - innerStart);
                    uint64_t taskBytes = 0;
                    if (MulOverflow(innerCount, elementSize, taskBytes) ||
                        AddOverflow(coreBytes, AlignUp(taskBytes, BLOCK_BYTES), coreBytes)) {
                        coreBytes = std::numeric_limits<uint64_t>::max();
                        break;
                    }
                }
                maxCoreBytes = std::max(maxCoreBytes, coreBytes);
            }
            uint64_t panelCalls = 0;
            if (MulOverflow(candidateTaskNum, CeilDiv(indexSize, candidateSourceChunk), panelCalls)) {
                panelCalls = std::numeric_limits<uint64_t>::max();
            }
            const uint64_t vectorPenalty = rowBytes % VECTOR_BYTES != 0;
            const bool better = maxCoreBytes < bestMaxCoreBytes ||
                (maxCoreBytes == bestMaxCoreBytes && candidateTaskNum < bestTaskNum) ||
                (maxCoreBytes == bestMaxCoreBytes && candidateTaskNum == bestTaskNum &&
                    panelCalls < bestPanelCalls) ||
                (maxCoreBytes == bestMaxCoreBytes && candidateTaskNum == bestTaskNum &&
                    panelCalls == bestPanelCalls && vectorPenalty < bestVectorPenalty) ||
                (maxCoreBytes == bestMaxCoreBytes && candidateTaskNum == bestTaskNum &&
                    panelCalls == bestPanelCalls && vectorPenalty == bestVectorPenalty &&
                    candidateSourceChunk > bestSourceChunk);
            if (better) {
                bestTile = candidate;
                bestTileNum = candidateTileNum;
                bestRowBytes = rowBytes;
                bestSourceChunk = candidateSourceChunk;
                bestTaskNum = candidateTaskNum;
                bestMaxCoreBytes = maxCoreBytes;
                bestPanelCalls = panelCalls;
                bestVectorPenalty = vectorPenalty;
            }
        }
        innerTile = bestTile;
        innerTileNum = bestTileNum;
        alignedTileBytes = bestRowBytes;
        sourceChunk = bestSourceChunk;
        taskNum = bestTaskNum;
    }

    if (sourceChunk == 0 || outerSize > std::numeric_limits<uint32_t>::max() ||
        innerTileNum > std::numeric_limits<uint32_t>::max() ||
        taskNum > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    SetCoreTiling(tiling, innerTile, 1, 1, STRATEGY_UB,
        innerTileNum, outerSize, taskNum, sourceChunk);
    return true;
}

// 当等宽 resident task 的最后一个 wave 未填满时，使用相邻的两种内层切分填满 wave。
// 混合 task 仍分别处理互不重叠的 inner 区域，且每个元素内的 index 顺序保持不变。
bool EnableMixedResidentTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t innerSize, uint64_t indexSize, uint32_t elementSize, uint32_t coreNum, uint64_t usableUb)
{
    if (tiling.get_strategy() != STRATEGY_UB || outerSize == 0 || axisSize == 0 || innerSize == 0 ||
        indexSize == 0 || elementSize == 0 || coreNum == 0 || tiling.get_innerTile() == 0 ||
        tiling.get_innerTileNum() < 2) {
        return false;
    }

    const uint64_t baseTaskNum = tiling.get_taskNum();
    const uint64_t baseTileNum = tiling.get_innerTileNum();
    const uint64_t waveNum = CeilDiv(baseTaskNum, coreNum);
    uint64_t targetTaskNum = 0;
    if (waveNum != 2 || MulOverflow(waveNum, coreNum, targetTaskNum) || targetTaskNum <= baseTaskNum) {
        return false;
    }
    const uint64_t mixedOuterCount = targetTaskNum - baseTaskNum;
    if (mixedOuterCount == 0 || mixedOuterCount >= outerSize ||
        baseTileNum == std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const uint64_t mixedTileNum = baseTileNum + 1;
    const uint64_t vectorElements = VECTOR_BYTES / elementSize;
    if (vectorElements == 0) {
        return false;
    }
    const uint64_t mixedInnerTile = AlignUp(CeilDiv(innerSize, mixedTileNum), vectorElements);
    if (mixedInnerTile == 0 || mixedInnerTile >= tiling.get_innerTile()) {
        return false;
    }

    const uint64_t baseOuterCount = outerSize - mixedOuterCount;
    uint64_t baseTaskCount = 0;
    uint64_t mixedTaskCount = 0;
    uint64_t actualTaskNum = 0;
    if (MulOverflow(baseOuterCount, baseTileNum, baseTaskCount) ||
        MulOverflow(mixedOuterCount, mixedTileNum, mixedTaskCount) ||
        AddOverflow(baseTaskCount, mixedTaskCount, actualTaskNum) ||
        baseTaskCount > coreNum || actualTaskNum != targetTaskNum) {
        return false;
    }

    uint64_t indexBytes = 0;
    uint64_t baseRowBytes = 0;
    uint64_t mixedRowBytes = 0;
    uint64_t baseAccumulatorBytes = 0;
    uint64_t mixedAccumulatorBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) ||
        MulOverflow(tiling.get_innerTile(), elementSize, baseRowBytes) ||
        MulOverflow(mixedInnerTile, elementSize, mixedRowBytes)) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    baseRowBytes = AlignUp(baseRowBytes, BLOCK_BYTES);
    mixedRowBytes = AlignUp(mixedRowBytes, BLOCK_BYTES);
    if (indexBytes >= usableUb || baseRowBytes == 0 || mixedRowBytes == 0 ||
        MulOverflow(axisSize, baseRowBytes, baseAccumulatorBytes) ||
        MulOverflow(axisSize, mixedRowBytes, mixedAccumulatorBytes) ||
        baseAccumulatorBytes > usableUb - indexBytes || mixedAccumulatorBytes > usableUb - indexBytes) {
        return false;
    }

    const uint64_t baseSourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
        (usableUb - indexBytes - baseAccumulatorBytes) / baseRowBytes});
    const uint64_t mixedSourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
        (usableUb - indexBytes - mixedAccumulatorBytes) / mixedRowBytes});
    if (baseSourceChunk == 0 || mixedSourceChunk == 0 ||
        CeilDiv(indexSize, mixedSourceChunk) >= CeilDiv(indexSize, baseSourceChunk)) {
        return false;
    }

    uint64_t basePayloadBytes = 0;
    uint64_t mixedPayloadBytes = 0;
    if (MulOverflow(baseSourceChunk, baseRowBytes, basePayloadBytes) ||
        AddOverflow(baseAccumulatorBytes, basePayloadBytes, basePayloadBytes) ||
        MulOverflow(mixedSourceChunk, mixedRowBytes, mixedPayloadBytes) ||
        AddOverflow(mixedAccumulatorBytes, mixedPayloadBytes, mixedPayloadBytes)) {
        return false;
    }
    const uint64_t residentBufferBytes = std::max(basePayloadBytes, mixedPayloadBytes);
    if (residentBufferBytes > usableUb - indexBytes ||
        mixedInnerTile > std::numeric_limits<uint32_t>::max() ||
        mixedTileNum > std::numeric_limits<uint32_t>::max() ||
        mixedOuterCount > std::numeric_limits<uint32_t>::max() ||
        residentBufferBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    tiling.set_strategy(STRATEGY_UB_MIXED);
    tiling.set_taskNum(static_cast<uint32_t>(targetTaskNum));
    tiling.set_sourceChunk(static_cast<uint32_t>(baseSourceChunk));
    tiling.set_mixedInnerTile(static_cast<uint32_t>(mixedInnerTile));
    tiling.set_mixedInnerTileNum(static_cast<uint32_t>(mixedTileNum));
    tiling.set_mixedOuterCount(static_cast<uint32_t>(mixedOuterCount));
    tiling.set_residentBufferBytes(static_cast<uint32_t>(residentBufferBytes));
    return true;
}

// 在 resident mixed 的 UB 预算内为 source panel 留出两个轮换槽位。该候选只改变
// source 搬入与 Vector 累加的核内时序，不改变任务切分、index 顺序或输出布局；若
// 两槽无法容纳完整的对齐 panel，则保留已验收的单槽路径。
bool EnableMixedResidentDoubleBufferTiling(IndexAddTilingData& tiling, uint64_t axisSize,
    uint64_t indexSize, uint32_t elementSize, uint64_t residentUb)
{
    if (tiling.get_strategy() != STRATEGY_UB_MIXED || elementSize != sizeof(int32_t) ||
        axisSize == 0 || indexSize == 0 || tiling.get_mixedInnerTile() == 0 ||
        tiling.get_mixedInnerTileNum() < 2 || residentUb == 0) {
        return false;
    }

    uint64_t indexBytes = 0;
    uint64_t baseRowBytes = 0;
    uint64_t mixedRowBytes = 0;
    uint64_t baseAccumulatorBytes = 0;
    uint64_t mixedAccumulatorBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) ||
        MulOverflow(tiling.get_innerTile(), elementSize, baseRowBytes) ||
        MulOverflow(tiling.get_mixedInnerTile(), elementSize, mixedRowBytes)) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    baseRowBytes = AlignUp(baseRowBytes, BLOCK_BYTES);
    mixedRowBytes = AlignUp(mixedRowBytes, BLOCK_BYTES);
    if (indexBytes >= residentUb || baseRowBytes == 0 || mixedRowBytes == 0 ||
        MulOverflow(axisSize, baseRowBytes, baseAccumulatorBytes) ||
        MulOverflow(axisSize, mixedRowBytes, mixedAccumulatorBytes) ||
        baseAccumulatorBytes >= residentUb - indexBytes ||
        mixedAccumulatorBytes >= residentUb - indexBytes) {
        return false;
    }

    const uint64_t available = residentUb - indexBytes;
    const uint64_t baseMaxChunk = (available - baseAccumulatorBytes) / (2 * baseRowBytes);
    const uint64_t mixedMaxChunk = (available - mixedAccumulatorBytes) / (2 * mixedRowBytes);
    const uint64_t baseChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT, baseMaxChunk});
    const uint64_t mixedChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT, mixedMaxChunk});
    if (baseChunk == 0 || mixedChunk == 0) {
        return false;
    }

    uint64_t baseBytes = 0;
    uint64_t mixedBytes = 0;
    uint64_t residentBytes = 0;
    if (MulOverflow(baseChunk, baseRowBytes, baseBytes) ||
        MulOverflow(mixedChunk, mixedRowBytes, mixedBytes) ||
        MulOverflow(baseBytes, 2, baseBytes) || MulOverflow(mixedBytes, 2, mixedBytes) ||
        AddOverflow(baseAccumulatorBytes, baseBytes, baseBytes) ||
        AddOverflow(mixedAccumulatorBytes, mixedBytes, mixedBytes)) {
        return false;
    }
    residentBytes = std::max(baseBytes, mixedBytes);
    if (residentBytes == 0 || residentBytes > available ||
        residentBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    // 至少要把原单槽容量压缩为两个 panel，才有可能形成真实的搬入/计算重叠。
    if (residentBytes > available ||
        CeilDiv(indexSize, baseChunk) <= 1 || CeilDiv(indexSize, mixedChunk) <= 1) {
        return false;
    }

    tiling.set_strategy(STRATEGY_UB_MIXED_DB);
    tiling.set_residentBufferBytes(static_cast<uint32_t>(residentBytes));
    tiling.set_residentDoubleBuffer(1);
    return true;
}

bool SetNarrowUbTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize, uint64_t indexSize,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    uint64_t axisBytes = 0;
    uint64_t indexBytes = 0;
    if (MulOverflow(axisSize, elementSize, axisBytes) ||
        MulOverflow(indexSize, sizeof(int32_t), indexBytes)) {
        return false;
    }
    const uint64_t accumulatorRowBytes = AlignUp(axisBytes, BLOCK_BYTES);
    const uint64_t accumulatorRowBlocks = accumulatorRowBytes / BLOCK_BYTES;
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    if (accumulatorRowBlocks == 0 || accumulatorRowBlocks > 255 || indexBytes >= ubSize) {
        return false;
    }

    // 每个 outer 同时预留一行累加器和一行 256B source mask。
    const uint64_t perOuterBytes = accumulatorRowBytes + VECTOR_BYTES;
    const uint64_t maxOuterTile = std::min<uint64_t>(255, (ubSize - indexBytes) / perOuterBytes);
    if (maxOuterTile == 0) {
        return false;
    }
    const uint64_t targetOuterTile = CeilDiv(outerSize, coreNum);
    const uint64_t outerTile = std::min(targetOuterTile, maxOuterTile);
    const uint64_t outerTileNum = CeilDiv(outerSize, outerTile);
    if (outerTileNum > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    SetCoreTiling(tiling, 1, outerTile, outerTile, STRATEGY_NARROW_UB,
        1, outerTileNum, outerTileNum, 1);
    return true;
}

bool SetGatherNarrowTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize, uint64_t indexSize,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    const uint64_t gatherCountLimit =
        elementSize == sizeof(uint16_t) ? GATHER_FP16_MAX_COUNT : GATHER_FP32_MAX_COUNT;
    if (axisSize == 0 || axisSize > gatherCountLimit || ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    // Gather 路径逐 outer 执行，每层包含一次 Gather 和一次 Add。
    const uint64_t targetOuter = std::min<uint64_t>(CeilDiv(outerSize, coreNum),
        std::numeric_limits<uint32_t>::max());
    const uint64_t lowerBoundLayers = CeilDiv(indexSize, axisSize);
    uint64_t lowerBoundCalls = 0;
    if (targetOuter == 0 || lowerBoundLayers == 0 ||
        MulOverflow(lowerBoundLayers, targetOuter, lowerBoundCalls) ||
        lowerBoundCalls > (indexSize - 1) / 2) {
        return false;
    }

    uint64_t axisBytes = 0;
    uint64_t indexBytes = 0;
    uint64_t sourceValidBytes = 0;
    uint64_t counterValidBytes = 0;
    if (MulOverflow(axisSize, elementSize, axisBytes) ||
        MulOverflow(indexSize, sizeof(int32_t), indexBytes) ||
        MulOverflow(indexSize, elementSize, sourceValidBytes) ||
        MulOverflow(axisSize, sizeof(uint32_t), counterValidBytes) ||
        axisBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        indexBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        sourceValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        counterValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }

    const uint64_t accumulatorRowBytes = AlignUp(axisBytes, BLOCK_BYTES);
    const uint64_t denseRowBytes = accumulatorRowBytes;
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    const uint64_t counterRowBytes = AlignUp(counterValidBytes, BLOCK_BYTES);
    const uint64_t sourceRowBytes = AlignUp(sourceValidBytes, BLOCK_BYTES);
    if (sourceRowBytes > std::numeric_limits<uint64_t>::max() - BLOCK_BYTES) {
        return false;
    }
    // 回退路径复用 source buffer 作为一行 256B 的 masked Add staging。
    const uint64_t sourceBufferBytes = std::max<uint64_t>(sourceRowBytes + BLOCK_BYTES, VECTOR_BYTES);

    uint64_t remainingUb = ubSize - GATHER_UB_RESERVE_BYTES;
    const uint64_t fixedBuffers[] = {
        indexBytes, counterRowBytes, accumulatorRowBytes, denseRowBytes, sourceBufferBytes};
    for (const uint64_t bufferBytes : fixedBuffers) {
        if (bufferBytes > remainingUb) {
            return false;
        }
        remainingUb -= bufferBytes;
    }

    const uint64_t layerCap = std::min(indexSize, remainingUb / counterRowBytes);
    uint64_t layerOffsetBytes = 0;
    if (layerCap < lowerBoundLayers || layerCap > std::numeric_limits<uint32_t>::max() ||
        MulOverflow(layerCap, counterRowBytes, layerOffsetBytes) ||
        layerOffsetBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const uint64_t outerTileNum = CeilDiv(outerSize, targetOuter);
    if (outerTileNum == 0 || outerTileNum > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    SetCoreTiling(tiling, 1, targetOuter, 1, STRATEGY_GATHER_NARROW,
        1, outerTileNum, outerTileNum, layerCap);
    // strategy 3 不使用 source chunk，该字段复用为 occurrence layer 容量。
    return true;
}

bool SetGatherBatchTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize, uint64_t indexSize,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    if (outerSize == 0 || axisSize == 0 || indexSize == 0 || elementSize == 0 || coreNum == 0 ||
        ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    uint64_t axisBytes = 0;
    uint64_t indexValidBytes = 0;
    uint64_t counterValidBytes = 0;
    uint64_t sourceValidBytes = 0;
    if (MulOverflow(axisSize, elementSize, axisBytes) ||
        MulOverflow(indexSize, sizeof(int32_t), indexValidBytes) ||
        MulOverflow(axisSize, sizeof(uint32_t), counterValidBytes) ||
        MulOverflow(indexSize, elementSize, sourceValidBytes) ||
        axisBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        indexValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        counterValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        sourceValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }

    const uint64_t alignedAxisBytes = AlignUp(axisBytes, BLOCK_BYTES);
    const uint64_t paddedAxisElements = alignedAxisBytes / elementSize;
    const uint64_t indexBytes = AlignUp(indexValidBytes, BLOCK_BYTES);
    const uint64_t counterBytes = AlignUp(counterValidBytes, BLOCK_BYTES);
    const uint64_t alignedSourceBytes = AlignUp(sourceValidBytes, BLOCK_BYTES);
    uint64_t offsetRowBytes = 0;
    uint64_t sourcePitch = 0;
    uint64_t offsetScratchRowBytes = 0;
    if (MulOverflow(paddedAxisElements, sizeof(uint32_t), offsetRowBytes) ||
        AddOverflow(alignedSourceBytes, BLOCK_BYTES, sourcePitch) ||
        MulOverflow(VECTOR_BYTES / elementSize, sizeof(uint32_t), offsetScratchRowBytes)) {
        return false;
    }

    const uint64_t lowerLayers = CeilDiv(indexSize, axisSize);
    const uint64_t dataVectorElements = VECTOR_BYTES / elementSize;
    const uint64_t offsetRepeats = CeilDiv(axisSize, 64);
    const uint64_t dataRepeats = CeilDiv(axisSize, dataVectorElements);
    uint64_t repeatsPerLayer = 0;
    uint64_t lowerBenefitCalls = 0;
    if (MulOverflow(dataRepeats, 2, repeatsPerLayer) ||
        AddOverflow(repeatsPerLayer, offsetRepeats, repeatsPerLayer) ||
        MulOverflow(lowerLayers, repeatsPerLayer, lowerBenefitCalls) || lowerBenefitCalls >= indexSize) {
        return false;
    }

    const uint64_t requiredLayerCap =
        lowerLayers > indexSize / 3 ? indexSize : lowerLayers * 3;
    const uint64_t targetBatch = CeilDiv(outerSize, coreNum);
    const uint64_t maxBatch = std::min<uint64_t>(255, targetBatch);
    const uint64_t ubBudget = ubSize - GATHER_UB_RESERVE_BYTES;
    uint64_t fixedBytes = 0;
    if (AddOverflow(indexBytes, counterBytes, fixedBytes) || fixedBytes >= ubBudget || offsetRowBytes == 0) {
        return false;
    }

    uint64_t perBatchBytes = 0;
    if (AddOverflow(sourcePitch, BLOCK_BYTES, perBatchBytes) ||
        AddOverflow(perBatchBytes, offsetScratchRowBytes, perBatchBytes) ||
        AddOverflow(perBatchBytes, 2 * VECTOR_BYTES, perBatchBytes)) {
        return false;
    }

    for (uint64_t batch = maxBatch; batch > 0; --batch) {
        uint64_t batchBytes = 0;
        uint64_t usedBytes = 0;
        const uint64_t dataSkewBytes = batch % 2 == 0 ? VECTOR_BYTES : 0;
        if (MulOverflow(batch, perBatchBytes, batchBytes) || AddOverflow(fixedBytes, batchBytes, usedBytes) ||
            AddOverflow(usedBytes, dataSkewBytes, usedBytes) || usedBytes >= ubBudget) {
            continue;
        }
        const uint64_t layerRaw = (ubBudget - usedBytes) / offsetRowBytes;
        uint64_t layerCap = std::min(indexSize, layerRaw);
        if (layerCap < requiredLayerCap) {
            continue;
        }

        const uint64_t taskNum = CeilDiv(outerSize, batch);
        if (taskNum == 0 || taskNum > std::numeric_limits<uint32_t>::max() ||
            layerCap > std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        SetCoreTiling(tiling, 1, batch, batch, STRATEGY_GATHER_BATCH,
            1, taskNum, taskNum, layerCap);
        return true;
    }
    return false;
}

// 对于 16-bit、axis 恰跨两个 Gather repeat 的窄轴，普通 batch Gather 要对每个
// axis chunk 分别发射 offset Add、Gather、Add。这里把每个 outer 的两个 repeat
// 连续打包为一行：尾部用 source sentinel 填满，Gather/Add 各少一次发射。index
// 缓冲在 metadata 建表后不再使用，复用为宽 offset 表，避免为该阶段额外常驻一块 UB。
bool TrySetGatherBatchWideTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t indexSize, uint32_t elementSize, uint32_t coreNum, uint64_t ubSize)
{
    if (tiling.get_strategy() != STRATEGY_GATHER_BATCH || elementSize != sizeof(uint16_t) ||
        outerSize == 0 || axisSize == 0 || indexSize == 0 || coreNum == 0 ||
        ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    const uint64_t gatherElements = VECTOR_BYTES / elementSize;
    const uint64_t wideElements = 2 * gatherElements;
    const uint64_t offsetElements = VECTOR_BYTES / sizeof(uint32_t);
    if (gatherElements == 0 || axisSize <= gatherElements || axisSize > wideElements ||
        offsetElements == 0) {
        return false;
    }

    const uint64_t normalBatch = tiling.get_outerBatch();
    const uint64_t normalLayerCap = tiling.get_sourceChunk();
    if (normalBatch == 0 || normalLayerCap == 0) {
        return false;
    }

    const uint64_t normalTail = axisSize - gatherElements;
    const uint64_t normalOffsetCalls = CeilDiv(gatherElements, offsetElements) +
        CeilDiv(normalTail, offsetElements);
    const uint64_t normalCallsPerLayer = normalOffsetCalls + 4; // 两段各 Gather + Add。
    const uint64_t wideCallsPerLayer = CeilDiv(wideElements, offsetElements) + 2;
    const uint64_t maxCoreOuter = CeilDiv(outerSize, coreNum);
    uint64_t normalBatchCalls = 0;
    uint64_t normalVectorCalls = 0;
    if (MulOverflow(CeilDiv(maxCoreOuter, normalBatch), normalCallsPerLayer, normalVectorCalls)) {
        return false;
    }
    normalBatchCalls = CeilDiv(maxCoreOuter, normalBatch);
    if (normalBatchCalls == 0 || normalVectorCalls == 0) {
        return false;
    }

    uint64_t indexValidBytes = 0;
    uint64_t counterValidBytes = 0;
    uint64_t sourceValidBytes = 0;
    uint64_t wideOffsetRowBytes = 0;
    uint64_t wideDataRowBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexValidBytes) ||
        MulOverflow(axisSize, sizeof(uint32_t), counterValidBytes) ||
        MulOverflow(indexSize, elementSize, sourceValidBytes) ||
        MulOverflow(wideElements, sizeof(uint32_t), wideOffsetRowBytes) ||
        MulOverflow(wideElements, elementSize, wideDataRowBytes) ||
        indexValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        counterValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        sourceValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }

    const uint64_t indexBytes = AlignUp(indexValidBytes, BLOCK_BYTES);
    const uint64_t counterBytes = AlignUp(counterValidBytes, BLOCK_BYTES);
    const uint64_t sourcePitchBytes = AlignUp(sourceValidBytes, BLOCK_BYTES) + BLOCK_BYTES;
    const uint64_t lowerLayers = CeilDiv(indexSize, axisSize);
    const uint64_t requiredLayerCap = lowerLayers > indexSize / 3 ? indexSize : lowerLayers * 3;
    const uint64_t targetBatch = CeilDiv(outerSize, coreNum);
    // 每个 outer 被拆为两个连续 repeat，硬件 repeatTime 的上限是 255。
    const uint64_t maxBatch = std::min<uint64_t>({127, targetBatch, MTE_BLOCK_COUNT_LIMIT});
    const uint64_t ubBudget = ubSize - GATHER_UB_RESERVE_BYTES;

    for (uint64_t batch = maxBatch; batch > 0; --batch) {
        uint64_t wideOffsetBytes = 0;
        uint64_t sourceBytes = 0;
        uint64_t dataBytes = 0;
        uint64_t rowBaseBytes = 0;
        uint64_t usedBytes = 0;
        if (MulOverflow(batch, wideOffsetRowBytes, wideOffsetBytes) ||
            MulOverflow(batch, sourcePitchBytes, sourceBytes) ||
            MulOverflow(batch, wideDataRowBytes, dataBytes) ||
            MulOverflow(batch, BLOCK_BYTES, rowBaseBytes)) {
            continue;
        }

        // wideOffset 复用 indexBuf，两个阶段的峰值取 max 而非相加。wide data
        // 行是 16 个 32B block，始终补一条 Vector 行让 accumulator/dense 错开 bank group。
        usedBytes = std::max(indexBytes, wideOffsetBytes);
        if (AddOverflow(usedBytes, counterBytes, usedBytes) ||
            AddOverflow(usedBytes, sourceBytes, usedBytes) ||
            AddOverflow(usedBytes, dataBytes, usedBytes) ||
            AddOverflow(usedBytes, dataBytes, usedBytes) ||
            AddOverflow(usedBytes, rowBaseBytes, usedBytes) ||
            AddOverflow(usedBytes, VECTOR_BYTES, usedBytes) || usedBytes >= ubBudget) {
            continue;
        }

        const uint64_t layerRaw = (ubBudget - usedBytes) / wideOffsetRowBytes;
        const uint64_t layerCap = std::min(indexSize, layerRaw);
        if (layerCap < requiredLayerCap) {
            continue;
        }

        uint64_t wideVectorCalls = 0;
        const uint64_t taskNum = CeilDiv(outerSize, batch);
        if (taskNum == 0 || taskNum > std::numeric_limits<uint32_t>::max() ||
            layerCap > std::numeric_limits<uint32_t>::max() ||
            MulOverflow(CeilDiv(maxCoreOuter, batch), wideCallsPerLayer, wideVectorCalls) ||
            wideVectorCalls >= normalVectorCalls) {
            continue;
        }

        SetCoreTiling(tiling, 1, batch, batch, STRATEGY_GATHER_BATCH_WIDE,
            1, taskNum, taskNum, layerCap);
        return true;
    }
    return false;
}

// FP32 cooperative 窄 Gather 的每个 outer 都保留完整 payload；在 UB 能容纳批量
// payload 且批量化能显著减少外层发射时，尝试复用 strategy 4 的 repeat 数据流。
// 所有门槛仅由运行时 shape、UB、核数和硬件 repeat 限制推导。
bool SetCooperativeGatherWorkspace(IndexAddTilingData& tiling, uint64_t axisSize, uint64_t indexSize,
    ge::DataType dataType, uint32_t elementSize, uint32_t coreNum, uint64_t ubSize, size_t& workspaceSize)
{
    const uint32_t strategy = tiling.get_strategy();
    if ((strategy != STRATEGY_GATHER_NARROW && strategy != STRATEGY_GATHER_BATCH &&
        strategy != STRATEGY_GATHER_BATCH_WIDE) ||
        coreNum <= 1 || axisSize == 0 || axisSize > MTE_BLOCK_COUNT_LIMIT ||
        indexSize < static_cast<uint64_t>(coreNum) * 8 ||
        indexSize > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    uint64_t sourceBytes = 0;
    if (MulOverflow(indexSize, elementSize, sourceBytes) ||
        sourceBytes > std::numeric_limits<uint32_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }

    uint64_t layerCap = 0;
    if (MulOverflow(CeilDiv(indexSize, axisSize), COOP_LAYER_CAP_FACTOR, layerCap)) {
        return false;
    }
    layerCap = std::min<uint64_t>(layerCap, tiling.get_sourceChunk());
    if (layerCap == 0 || layerCap > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    uint64_t counterValidBytes = 0;
    uint64_t axisValidBytes = 0;
    if (MulOverflow(axisSize, sizeof(uint32_t), counterValidBytes) ||
        MulOverflow(axisSize, elementSize, axisValidBytes)) {
        return false;
    }
    const uint64_t counterRowBytes = AlignUp(counterValidBytes, BLOCK_BYTES);
    const uint64_t accumulatorRowBytes = AlignUp(axisValidBytes, BLOCK_BYTES);
    const uint64_t paddedAxisElements = accumulatorRowBytes / elementSize;
    const uint64_t batchOffsetElements = strategy == STRATEGY_GATHER_BATCH_WIDE ?
        2 * (VECTOR_BYTES / elementSize) : paddedAxisElements;
    uint64_t batchLayerOffsetRowBytes = 0;
    if (batchOffsetElements == 0 ||
        MulOverflow(batchOffsetElements, sizeof(uint32_t), batchLayerOffsetRowBytes)) {
        return false;
    }
    const uint64_t layerOffsetRowBytes =
        strategy == STRATEGY_GATHER_NARROW ? counterRowBytes : batchLayerOffsetRowBytes;

    uint64_t layerBufferBytes = 0;
    uint64_t compactBytes = 0;
    uint64_t rawRowBytes = 0;
    if (MulOverflow(tiling.get_sourceChunk(), layerOffsetRowBytes, layerBufferBytes) ||
        MulOverflow(layerCap, layerOffsetRowBytes, compactBytes) ||
        MulOverflow(axisSize, COOP_RAW_SLOT_BYTES, rawRowBytes)) {
        return false;
    }

    const uint64_t maxOwnedAxes = CeilDiv(axisSize, coreNum);
    uint64_t countMatrixRowBytes = 0;
    uint64_t countMatrixBytes = 0;
    uint64_t phaseOneBytes = 0;
    if (MulOverflow(maxOwnedAxes, sizeof(uint32_t), countMatrixRowBytes) ||
        countMatrixRowBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }
    countMatrixRowBytes = AlignUp(countMatrixRowBytes, BLOCK_BYTES);
    if (MulOverflow(coreNum, countMatrixRowBytes, countMatrixBytes)) {
        return false;
    }
    // prefix 在 UB 内复用 count 子矩阵完成转置，不再额外保留 axis-major staging。
    phaseOneBytes = countMatrixBytes;

    uint64_t totalRawSlots = 0;
    uint64_t initBytes = 0;
    uint64_t indexStageBytes = 0;
    if (MulOverflow(layerCap, axisSize, totalRawSlots) ||
        MulOverflow(CeilDiv(totalRawSlots, coreNum), COOP_RAW_SLOT_BYTES, initBytes) ||
        MulOverflow(CeilDiv(indexSize, coreNum), COOP_RAW_SLOT_BYTES, indexStageBytes) ||
        AddOverflow(rawRowBytes, indexStageBytes, indexStageBytes)) {
        return false;
    }

    uint64_t requiredLayerBytes = std::max({phaseOneBytes, initBytes, indexStageBytes});
    uint64_t rankBytes = 0;
    if (MulOverflow(CeilDiv(indexSize, coreNum), sizeof(uint32_t), rankBytes)) {
        return false;
    }
    if (strategy == STRATEGY_GATHER_NARROW) {
        uint64_t compactScratchBytes = 0;
        if (AddOverflow(layerOffsetRowBytes, rawRowBytes, compactScratchBytes) ||
            AddOverflow(compactScratchBytes, counterRowBytes, compactScratchBytes) ||
            rankBytes > AlignUp(axisValidBytes, BLOCK_BYTES)) {
            return false;
        }
        requiredLayerBytes = std::max(requiredLayerBytes, compactScratchBytes);
    } else {
        uint64_t rawScratchBytes = 0;
        uint64_t rankScratchBytes = 0;
        if (MulOverflow(tiling.get_outerBatch(), VECTOR_BYTES, rawScratchBytes) ||
            (tiling.get_outerBatch() % 2 == 0 &&
                AddOverflow(rawScratchBytes, VECTOR_BYTES, rawScratchBytes)) ||
            MulOverflow(tiling.get_outerBatch(), VECTOR_BYTES, rankScratchBytes) ||
            rawScratchBytes < rawRowBytes || rankBytes > rankScratchBytes) {
            return false;
        }
    }
    if (requiredLayerBytes > layerBufferBytes) {
        return false;
    }

    const uint64_t countPitchBytes = counterRowBytes;
    uint64_t countBytes = 0;
    uint64_t prefixBytes = 0;
    uint64_t coreMaxBytes = 0;
    uint64_t rawBytes = 0;
    // workspace 中 prefix 以消费者 core 为首维，供后续一次连续搬入。
    const uint64_t prefixCorePitchBytes = counterRowBytes;
    if (MulOverflow(coreNum, countPitchBytes, countBytes) ||
        MulOverflow(coreNum, prefixCorePitchBytes, prefixBytes) ||
        MulOverflow(coreNum, COOP_RAW_SLOT_BYTES, coreMaxBytes) ||
        MulOverflow(totalRawSlots, COOP_RAW_SLOT_BYTES, rawBytes)) {
        return false;
    }

    uint64_t workspaceBytes = 0;
    if (!AppendAlignedRegion(workspaceBytes, countBytes) ||
        !AppendAlignedRegion(workspaceBytes, prefixBytes) ||
        !AppendAlignedRegion(workspaceBytes, coreMaxBytes) ||
        !AppendAlignedRegion(workspaceBytes, COOP_RAW_SLOT_BYTES) ||
        !AppendAlignedRegion(workspaceBytes, rawBytes) ||
        !AppendAlignedRegion(workspaceBytes, compactBytes) ||
        workspaceBytes > std::numeric_limits<size_t>::max()) {
        return false;
    }

    tiling.set_coopLayerCap(static_cast<uint32_t>(layerCap));
    const uint64_t denseSkewBytes = dataType == ge::DT_FLOAT ?
        GATHER_PIPELINE_FP32_DENSE_SKEW_BYTES : GATHER_PIPELINE_DENSE_SKEW_BYTES;
    if (strategy == STRATEGY_GATHER_NARROW && tiling.get_outerTile() >= 3 && requiredLayerBytes <= compactBytes &&
        ubSize > GATHER_UB_RESERVE_BYTES) {
        const uint64_t indexBytes = AlignUp(indexSize * sizeof(int32_t), BLOCK_BYTES);
        const uint64_t sourceRowBytes = AlignUp(sourceBytes, BLOCK_BYTES);
        uint64_t slotBytes = 0;
        uint64_t usedBytes = 0;
        if (GetGatherDoubleBufferBytes(indexBytes, sourceRowBytes, accumulatorRowBytes,
                counterRowBytes, compactBytes, denseSkewBytes, slotBytes, usedBytes) &&
            slotBytes <= std::numeric_limits<uint32_t>::max() &&
            usedBytes <= ubSize - GATHER_UB_RESERVE_BYTES) {
            tiling.set_strategy(STRATEGY_GATHER_NARROW_DB);
            // 双槽路径只保留 cooperative 实际需要的 occurrence layer。
            tiling.set_sourceChunk(static_cast<uint32_t>(layerCap));
        }
    }
    workspaceSize = static_cast<size_t>(workspaceBytes);
    return true;
}

// 16 行转置块路径在 kernel 内额外同时保留 source staging、转置 source 和
// accumulator；必须用 host 实测 UB 容量逐项预算，不能只凭 TransData 的 repeat 上限启用。
bool EnableTransposeBlockGather(IndexAddTilingData& tiling, ge::DataType dataType,
    uint64_t axisSize, uint64_t indexSize, uint64_t ubSize)
{
    constexpr uint64_t TRANSPOSE_ROWS = 16;
    constexpr uint64_t ELEMENT_BYTES = sizeof(uint16_t);
    const uint64_t gatherElements = VECTOR_BYTES / ELEMENT_BYTES;
    const uint64_t axisPadElements = 2 * gatherElements;
    if (tiling.get_strategy() != STRATEGY_GATHER_BATCH_WIDE || dataType != ge::DT_FLOAT16 ||
        axisSize <= gatherElements || axisSize > axisPadElements || indexSize < TRANSPOSE_ROWS ||
        indexSize % TRANSPOSE_ROWS != 0 || indexSize / TRANSPOSE_ROWS > 255 ||
        tiling.get_sourceChunk() == 0 || tiling.get_coopLayerCap() == 0 ||
        ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    uint64_t indexValidBytes = 0;
    uint64_t counterValidBytes = 0;
    uint64_t layerRowBytes = 0;
    uint64_t layerBytes = 0;
    uint64_t rawScratchBytes = 0;
    uint64_t accumulatorBytes = 0;
    uint64_t sourceStageElements = 0;
    uint64_t sourceStageBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexValidBytes) ||
        MulOverflow(axisSize, sizeof(uint32_t), counterValidBytes) ||
        MulOverflow(axisPadElements, sizeof(uint32_t), layerRowBytes) ||
        MulOverflow(tiling.get_sourceChunk(), layerRowBytes, layerBytes) ||
        MulOverflow(axisSize, COOP_RAW_SLOT_BYTES, rawScratchBytes) ||
        MulOverflow(axisPadElements, TRANSPOSE_ROWS, accumulatorBytes) ||
        MulOverflow(accumulatorBytes, ELEMENT_BYTES, accumulatorBytes) ||
        MulOverflow(TRANSPOSE_ROWS, indexSize, sourceStageElements) ||
        MulOverflow(sourceStageElements, ELEMENT_BYTES, sourceStageBytes) ||
        indexValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1) ||
        counterValidBytes > std::numeric_limits<uint64_t>::max() - (BLOCK_BYTES - 1)) {
        return false;
    }

    const uint64_t indexBytes = AlignUp(indexValidBytes, BLOCK_BYTES);
    const uint64_t counterBytes = AlignUp(counterValidBytes, BLOCK_BYTES);
    // dataBuf_ 在协同建表阶段保存 raw scratch，payload 阶段复用为双缓冲 prefetchedSelf；
    // 准入预算按二者较大值计入，同时 sourceWorkBytes 容纳双缓冲 sourceRows。
    const uint64_t prefetchSelfBytes = 2 * accumulatorBytes;
    const uint64_t dataBufBytes = std::max(rawScratchBytes, prefetchSelfBytes);
    uint64_t sourceTransposeBytes = 0;
    uint64_t sourceWorkBytes = 0;
    uint64_t usedBytes = indexBytes;
    if (AddOverflow(sourceStageBytes, BLOCK_BYTES, sourceTransposeBytes) ||
        AddOverflow(2 * sourceStageBytes, sourceTransposeBytes, sourceWorkBytes) ||
        AddOverflow(sourceWorkBytes, accumulatorBytes, sourceWorkBytes) ||
        AddOverflow(usedBytes, counterBytes, usedBytes) ||
        AddOverflow(usedBytes, layerBytes, usedBytes) ||
        AddOverflow(usedBytes, dataBufBytes, usedBytes) ||
        AddOverflow(usedBytes, sourceWorkBytes, usedBytes) ||
        AddOverflow(usedBytes, accumulatorBytes, usedBytes)) {
        return false;
    }
    return usedBytes <= ubSize - GATHER_UB_RESERVE_BYTES;
}

// FP32 的 16 行转置以 8 行有效 half 作为 Gatherb payload。source chunk 上限
// 由 UB、TransDataTo5HD repeat 和 workspace 预算共同决定；超过资源上限时所有
// core 同步回退到既有保序窄路径。
bool EnableFp32TransposeBlockGather(IndexAddTilingData& tiling, ge::DataType dataType,
    uint64_t axisSize, uint64_t indexSize, uint64_t ubSize, size_t& workspaceSize)
{
    if (tiling.get_strategy() != STRATEGY_GATHER_NARROW_DB || dataType != ge::DT_FLOAT ||
        axisSize < 4 * (VECTOR_BYTES / sizeof(float)) || indexSize < FP32_TRANSPOSE_ACTIVE_ROWS ||
        ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    const uint64_t axisPad = AlignUp(axisSize, FP32_TRANSPOSE_COL_ALIGNMENT);
    if (axisPad == 0 || axisPad / FP32_TRANSPOSE_COL_ALIGNMENT > 255 ||
        axisPad > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    uint64_t indexBytes = 0;
    uint64_t counterBytes = 0;
    uint64_t rawTableBytes = 0;
    uint64_t accumulatorBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) ||
        MulOverflow(axisPad, sizeof(uint32_t), counterBytes) ||
        MulOverflow(FP32_TRANSPOSE_LAYER_CAP, counterBytes, rawTableBytes) ||
        MulOverflow(FP32_TRANSPOSE_ROWS, axisPad, accumulatorBytes) ||
        MulOverflow(accumulatorBytes, sizeof(float), accumulatorBytes)) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    counterBytes = AlignUp(counterBytes, BLOCK_BYTES);

    // metadata 的完整表暂借 denseBuf（其容量为 16 个 axis 行），payload 保持
    // Gatherb 直接消费的 32-bit byte-offset 表，不引入逐层解码。
    if (rawTableBytes > accumulatorBytes) {
        return false;
    }
    const uint64_t usableUb = ubSize - GATHER_UB_RESERVE_BYTES;
    const uint64_t sourceSentinelBytes = FP32_TRANSPOSE_ROWS * sizeof(float);
    // payload 的 Gather 输出只需要一个低/高 half 共用的临时平面；expanded
    // offset 表放在该平面之后，大小随当前 payload 分组变化，而不是固定占满
    // 16 行 accumulator。这样释放出的 UB 可优先用于扩大 source chunk。
    const uint64_t accumulatorPlaneBytes =
        FP32_TRANSPOSE_COL_ALIGNMENT * axisPad * sizeof(float);
    uint64_t bytesPerSourceColumn = 0;
    if (MulOverflow(FP32_TRANSPOSE_ACTIVE_ROWS + FP32_TRANSPOSE_ROWS, sizeof(float),
            bytesPerSourceColumn)) {
        return false;
    }
    const uint64_t maxSourceChunk = FP32_TRANSPOSE_MAX_REPEAT * FP32_TRANSPOSE_COL_ALIGNMENT;

    auto CalculateSourceChunk = [&](uint64_t payloadLayerCap, uint32_t offsetBytes,
        uint64_t& sourceChunk) -> bool {
        // metadata 的 counter 与 payload offset 表严格处于两个阶段。payload 的
        // 行数即实际分组上限，不为预取额外保留一行，以免挤压 source 连续块。
        const uint64_t payloadLayerRows = payloadLayerCap;
        // compact 表缩小了 layerOffsetBuf，但会把更多 source 行压入同一 UB。
        // TPipe/向量临时状态还需要少量未显式计数的余量；仅对 compact 保留
        // 两个 Vector 行，避免边界上刚好装满时触发 VEC 越界。
        const uint64_t sourceSafetyBytes = offsetBytes == sizeof(uint16_t) ?
            2 * VECTOR_BYTES : 0;
        uint64_t payloadRowBytes = 0;
        uint64_t payloadTableBytes = 0;
        uint64_t expandedScratchBytes = 0;
        uint64_t dataBufBytes = 0;
        uint64_t fixedPayloadBytes = 0;
        if (payloadLayerRows == 0 ||
            MulOverflow(axisPad, offsetBytes, payloadRowBytes) ||
            MulOverflow(payloadLayerRows, payloadRowBytes, payloadTableBytes) ||
            MulOverflow(payloadLayerRows, axisPad, expandedScratchBytes) ||
            MulOverflow(expandedScratchBytes, sizeof(uint32_t), expandedScratchBytes) ||
            AddOverflow(accumulatorPlaneBytes, expandedScratchBytes, dataBufBytes) ||
            payloadTableBytes > accumulatorBytes ||
            // payload 的紧凑表、输出 staging、累加器和 metadata counter 分别
            // 常驻 layerOffsetBuf/dataBuf/denseBuf/counterBuf；dataBuf 的 expanded
            // offset scratch 只按当前分组保留，不能再按完整 16 行 accumulator 计数。
            AddOverflow(payloadTableBytes, dataBufBytes, fixedPayloadBytes) ||
            AddOverflow(fixedPayloadBytes, accumulatorBytes, fixedPayloadBytes) ||
            AddOverflow(fixedPayloadBytes, counterBytes, fixedPayloadBytes) ||
            fixedPayloadBytes >= usableUb) {
            return false;
        }

        // 先按不含 header 的上界计算 source 容量，再以实际 chunk 数回填 header
        // 开销至收敛。header 随 chunk 变小只会增大，循环单调收敛而不依赖固定 shape。
        const uint64_t sourceBufferBudget = usableUb - fixedPayloadBytes;
        if (indexBytes > sourceBufferBudget ||
            sourceBufferBudget <= sourceSentinelBytes + sourceSafetyBytes) {
            return false;
        }
        uint64_t sourceCapacity =
            (sourceBufferBudget - sourceSentinelBytes - sourceSafetyBytes) /
            bytesPerSourceColumn;
        sourceChunk = std::min<uint64_t>(sourceCapacity, maxSourceChunk);
        sourceChunk = std::min<uint64_t>(sourceChunk, indexSize);
        sourceChunk = sourceChunk / FP32_TRANSPOSE_COL_ALIGNMENT * FP32_TRANSPOSE_COL_ALIGNMENT;
        while (sourceChunk >= FP32_TRANSPOSE_COL_ALIGNMENT) {
            const uint64_t chunkCount = CeilDiv(indexSize, sourceChunk);
            uint64_t headerBytes = 0;
            uint64_t fixedBytes = 0;
            if (chunkCount == 0 ||
                MulOverflow(chunkCount + 1, BLOCK_BYTES, headerBytes) ||
                headerBytes > counterBytes ||
                AddOverflow(fixedPayloadBytes, headerBytes, fixedBytes) ||
                fixedBytes >= usableUb) {
                return false;
            }
            const uint64_t adjustedBudget = usableUb - fixedBytes;
            if (indexBytes > adjustedBudget ||
                adjustedBudget <= sourceSentinelBytes + sourceSafetyBytes) {
                return false;
            }
            sourceCapacity = (adjustedBudget - sourceSentinelBytes - sourceSafetyBytes) /
                bytesPerSourceColumn;
            uint64_t adjustedChunk = std::min<uint64_t>(sourceCapacity, maxSourceChunk);
            adjustedChunk = std::min<uint64_t>(adjustedChunk, indexSize);
            adjustedChunk = adjustedChunk / FP32_TRANSPOSE_COL_ALIGNMENT *
                FP32_TRANSPOSE_COL_ALIGNMENT;
            if (adjustedChunk == sourceChunk) {
                return sourceChunk / FP32_TRANSPOSE_COL_ALIGNMENT <= FP32_TRANSPOSE_MAX_REPEAT &&
                    sourceChunk <= std::numeric_limits<uint32_t>::max();
            }
            // header 只会让容量收缩；若出现反向变化，拒绝该候选以保持预算保守。
            if (adjustedChunk > sourceChunk) {
                return false;
            }
            sourceChunk = adjustedChunk;
        }
        return false;
    };

    auto SelectSourceChunk = [&](uint64_t& sourceChunk, uint64_t& payloadLayerCap,
        uint32_t& offsetBytes) -> bool {
        uint64_t wideSourceChunk = 0;
        if (!CalculateSourceChunk(FP32_TRANSPOSE_LAYER_CAP, sizeof(uint32_t), wideSourceChunk)) {
            return false;
        }
        // 32B record 布局下，compact 表的最大 source chunk 由有符号
        // int16 的可精确表示范围推导；超出范围时仍使用 32 位 offset。
        const uint64_t compactSourceChunkLimit =
            (static_cast<uint64_t>(std::numeric_limits<int16_t>::max()) /
                FP32_TRANSPOSE_OFFSET_SCALE_BYTES) /
            FP32_TRANSPOSE_COL_ALIGNMENT * FP32_TRANSPOSE_COL_ALIGNMENT;
        sourceChunk = wideSourceChunk;
        offsetBytes = sizeof(uint32_t);
        if (wideSourceChunk <= compactSourceChunkLimit) {
            uint64_t compactSourceChunk = 0;
            if (CalculateSourceChunk(FP32_TRANSPOSE_LAYER_CAP, sizeof(uint16_t),
                    compactSourceChunk) && compactSourceChunk <= compactSourceChunkLimit) {
                sourceChunk = compactSourceChunk;
                offsetBytes = sizeof(uint16_t);
            }
        }
        payloadLayerCap = FP32_TRANSPOSE_LAYER_CAP;
        const uint64_t baselineChunkCount = CeilDiv(indexSize, sourceChunk);
        // 只有实际减少 source chunk 数时才压缩 payload 表。优先选择最大的可用层数，
        // 避免为没有实际搬运收益的额外 table DMA 付出代价。
        for (uint64_t candidateLayerCap = FP32_TRANSPOSE_LAYER_CAP - 1;
             candidateLayerCap != 0; --candidateLayerCap) {
            uint64_t candidateWideChunk = 0;
            if (!CalculateSourceChunk(candidateLayerCap, sizeof(uint32_t), candidateWideChunk)) {
                continue;
            }
            uint64_t candidateSourceChunk = candidateWideChunk;
            uint32_t candidateOffsetBytes = sizeof(uint32_t);
            if (candidateWideChunk <= compactSourceChunkLimit) {
                uint64_t candidateCompactChunk = 0;
                if (CalculateSourceChunk(candidateLayerCap, sizeof(uint16_t),
                        candidateCompactChunk) && candidateCompactChunk <= compactSourceChunkLimit) {
                    candidateSourceChunk = candidateCompactChunk;
                    candidateOffsetBytes = sizeof(uint16_t);
                }
            }
            if (CeilDiv(indexSize, candidateSourceChunk) < baselineChunkCount) {
                payloadLayerCap = candidateLayerCap;
                sourceChunk = candidateSourceChunk;
                offsetBytes = candidateOffsetBytes;
                break;
            }
        }
        return true;
    };

    uint64_t rawSourceChunk = 0;
    uint64_t rawPayloadLayerCap = 0;
    uint32_t rawOffsetBytes = sizeof(uint32_t);
    if (!SelectSourceChunk(rawSourceChunk, rawPayloadLayerCap, rawOffsetBytes)) {
        return false;
    }

    uint64_t sourceChunk = rawSourceChunk;
    uint64_t payloadLayerCap = rawPayloadLayerCap;
    const uint32_t offsetBytes = rawOffsetBytes;
    uint64_t offsetTableElements = 0;
    uint64_t offsetTableBytes = 0;
    if (MulOverflow(FP32_TRANSPOSE_LAYER_CAP, axisPad, offsetTableElements) ||
        MulOverflow(offsetTableElements, offsetBytes, offsetTableBytes)) {
        return false;
    }

    const uint64_t chunkCount = CeilDiv(indexSize, sourceChunk);
    // 首批 self 在复用 source staging 时需要完整 16 行容量；只有 source
    // chunk 不小于 padded axis，才能在不额外扩大 UB 的前提下承载这段暂存。
    if (sourceChunk < axisPad) {
        return false;
    }
    uint64_t headerBytes = 0;
    uint64_t tableSlotBytes = 0;
    uint64_t tableWorkspaceBytes = 0;
    uint64_t candidateWorkspaceBytes = 0;
    if (chunkCount == 0 || chunkCount > std::numeric_limits<uint32_t>::max() ||
        MulOverflow(chunkCount + 1, BLOCK_BYTES, headerBytes) ||
        offsetTableBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGNMENT - 1) ||
        MulOverflow(chunkCount, AlignUp(offsetTableBytes, WORKSPACE_ALIGNMENT), tableWorkspaceBytes) ||
        !AppendAlignedRegion(candidateWorkspaceBytes, headerBytes) ||
        !AppendAlignedRegion(candidateWorkspaceBytes, tableWorkspaceBytes)) {
        return false;
    }
    if (candidateWorkspaceBytes > std::numeric_limits<size_t>::max()) {
        return false;
    }
    tableSlotBytes = AlignUp(offsetTableBytes, WORKSPACE_ALIGNMENT);
    // payload 将 header 常驻 counterBuf，建表时借 dataBuf 首个 32B 槽发布；
    // 因而 header 不能超出二者任一容量。
    if (tableSlotBytes == 0 || headerBytes > accumulatorBytes || headerBytes > counterBytes) {
        return false;
    }

    tiling.set_transposeSourceChunk(static_cast<uint32_t>(sourceChunk));
    tiling.set_transposeLayerCap(static_cast<uint32_t>(FP32_TRANSPOSE_LAYER_CAP));
    tiling.set_transposePayloadLayerCap(static_cast<uint32_t>(payloadLayerCap));
    tiling.set_transposeOffsetBytes(offsetBytes);
    workspaceSize = static_cast<size_t>(candidateWorkspaceBytes);
    return true;
}

// FP32 窄 Gather 的尾部若跨越一个额外的 256B repeat，完整 UB repeat 可减少
// Gather/Add 的尾指令。这里刻意不扩展 workspace compact 表：compact 仍以逻辑轴宽
// DMA，kernel 在 UB 中原地展开其行距并补齐 source sentinel，避免把 Vector 收益
// 直接换成额外 GM 流量。
bool EnableGatherNarrowDbUbPaddedRepeat(IndexAddTilingData& tiling, ge::DataType dataType,
    uint64_t axisSize, uint64_t indexSize, uint64_t ubSize)
{
    if (tiling.get_strategy() != STRATEGY_GATHER_NARROW_DB || dataType != ge::DT_FLOAT ||
        axisSize == 0 || indexSize == 0 || tiling.get_sourceChunk() == 0 ||
        ubSize <= GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    constexpr uint64_t GATHER_ELEMENTS = VECTOR_BYTES / sizeof(float);
    if (GATHER_ELEMENTS == 0 || axisSize < 4 * GATHER_ELEMENTS) {
        return false;
    }
    uint64_t axisBytes = 0;
    if (MulOverflow(axisSize, sizeof(float), axisBytes) ||
        axisSize > std::numeric_limits<uint64_t>::max() - (GATHER_ELEMENTS - 1)) {
        return false;
    }
    const uint64_t baseRowElements = AlignUp(axisBytes, BLOCK_BYTES) / sizeof(float);
    const uint64_t paddedAxis = AlignUp(axisSize, GATHER_ELEMENTS);
    if (paddedAxis <= baseRowElements || paddedAxis > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    uint64_t indexBytes = 0;
    uint64_t sourceBytes = 0;
    uint64_t baseRowBytes = 0;
    uint64_t paddedRowBytes = 0;
    uint64_t compactBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) ||
        MulOverflow(indexSize, sizeof(float), sourceBytes) ||
        MulOverflow(baseRowElements, sizeof(uint32_t), baseRowBytes) ||
        MulOverflow(paddedAxis, sizeof(uint32_t), paddedRowBytes) ||
        MulOverflow(tiling.get_sourceChunk(), paddedRowBytes, compactBytes)) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);
    sourceBytes = AlignUp(sourceBytes, BLOCK_BYTES);
    if (baseRowBytes == 0 || paddedRowBytes == 0 || compactBytes == 0 ||
        sourceBytes > std::numeric_limits<uint32_t>::max() - BLOCK_BYTES) {
        return false;
    }

    uint64_t slotBytes = 0;
    uint64_t usedBytes = 0;
    if (!GetGatherDoubleBufferBytes(indexBytes, sourceBytes, paddedRowBytes, baseRowBytes,
            compactBytes, GATHER_PIPELINE_FP32_DENSE_SKEW_BYTES, slotBytes, usedBytes) ||
        slotBytes > std::numeric_limits<uint32_t>::max() ||
        usedBytes > ubSize - GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    tiling.set_gatherUbPaddedAxis(static_cast<uint32_t>(paddedAxis));
    return true;
}

bool SetBf16ResidentTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t innerSize, uint64_t indexSize, uint32_t coreNum, uint64_t usableUb)
{
    if (outerSize == 0 || axisSize == 0 || innerSize == 0 ||
        indexSize < MIN_RESIDENT_INDEX_ELEMENTS || coreNum == 0) {
        return false;
    }
    uint64_t indexBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) || indexBytes > usableUb) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);

    // 至少容纳一行 source，由 UB 容量先收敛搜索上界，避免随 innerSize 线性扫描。
    if (indexBytes >= usableUb || usableUb - indexBytes <= VECTOR_BYTES) {
        return false;
    }
    uint64_t bytesPerTileElement = 0;
    if (MulOverflow(axisSize, 2, bytesPerTileElement) ||
        AddOverflow(bytesPerTileElement, 14, bytesPerTileElement)) {
        return false;
    }
    const uint64_t maxFeasibleTile =
        (usableUb - indexBytes - VECTOR_BYTES) / bytesPerTileElement / 16 * 16;
    if (maxFeasibleTile == 0) {
        return false;
    }

    bool found = false;
    uint64_t bestTile = 0;
    uint64_t bestTileNum = 0;
    uint64_t bestChunk = 0;
    uint64_t bestTaskNum = 0;
    uint64_t bestWaves = std::numeric_limits<uint64_t>::max();
    uint64_t bestPanels = std::numeric_limits<uint64_t>::max();
    uint64_t bestVectorRepeats = std::numeric_limits<uint64_t>::max();
    uint64_t bestCoreDeficit = std::numeric_limits<uint64_t>::max();
    uint64_t bestRowBytes = std::numeric_limits<uint64_t>::max();

    const uint64_t maxTile =
        innerSize <= maxFeasibleTile ? AlignUp(innerSize, 16) : maxFeasibleTile;
    for (uint64_t tile = 16; tile <= maxTile; tile += 16) {
        const uint64_t bf16RowBytes = AlignUp(tile * sizeof(uint16_t), BLOCK_BYTES);
        const uint64_t fp32RowBytes = AlignUp(tile * sizeof(float), BLOCK_BYTES);
        uint64_t accumulatorBytes = 0;
        uint64_t fixedBytes = 0;
        uint64_t perSourceBytes = 0;
        if (MulOverflow(axisSize, bf16RowBytes, accumulatorBytes) ||
            AddOverflow(indexBytes, accumulatorBytes, fixedBytes) ||
            AddOverflow(fixedBytes, BF16_INDEPENDENT_WORK_ROWS * fp32RowBytes + VECTOR_BYTES, fixedBytes) ||
            AddOverflow(bf16RowBytes, fp32RowBytes, perSourceBytes) || fixedBytes >= usableUb) {
            continue;
        }
        const uint64_t sourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
            (usableUb - fixedBytes) / perSourceBytes});
        const uint64_t tileNum = CeilDiv(innerSize, tile);
        uint64_t taskNum = 0;
        if (sourceChunk != 0 && !MulOverflow(outerSize, tileNum, taskNum) && taskNum != 0 &&
            taskNum <= std::numeric_limits<uint32_t>::max()) {
            const uint64_t waves = CeilDiv(taskNum, coreNum);
            const uint64_t panels = CeilDiv(indexSize, sourceChunk);
            const uint64_t vectorRepeats = CeilDiv(tile, VECTOR_BYTES / sizeof(float));
            const uint64_t blockDim = std::min<uint64_t>(taskNum, coreNum);
            const uint64_t coreDeficit = coreNum - blockDim;
            const bool better = !found || waves < bestWaves ||
                (waves == bestWaves && panels < bestPanels) ||
                (waves == bestWaves && panels == bestPanels && vectorRepeats < bestVectorRepeats) ||
                (waves == bestWaves && panels == bestPanels && vectorRepeats == bestVectorRepeats &&
                    coreDeficit < bestCoreDeficit) ||
                (waves == bestWaves && panels == bestPanels && vectorRepeats == bestVectorRepeats &&
                    coreDeficit == bestCoreDeficit && bf16RowBytes < bestRowBytes);
            if (better) {
                found = true;
                bestTile = tile;
                bestTileNum = tileNum;
                bestChunk = sourceChunk;
                bestTaskNum = taskNum;
                bestWaves = waves;
                bestPanels = panels;
                bestVectorRepeats = vectorRepeats;
                bestCoreDeficit = coreDeficit;
                bestRowBytes = bf16RowBytes;
            }
        }
    }
    if (!found) {
        return false;
    }
    SetCoreTiling(tiling, bestTile, 1, 1, STRATEGY_BF16_RESIDENT,
        bestTileNum, outerSize, bestTaskNum, bestChunk);
    return true;
}

bool EnableBf16SourceDoubleBuffer(IndexAddTilingData& tiling, uint64_t axisSize,
    uint64_t indexSize, uint64_t usableUb)
{
    if (tiling.get_strategy() != STRATEGY_BF16_RESIDENT || indexSize < 2) {
        return false;
    }

    const uint64_t bf16RowBytes = AlignUp(tiling.get_innerTile() * sizeof(uint16_t), BLOCK_BYTES);
    const uint64_t fp32RowBytes = AlignUp(tiling.get_innerTile() * sizeof(float), BLOCK_BYTES);
    const uint64_t indexBytes = AlignUp(indexSize * sizeof(int32_t), BLOCK_BYTES);
    uint64_t accumulatorBytes = 0;
    uint64_t fixedBytes = indexBytes;
    uint64_t sourceBytes = 0;
    if (MulOverflow(axisSize, bf16RowBytes, accumulatorBytes) ||
        MulOverflow(indexSize, bf16RowBytes, sourceBytes) ||
        sourceBytes < 2 * GATHER_UB_RESERVE_BYTES ||
        AddOverflow(fixedBytes, accumulatorBytes, fixedBytes) ||
        AddOverflow(fixedBytes, BF16_INDEPENDENT_WORK_ROWS * fp32RowBytes + VECTOR_BYTES, fixedBytes) ||
        fixedBytes >= usableUb) {
        return false;
    }

    const uint64_t bytesPerSourceRow = 2 * bf16RowBytes + fp32RowBytes;
    uint64_t maxChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT,
        (usableUb - fixedBytes) / bytesPerSourceRow});
    if (maxChunk == 0) {
        return false;
    }

    const uint64_t panelNum = std::max<uint64_t>(2, CeilDiv(indexSize, maxChunk));
    const uint64_t sourceChunk = CeilDiv(indexSize, panelNum);
    uint64_t totalBytes = fixedBytes;
    uint64_t doubleBufferBytes = 0;
    uint64_t fp32SourceBytes = 0;
    if (sourceChunk >= indexSize || sourceChunk > maxChunk ||
        MulOverflow(2 * sourceChunk, bf16RowBytes, doubleBufferBytes) ||
        MulOverflow(sourceChunk, fp32RowBytes, fp32SourceBytes) ||
        AddOverflow(totalBytes, doubleBufferBytes, totalBytes) ||
        AddOverflow(totalBytes, fp32SourceBytes, totalBytes) || totalBytes > usableUb ||
        sourceChunk * bf16RowBytes < GATHER_UB_RESERVE_BYTES) {
        return false;
    }

    tiling.set_strategy(STRATEGY_BF16_SOURCE_DB);
    tiling.set_sourceChunk(static_cast<uint32_t>(sourceChunk));
    return true;
}

bool SetInt8AxisTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t innerSize, uint64_t indexSize, uint32_t coreNum, uint64_t usableUb,
    uint64_t residentInnerTile, uint64_t residentTaskNum)
{
    if (outerSize == 0 || axisSize < MIN_AXIS_TARGET_ROWS ||
        innerSize < MIN_INT8_INNER_ELEMENTS || indexSize < MIN_RESIDENT_INDEX_ELEMENTS || coreNum == 0) {
        return false;
    }
    const uint64_t compareElements = AlignUp(indexSize, VECTOR_BYTES / sizeof(int32_t));
    uint64_t indexBytes = 0;
    if (compareElements > std::numeric_limits<int32_t>::max() ||
        MulOverflow(compareElements, sizeof(int32_t), indexBytes) || indexBytes > usableUb) {
        return false;
    }
    const uint64_t compareMaskBytes = AlignUp(compareElements / 8, BLOCK_BYTES);
    const uint64_t maskGapBytes = std::max<uint64_t>(VECTOR_BYTES, compareMaskBytes);
    if (indexBytes >= usableUb || usableUb - indexBytes <= maskGapBytes) {
        return false;
    }

    // 每个任务只驻留一个目的行：raw + half + 2*int32 + float，共 15B/元素。
    constexpr uint64_t BYTES_PER_TILE_ELEMENT = 15;
    const uint64_t maxTile =
        (usableUb - indexBytes - maskGapBytes) / BYTES_PER_TILE_ELEMENT / BLOCK_BYTES * BLOCK_BYTES;
    if (maxTile == 0) {
        return false;
    }

    uint64_t outerAxisTasks = 0;
    uint64_t maxTaskNum = 0;
    if (MulOverflow(outerSize, axisSize, outerAxisTasks) || outerAxisTasks == 0 ||
        MulOverflow(coreNum, MAX_SEARCH_TASKS_PER_CORE, maxTaskNum)) {
        return false;
    }
    const uint64_t targetTile = innerSize;
    const uint64_t innerTile = targetTile <= maxTile ? AlignUp(targetTile, BLOCK_BYTES) : maxTile;
    const uint64_t innerTileNum = CeilDiv(innerSize, innerTile);
    uint64_t taskNum = 0;
    if (innerTile == 0 || MulOverflow(outerAxisTasks, innerTileNum, taskNum) ||
        taskNum <= residentTaskNum || residentTaskNum >= coreNum || taskNum > maxTaskNum ||
        taskNum > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    // 以均衡出现次数估算两条路径的关键核工作量，仅在轴向方案留有 2 倍裕量时切换。
    constexpr uint64_t INT32_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(int32_t);
    const uint64_t balancedOccurrences = CeilDiv(indexSize, axisSize);
    const uint64_t residentRepeats = CeilDiv(residentInnerTile, INT32_VECTOR_ELEMENTS);
    const uint64_t axisRepeats = CeilDiv(innerTile, INT32_VECTOR_ELEMENTS);
    const uint64_t residentWaves = CeilDiv(residentTaskNum, coreNum);
    const uint64_t axisWaves = CeilDiv(taskNum, coreNum);
    const uint64_t residentCost =
        residentWaves * residentRepeats * (7 * axisSize + 3 * indexSize) + residentTaskNum;
    const uint64_t axisCost =
        axisWaves * (axisRepeats * (7 + 3 * balancedOccurrences) + indexSize) + taskNum;
    if (axisCost > residentCost / 2) {
        return false;
    }

    SetCoreTiling(tiling, innerTile, 1, 1, STRATEGY_INT8_AXIS,
        innerTileNum, outerAxisTasks, taskNum, 1);
    return true;
}

bool SetInt8ResidentTiling(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t innerSize, uint64_t indexSize, uint32_t coreNum, uint64_t usableUb)
{
    if (outerSize == 0 || axisSize == 0 || innerSize < MIN_INT8_INNER_ELEMENTS ||
        indexSize < MIN_RESIDENT_INDEX_ELEMENTS || coreNum == 0) {
        return false;
    }
    uint64_t indexBytes = 0;
    if (MulOverflow(indexSize, sizeof(int32_t), indexBytes) || indexBytes > usableUb) {
        return false;
    }
    indexBytes = AlignUp(indexBytes, BLOCK_BYTES);

    // 最小 source chunk 为一行，据此推导 resident 布局可容纳的最大 tile。
    if (indexBytes >= usableUb || usableUb - indexBytes <= VECTOR_BYTES) {
        return false;
    }
    uint64_t bytesPerTileElement = 0;
    if (MulOverflow(axisSize, 11, bytesPerTileElement) ||
        AddOverflow(bytesPerTileElement, 4, bytesPerTileElement)) {
        return false;
    }
    const uint64_t maxFeasibleTile =
        (usableUb - indexBytes - VECTOR_BYTES) / bytesPerTileElement / BLOCK_BYTES * BLOCK_BYTES;
    if (maxFeasibleTile == 0) {
        return false;
    }

    bool found = false;
    uint64_t bestTile = 0;
    uint64_t bestTileNum = 0;
    uint64_t bestChunk = 0;
    uint64_t bestTaskNum = 0;
    uint64_t bestWaves = std::numeric_limits<uint64_t>::max();
    uint64_t bestPanels = std::numeric_limits<uint64_t>::max();
    uint64_t bestVectorRepeats = std::numeric_limits<uint64_t>::max();
    uint64_t bestCoreDeficit = std::numeric_limits<uint64_t>::max();

    const uint64_t maxTile =
        innerSize <= maxFeasibleTile ? AlignUp(innerSize, BLOCK_BYTES) : maxFeasibleTile;
    for (uint64_t tile = BLOCK_BYTES; tile <= maxTile; tile += BLOCK_BYTES) {
        const uint64_t int8RowBytes = AlignUp(tile, BLOCK_BYTES);
        const uint64_t halfRowBytes = AlignUp(tile * sizeof(uint16_t), BLOCK_BYTES);
        const uint64_t int32RowBytes = AlignUp(tile * sizeof(int32_t), BLOCK_BYTES);
        uint64_t fixedBytes = indexBytes;
        uint64_t axisIntBytes = 0;
        if (MulOverflow(axisSize, int32RowBytes, axisIntBytes) ||
            AddOverflow(fixedBytes, 2 * axisIntBytes + VECTOR_BYTES, fixedBytes) || fixedBytes >= usableUb) {
            continue;
        }
        const uint64_t rawHalfRowBytes = int8RowBytes + halfRowBytes;
        uint64_t axisRawHalfBytes = 0;
        if (MulOverflow(axisSize, rawHalfRowBytes, axisRawHalfBytes) ||
            axisRawHalfBytes > usableUb - fixedBytes) {
            continue;
        }
        uint64_t maxChunk = 0;
        const uint64_t remainingAfterAxisRaw = usableUb - fixedBytes - axisRawHalfBytes;
        const uint64_t withinAxisChunk = remainingAfterAxisRaw / int32RowBytes;
        if (withinAxisChunk < axisSize) {
            maxChunk = withinAxisChunk;
        } else {
            const uint64_t perSourceBytes = rawHalfRowBytes + int32RowBytes;
            maxChunk = (usableUb - fixedBytes) / perSourceBytes;
        }
        const uint64_t sourceChunk = std::min<uint64_t>({indexSize, MTE_BLOCK_COUNT_LIMIT, maxChunk});
        const uint64_t tileNum = CeilDiv(innerSize, tile);
        uint64_t taskNum = 0;
        if (sourceChunk != 0 && !MulOverflow(outerSize, tileNum, taskNum) && taskNum != 0 &&
            taskNum <= std::numeric_limits<uint32_t>::max()) {
            const uint64_t waves = CeilDiv(taskNum, coreNum);
            const uint64_t panels = CeilDiv(indexSize, sourceChunk);
            const uint64_t vectorRepeats = CeilDiv(tile, VECTOR_BYTES / sizeof(int32_t));
            const uint64_t blockDim = std::min<uint64_t>(taskNum, coreNum);
            const uint64_t coreDeficit = coreNum - blockDim;
            const bool better = !found || waves < bestWaves ||
                (waves == bestWaves && panels < bestPanels) ||
                (waves == bestWaves && panels == bestPanels && vectorRepeats < bestVectorRepeats) ||
                (waves == bestWaves && panels == bestPanels && vectorRepeats == bestVectorRepeats &&
                    coreDeficit < bestCoreDeficit);
            if (better) {
                found = true;
                bestTile = tile;
                bestTileNum = tileNum;
                bestChunk = sourceChunk;
                bestTaskNum = taskNum;
                bestWaves = waves;
                bestPanels = panels;
                bestVectorRepeats = vectorRepeats;
                bestCoreDeficit = coreDeficit;
            }
        }
    }
    if (!found) {
        return false;
    }
    SetCoreTiling(tiling, bestTile, 1, 1, STRATEGY_INT8_RESIDENT,
        bestTileNum, outerSize, bestTaskNum, bestChunk);
    return true;
}

void InitializeTilingDefaults(IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize,
    uint64_t innerSize, uint64_t indexSize, uint32_t elementSize)
{
    tiling.set_outerSize(outerSize);
    tiling.set_axisSize(axisSize);
    tiling.set_innerSize(innerSize);
    tiling.set_indexSize(indexSize);
    tiling.set_elementSize(elementSize);
    tiling.set_coopLayerCap(0);
    tiling.set_mixedInnerTile(0);
    tiling.set_mixedInnerTileNum(0);
    tiling.set_mixedOuterCount(0);
    tiling.set_residentBufferBytes(0);
    tiling.set_residentDoubleBuffer(0);
    tiling.set_gatherUbPaddedAxis(0);
    tiling.set_transposeBlockGather(0);
    tiling.set_transposeSourceChunk(0);
    tiling.set_transposeLayerCap(0);
    tiling.set_transposePayloadLayerCap(0);
    tiling.set_transposeOffsetBytes(0);
}

bool SelectBaseTiling(IndexAddTilingData& tiling, ge::DataType dataType,
    uint64_t outerSize, uint64_t axisSize, uint64_t innerSize, uint64_t indexSize,
    uint32_t elementSize, uint32_t coreNum, uint64_t ubSize, uint64_t usableUb, uint64_t residentUb)
{
    uint64_t innerBytes = 0;
    if (MulOverflow(innerSize, elementSize, innerBytes)) {
        return false;
    }
    bool tilingSuccess = false;
    if (dataType == ge::DT_BF16) {
        tilingSuccess = SetBf16ResidentTiling(
            tiling, outerSize, axisSize, innerSize, indexSize, coreNum, usableUb);
        if (tilingSuccess) {
            EnableBf16SourceDoubleBuffer(tiling, axisSize, indexSize,
                usableUb);
        }
    } else if (dataType == ge::DT_INT8) {
        tilingSuccess = SetInt8ResidentTiling(
            tiling, outerSize, axisSize, innerSize, indexSize, coreNum, usableUb);
        if (tilingSuccess) {
            SetInt8AxisTiling(tiling, outerSize, axisSize, innerSize, indexSize, coreNum, usableUb,
                tiling.get_innerTile(), tiling.get_taskNum());
        }
    }

    const bool supportsVectorAdd = dataType != ge::DT_BF16 && dataType != ge::DT_INT8;
    const bool supportsGatherNarrow =
        dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT || dataType == ge::DT_INT32;
    const bool scalarInnerTopology = IsScalarInnerTopology(innerSize, innerBytes, elementSize);
    const bool useGatherNarrowTopology = supportsGatherNarrow && scalarInnerTopology;
    if (useGatherNarrowTopology) {
        tilingSuccess = SetGatherNarrowTiling(
            tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize);
    }
    if (!tilingSuccess && useGatherNarrowTopology) {
        tilingSuccess = SetGatherBatchTiling(
            tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize);
        if (tilingSuccess) {
            TrySetGatherBatchWideTiling(
                tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize);
        }
    }
    if (supportsVectorAdd && scalarInnerTopology && !tilingSuccess) {
        tilingSuccess = SetNarrowUbTiling(
            tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize);
    }
    if (!tilingSuccess && supportsVectorAdd && innerBytes >= VECTOR_BYTES) {
        tilingSuccess = SetUbTiling(
            tiling, outerSize, axisSize, innerSize, indexSize, elementSize, coreNum, residentUb);
    }
    if (!tilingSuccess) {
        tilingSuccess = SetAtomicTiling(
            tiling, outerSize, innerSize, indexSize, elementSize, coreNum,
            usableUb);
    }
    return tilingSuccess;
}

bool FinalizeTiling(gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform,
    IndexAddTilingData& tiling, uint64_t outerSize, uint64_t axisSize, uint64_t innerSize,
    uint64_t indexSize, ge::DataType dataType, uint32_t elementSize, uint32_t coreNum,
    uint64_t ubSize, uint64_t residentUb)
{
    if (tiling.get_strategy() == STRATEGY_UB) {
        EnableMixedResidentTiling(
            tiling, outerSize, axisSize, innerSize, indexSize, elementSize, coreNum, residentUb);
        EnableMixedResidentDoubleBufferTiling(
            tiling, axisSize, indexSize, elementSize, residentUb);
    }

    size_t cooperativeWorkspaceSize = 0;
    bool useCooperativeGather = SetCooperativeGatherWorkspace(
        tiling, axisSize, indexSize, dataType, elementSize, coreNum, ubSize, cooperativeWorkspaceSize);
    if (tiling.get_strategy() == STRATEGY_GATHER_BATCH_WIDE && !useCooperativeGather) {
        if (!SetGatherBatchTiling(
                tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize)) {
            return false;
        }
        TrySetGatherBatchWideTiling(
            tiling, outerSize, axisSize, indexSize, elementSize, coreNum, ubSize);
        cooperativeWorkspaceSize = 0;
        useCooperativeGather = SetCooperativeGatherWorkspace(
            tiling, axisSize, indexSize, dataType, elementSize, coreNum, ubSize,
            cooperativeWorkspaceSize);
    }

    uint32_t transposeBlockGatherMode = 0;
    if (useCooperativeGather) {
        EnableGatherNarrowDbUbPaddedRepeat(tiling, dataType, axisSize, indexSize, ubSize);
        if (EnableFp32TransposeBlockGather(
                tiling, dataType, axisSize, indexSize, ubSize, cooperativeWorkspaceSize)) {
            transposeBlockGatherMode = 2;
        } else if (EnableTransposeBlockGather(
                tiling, dataType, axisSize, indexSize, ubSize)) {
            transposeBlockGatherMode = 1;
        }
    }
    tiling.set_transposeBlockGather(transposeBlockGatherMode);

    uint64_t l2Size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    const bool useSourceL2Bypass = ShouldBypassResidentSourceL2(
        tiling.get_strategy(), outerSize, indexSize, innerSize, elementSize, l2Size);
    const uint32_t blockDim = useCooperativeGather ? coreNum : std::min(coreNum, tiling.get_taskNum());
    tiling.set_blockDim(blockDim);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(GET_TPL_TILING_KEY(useCooperativeGather ? COOP_TILING_KEY :
        (useSourceL2Bypass ? SOURCE_L2_BYPASS_TILING_KEY : 0)));
    if (useCooperativeGather) {
        context->SetScheduleMode(1);
    }

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    auto* rawTiling = context->GetRawTilingData();
    if (workspaceSize == nullptr || rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return false;
    }
    if (useCooperativeGather) {
        const size_t systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
        if (cooperativeWorkspaceSize > std::numeric_limits<size_t>::max() - systemWorkspaceSize) {
            return false;
        }
        workspaceSize[0] = systemWorkspaceSize + cooperativeWorkspaceSize;
    } else {
        workspaceSize[0] = 0;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return true;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outerSize = 0;
    uint64_t axisSize = 0;
    uint64_t innerSize = 0;
    uint64_t indexSize = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint32_t elementSize = 0;
    if (!GetLogicalShape(context, outerSize, axisSize, innerSize, indexSize, dataType, elementSize)) {
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || ubSize < BLOCK_BYTES) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t usableUb = ubSize / UB_USE_DENOMINATOR * UB_USE_NUMERATOR / BLOCK_BYTES * BLOCK_BYTES;
    // 普通 resident 只有三块精确可计量的 TBuf，保留一个 Vector block 避免占满 UB。
    const uint64_t residentUb = ubSize > VECTOR_BYTES ?
        (ubSize - VECTOR_BYTES) / BLOCK_BYTES * BLOCK_BYTES : usableUb;

    IndexAddTilingData tiling;
    InitializeTilingDefaults(tiling, outerSize, axisSize, innerSize, indexSize, elementSize);
    const bool tilingSuccess = SelectBaseTiling(tiling, dataType, outerSize, axisSize, innerSize,
        indexSize, elementSize, coreNum, ubSize, usableUb, residentUb);
    if (!tilingSuccess) {
        return ge::GRAPH_FAILED;
    }
    return FinalizeTiling(context, platform, tiling, outerSize, axisSize, innerSize, indexSize,
        dataType, elementSize, coreNum, ubSize, residentUb) ?
        ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(IndexAdd);
}
