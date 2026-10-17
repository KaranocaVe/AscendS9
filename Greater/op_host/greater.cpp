#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace {

constexpr uint8_t SELF_BROADCAST = 1;
constexpr uint8_t OTHER_BROADCAST = 2;
constexpr uint64_t SELECT_TMP_BYTES = 8 * 1024;
constexpr uint64_t BUFFER_PHASE_BYTES = 256;
constexpr uint64_t UB_DATA_BLOCK_BYTES = 32;
constexpr uint32_t TILING_VERSION = 4;
constexpr uint32_t MAX_REPEAT_BLOCK_COUNT = 4095;

struct CollapsedTopology {
    uint32_t rank{0};
    uint64_t totalElements{1};
    std::array<uint64_t, optiling::GREATER_MAX_RANK> dims{};
    std::array<uint64_t, optiling::GREATER_MAX_RANK> selfStrides{};
    std::array<uint64_t, optiling::GREATER_MAX_RANK> otherStrides{};
    std::array<uint8_t, optiling::GREATER_MAX_RANK> flags{};
};

struct DataTypePlan {
    uint32_t alignment{0};
    uint32_t rawBytes{0};
    uint32_t calcBytes{0};
    uint32_t persistentBytes{0};
    uint32_t linearBufferNum{2};
    uint64_t rowReserveBytes{SELECT_TMP_BYTES};
};

struct RowLayout {
    uint64_t outer{1};
    uint64_t repeat{1};
    uint64_t inner{1};
    uint8_t broadcastFlag{0};
};

struct PartitionStats {
    uint64_t mte2Bytes{0};
    uint64_t mte3Bytes{0};
    uint64_t vectorElements{0};
    uint64_t dmaCommands{0};
    uint64_t fragments{0};
    uint64_t tailBytes{0};
};

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

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result)
{
    if (alignment == 0) {
        result = value;
        return true;
    }
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    const uint64_t increment = alignment - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - increment) {
        return false;
    }
    result = value + increment;
    return true;
}

bool ResolveBroadcastDim(int64_t selfDim, int64_t otherDim, uint64_t& outputDim)
{
    if (selfDim < 0 || otherDim < 0) {
        return false;
    }
    if (selfDim == otherDim) {
        outputDim = static_cast<uint64_t>(selfDim);
        return true;
    }
    if (selfDim == 1) {
        outputDim = static_cast<uint64_t>(otherDim);
        return true;
    }
    if (otherDim == 1) {
        outputDim = static_cast<uint64_t>(selfDim);
        return true;
    }
    return false;
}

bool BuildCollapsedTopology(const gert::Shape& selfShape, const gert::Shape& otherShape,
    CollapsedTopology& topology)
{
    const size_t selfRank = selfShape.GetDimNum();
    const size_t otherRank = otherShape.GetDimNum();
    const size_t outputRank = std::max(selfRank, otherRank);
    if (outputRank > optiling::GREATER_MAX_RANK) {
        return false;
    }

    topology = CollapsedTopology{};
    if (outputRank == 0) {
        topology.rank = 1;
        topology.dims[0] = 1;
        topology.selfStrides[0] = 1;
        topology.otherStrides[0] = 1;
        return true;
    }

    for (size_t axis = 0; axis < outputRank; ++axis) {
        const size_t selfOffset = outputRank - selfRank;
        const size_t otherOffset = outputRank - otherRank;
        const int64_t selfDim = axis < selfOffset ? 1 : selfShape.GetDim(axis - selfOffset);
        const int64_t otherDim = axis < otherOffset ? 1 : otherShape.GetDim(axis - otherOffset);
        uint64_t outputDim = 0;
        if (!ResolveBroadcastDim(selfDim, otherDim, outputDim)) {
            return false;
        }

        uint8_t flag = 0;
        if (selfDim == 1 && outputDim != 1) {
            flag |= SELF_BROADCAST;
        }
        if (otherDim == 1 && outputDim != 1) {
            flag |= OTHER_BROADCAST;
        }

        if (topology.rank != 0 && topology.flags[topology.rank - 1] == flag) {
            uint64_t mergedDim = 0;
            if (MulOverflow(topology.dims[topology.rank - 1], outputDim, mergedDim)) {
                return false;
            }
            topology.dims[topology.rank - 1] = mergedDim;
        } else {
            topology.dims[topology.rank] = outputDim;
            topology.flags[topology.rank] = flag;
            ++topology.rank;
        }
    }

    uint64_t outputSuffix = 1;
    uint64_t selfSuffix = 1;
    uint64_t otherSuffix = 1;
    for (uint32_t axis = topology.rank; axis > 0;) {
        --axis;
        const bool selfBroadcast = (topology.flags[axis] & SELF_BROADCAST) != 0;
        const bool otherBroadcast = (topology.flags[axis] & OTHER_BROADCAST) != 0;
        topology.selfStrides[axis] = selfBroadcast ? 0 : selfSuffix;
        topology.otherStrides[axis] = otherBroadcast ? 0 : otherSuffix;
        if ((!selfBroadcast && MulOverflow(selfSuffix, topology.dims[axis], selfSuffix)) ||
            (!otherBroadcast && MulOverflow(otherSuffix, topology.dims[axis], otherSuffix)) ||
            MulOverflow(outputSuffix, topology.dims[axis], outputSuffix)) {
            return false;
        }
    }
    topology.totalElements = outputSuffix;
    return true;
}

bool GetDataTypePlan(ge::DataType dataType, DataTypePlan& plan)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            plan.alignment = 128;
            plan.rawBytes = 2;
            plan.calcBytes = 0;
            plan.persistentBytes = 0;
            plan.linearBufferNum = 1;
            plan.rowReserveBytes = SELECT_TMP_BYTES;
            return true;
        case ge::DT_FLOAT:
            plan.alignment = 64;
            plan.rawBytes = 4;
            plan.calcBytes = 0;
            plan.persistentBytes = 0;
            plan.linearBufferNum = 1;
            plan.rowReserveBytes = SELECT_TMP_BYTES;
            return true;
        case ge::DT_BF16:
            plan.alignment = 64;
            plan.rawBytes = 2;
            plan.calcBytes = 4;
            plan.persistentBytes = 0;
            plan.linearBufferNum = 1;
            plan.rowReserveBytes = SELECT_TMP_BYTES;
            return true;
        case ge::DT_INT8:
            plan.alignment = 128;
            plan.rawBytes = 1;
            plan.calcBytes = 2;
            plan.persistentBytes = 0;
            plan.linearBufferNum = 1;
            plan.rowReserveBytes = SELECT_TMP_BYTES;
            return true;
        case ge::DT_INT32:
            plan.alignment = 64;
            plan.rawBytes = 4;
            plan.calcBytes = 0;
            plan.persistentBytes = 2;
            plan.linearBufferNum = 1;
            // INT32 行路径的 Select 使用已分配的 full/zero buffer，不需要通用临时预留。
            plan.rowReserveBytes = 0;
            return true;
        default:
            return false;
    }
}

bool IsLinear(const CollapsedTopology& topology)
{
    for (uint32_t axis = 0; axis < topology.rank; ++axis) {
        if (topology.flags[axis] != 0) {
            return false;
        }
    }
    return true;
}

bool MatchResidentRow(const CollapsedTopology& topology, RowLayout& layout)
{
    if (topology.rank < 2) {
        return false;
    }
    int32_t broadcastAxis = -1;
    uint8_t broadcastFlag = 0;
    for (uint32_t axis = 0; axis < topology.rank; ++axis) {
        if (topology.flags[axis] == 0) {
            continue;
        }
        if (topology.flags[axis] != SELF_BROADCAST &&
            topology.flags[axis] != OTHER_BROADCAST) {
            return false;
        }
        if (broadcastAxis >= 0) {
            return false;
        }
        broadcastAxis = static_cast<int32_t>(axis);
        broadcastFlag = topology.flags[axis];
    }
    if (broadcastAxis < 0 || broadcastAxis + 1 >= static_cast<int32_t>(topology.rank)) {
        return false;
    }

    uint64_t outer = 1;
    for (int32_t axis = 0; axis < broadcastAxis; ++axis) {
        if (MulOverflow(outer, topology.dims[axis], outer)) {
            return false;
        }
    }
    uint64_t inner = 1;
    for (uint32_t axis = static_cast<uint32_t>(broadcastAxis + 1); axis < topology.rank; ++axis) {
        if (MulOverflow(inner, topology.dims[axis], inner)) {
            return false;
        }
    }
    layout.outer = outer;
    layout.repeat = topology.dims[broadcastAxis];
    layout.inner = inner;
    layout.broadcastFlag = broadcastFlag;
    return true;
}

uint64_t AlignedBytes(uint64_t elements, uint32_t bytesPerElement)
{
    uint64_t bytes = 0;
    if (MulOverflow(elements, bytesPerElement, bytes)) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t aligned = 0;
    if (!AlignUp(bytes, 32, aligned)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return aligned;
}

uint64_t TileTransferBytes(uint64_t inner, uint32_t tileElements, uint32_t bytesPerElement)
{
    if (inner == 0 || tileElements == 0) {
        return 0;
    }
    const uint64_t fullTiles = inner / tileElements;
    const uint64_t tail = inner % tileElements;
    uint64_t fullBytes = 0;
    if (MulOverflow(fullTiles, AlignedBytes(tileElements, bytesPerElement), fullBytes)) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t tailBytes = tail == 0 ? 0 : AlignedBytes(tail, bytesPerElement);
    if (AddOverflow(fullBytes, tailBytes, fullBytes)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return fullBytes;
}

bool AppendBuffer(uint64_t& cursor, uint64_t bytes)
{
    uint64_t aligned = 0;
    if (!AlignUp(cursor, BUFFER_PHASE_BYTES, aligned) ||
        AddOverflow(aligned, bytes, cursor) ||
        AddOverflow(cursor, BUFFER_PHASE_BYTES, cursor)) {
        return false;
    }
    return true;
}

bool LinearFits(uint64_t ubSize, uint32_t tileElements, const DataTypePlan& plan)
{
    uint64_t cursor = SELECT_TMP_BYTES;
    const uint64_t rawBytes = AlignedBytes(tileElements, plan.rawBytes);
    const uint64_t outputBytes = AlignedBytes(tileElements, 1);
    const uint64_t calcBytes = AlignedBytes(tileElements, plan.calcBytes);
    if (rawBytes == std::numeric_limits<uint64_t>::max() ||
        outputBytes == std::numeric_limits<uint64_t>::max() ||
        (plan.calcBytes != 0 && calcBytes == std::numeric_limits<uint64_t>::max())) {
        return false;
    }

    if (plan.calcBytes == 0 && plan.rawBytes != 1 &&
        !AppendBuffer(cursor, BUFFER_PHASE_BYTES)) {
        return false;
    }
    for (uint32_t inputQueue = 0; inputQueue < 2; ++inputQueue) {
        for (uint32_t buffer = 0; buffer < plan.linearBufferNum; ++buffer) {
            if (!AppendBuffer(cursor, rawBytes)) {
                return false;
            }
        }
    }
    for (uint32_t buffer = 0; buffer < plan.linearBufferNum; ++buffer) {
        if (!AppendBuffer(cursor, outputBytes)) {
            return false;
        }
    }
    return
        (plan.calcBytes == 0 || (AppendBuffer(cursor, calcBytes) &&
            AppendBuffer(cursor, calcBytes))) &&
        cursor <= ubSize;
}

bool RowFits(uint64_t ubSize, uint32_t tileElements, uint32_t rowBatch,
    const DataTypePlan& plan)
{
    uint64_t cursor = plan.rowReserveBytes;
    uint64_t residentBytes = 0;
    if (MulOverflow(tileElements, plan.rawBytes, residentBytes) ||
        AddOverflow(residentBytes, BUFFER_PHASE_BYTES, residentBytes)) {
        return false;
    }
    const uint64_t residentCalcBytes = AlignedBytes(tileElements, plan.calcBytes);
    const uint64_t fullBytes = AlignedBytes(
        static_cast<uint64_t>(tileElements) * rowBatch, plan.rawBytes);
    const uint64_t outputBytes = AlignedBytes(
        static_cast<uint64_t>(tileElements) * rowBatch, 1);
    if (residentCalcBytes == std::numeric_limits<uint64_t>::max() ||
        fullBytes == std::numeric_limits<uint64_t>::max() ||
        outputBytes == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    if (!AppendBuffer(cursor, residentBytes) ||
        !AppendBuffer(cursor, residentCalcBytes) ||
        !AppendBuffer(cursor, fullBytes) ||
        !AppendBuffer(cursor, fullBytes) ||
        (plan.calcBytes != 0 &&
            (!AppendBuffer(cursor, residentCalcBytes) ||
             !AppendBuffer(cursor, residentCalcBytes))) ||
        !AppendBuffer(cursor, outputBytes) ||
        !AppendBuffer(cursor, outputBytes)) {
        return false;
    }
    if (plan.persistentBytes != 0 &&
        !AppendBuffer(cursor, AlignedBytes(tileElements, plan.persistentBytes))) {
        return false;
    }
    return cursor <= ubSize;
}

uint32_t GreatestCommonDivisor(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

// INT32 行宽不是 32B 整数倍时，最小相位周期内的 GM 数据重新回到 32B 边界。
// 该路径将这一整相位包连续搬入，再在 UB 中把常驻行展开为同一紧凑布局。
bool PackedResidentRowFits(uint64_t ubSize, const DataTypePlan& plan, uint64_t inner,
    uint64_t repeat, uint32_t tileElements, uint32_t& packetRowsOut)
{
    packetRowsOut = 0;
    if (plan.rawBytes != sizeof(int32_t) || plan.persistentBytes != 2 ||
        inner == 0 || inner > tileElements || tileElements % 8 != 0 ||
        inner > std::numeric_limits<uint32_t>::max() / sizeof(int32_t)) {
        return false;
    }

    const uint32_t rowBytes = static_cast<uint32_t>(inner) * sizeof(int32_t);
    const uint32_t phaseRows = 32 / GreatestCommonDivisor(rowBytes, 32);
    if (phaseRows <= 1 || repeat < phaseRows || repeat % phaseRows != 0) {
        return false;
    }

    // DataCopyPad 可处理非 32B 整除的连续包。选择不超过 64 行的最大 repeat
    // 因子，以减少 MTE2 发射，同时避免超大的单次搬运和尾包分支。
    const uint64_t maxCandidate = std::min<uint64_t>(repeat, 64);
    for (uint64_t candidate = maxCandidate; candidate >= phaseRows; --candidate) {
        // 每个连续包的 GM 起址必须回到 32B 边界，否则 DAV_2201 的
        // DataCopyPad 会对后续包产生相位错位。
        if (repeat % candidate != 0 || candidate % phaseRows != 0) {
            continue;
        }
        uint64_t packetElements = 0;
        uint64_t paddedPacketElements = 0;
        if (MulOverflow(candidate, inner, packetElements) ||
            MulOverflow(candidate, tileElements, paddedPacketElements) ||
            packetElements > std::numeric_limits<uint32_t>::max() ||
            paddedPacketElements > std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        uint64_t computeElements = 0;
        if (!AlignUp(packetElements, 64, computeElements) ||
            computeElements > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        uint64_t residentBytes = 0;
        uint64_t packetBytes = 0;
        uint64_t paddedPacketBytes = 0;
        if (MulOverflow(tileElements, sizeof(int32_t), residentBytes) ||
            MulOverflow(computeElements, sizeof(int32_t), packetBytes) ||
            MulOverflow(paddedPacketElements, sizeof(int32_t), paddedPacketBytes)) {
            continue;
        }
        // 与 kernel 的实际分配一一对应：resident、两路连续 full、两路 bool 输出、
        // 常驻行的 padded/packed 展开以及 Broadcast 的模式暂存。
        uint64_t cursor = 0;
        if (!AppendBuffer(cursor, residentBytes) ||
            !AppendBuffer(cursor, packetBytes) ||
            !AppendBuffer(cursor, packetBytes) ||
            !AppendBuffer(cursor, computeElements) ||
            !AppendBuffer(cursor, computeElements) ||
            !AppendBuffer(cursor, paddedPacketBytes + 2 * BUFFER_PHASE_BYTES) ||
            !AppendBuffer(cursor, packetBytes) ||
            !AppendBuffer(cursor, BUFFER_PHASE_BYTES)) {
            continue;
        }
        if (cursor <= ubSize) {
            packetRowsOut = static_cast<uint32_t>(candidate);
            return true;
        }
    }
    return false;
}

// INT32 行宽未对齐时，按 32B 行距将 full 行直接搬入 UB。每个 full 行保留一个
// 额外 DataBlock 间隔，既满足 DataCopyPad 的独立行布局，也让 BinaryRepeat 的
// 物理行距与 resident 的零步长复用分离。选择与最小相位周期一致的 packet，避免
// 在 repeat 末尾引入额外分支。
bool PitchedResidentRowFits(uint64_t ubSize, const DataTypePlan& plan, uint64_t inner,
    uint64_t repeat, uint32_t tileElements, uint32_t& packetRowsOut)
{
    packetRowsOut = 0;
    if (plan.rawBytes != sizeof(int32_t) || plan.persistentBytes != 2 || inner == 0 ||
        inner > tileElements || inner > std::numeric_limits<uint32_t>::max() / sizeof(int32_t)) {
        return false;
    }

    const uint32_t rowBytes = static_cast<uint32_t>(inner) * sizeof(int32_t);
    const uint32_t phaseRows = 32 / GreatestCommonDivisor(rowBytes, 32);
    if (phaseRows <= 1 || repeat < static_cast<uint64_t>(phaseRows) * 2 ||
        repeat % phaseRows != 0 || phaseRows > MAX_REPEAT_BLOCK_COUNT) {
        return false;
    }

    uint64_t computeElements = 0;
    uint64_t fullPitchElements = 0;
    uint64_t outputPitchElements = 0;
    if (!AlignUp(inner, 64, computeElements) ||
        !AddOverflow(computeElements, 8, fullPitchElements) ||
        !AlignUp(inner, 32, outputPitchElements) ||
        !AddOverflow(outputPitchElements, 32, outputPitchElements) ||
        computeElements > std::numeric_limits<uint32_t>::max() ||
        fullPitchElements > std::numeric_limits<uint32_t>::max() ||
        outputPitchElements > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    // BinaryRepeatParams 的 repeat stride 为 uint8_t，且每个 INT32 行必须落在
    // 255 个 DataBlock 以内；输出 bool 行同样受该约束。
    if (fullPitchElements % 8 != 0 || outputPitchElements % 32 != 0 ||
        fullPitchElements / 8 > std::numeric_limits<uint8_t>::max() ||
        outputPitchElements / 32 > std::numeric_limits<uint8_t>::max() ||
        phaseRows > std::numeric_limits<uint8_t>::max()) {
        return false;
    }

    // BinaryRepeatParams 的 batch 是 uint8_t；从 repeat 的合法相位倍数中，
    // 选择能容纳在 UB 的最大包，减少 full/output 的队列发射次数。包长只由
    // 行宽相位、repeat 和实际 UB 账本决定，尾包允许小于该值。
    const uint64_t maxCandidate = std::min<uint64_t>(
        std::min<uint64_t>(repeat, MAX_REPEAT_BLOCK_COUNT),
        std::numeric_limits<uint8_t>::max());
    const uint64_t residentBytes = static_cast<uint64_t>(computeElements) * sizeof(int32_t);
    const uint64_t zeroBytes = static_cast<uint64_t>(computeElements) * sizeof(uint16_t);
    for (uint64_t candidate = maxCandidate; candidate >= phaseRows; --candidate) {
        if (candidate % phaseRows != 0) {
            continue;
        }
        uint64_t fullSlotBytes = 0;
        uint64_t outputSlotBytes = 0;
        if (MulOverflow(candidate, fullPitchElements, fullSlotBytes) ||
            MulOverflow(fullSlotBytes, sizeof(int32_t), fullSlotBytes) ||
            MulOverflow(candidate, outputPitchElements, outputSlotBytes) ||
            MulOverflow(outputSlotBytes, sizeof(uint8_t), outputSlotBytes)) {
            continue;
        }

        // 与 KernelResidentPitchedRow 的实际队列和 zero buffer 一一对应。full/output
        // 各保留两个槽，确保搬入、计算和写回可以持续重叠。
        uint64_t cursor = 0;
        if (!AppendBuffer(cursor, residentBytes) ||
            !AppendBuffer(cursor, fullSlotBytes) ||
            !AppendBuffer(cursor, fullSlotBytes) ||
            !AppendBuffer(cursor, outputSlotBytes) ||
            !AppendBuffer(cursor, outputSlotBytes) ||
            !AppendBuffer(cursor, zeroBytes)) {
            continue;
        }
        if (cursor <= ubSize) {
            packetRowsOut = static_cast<uint32_t>(candidate);
            return true;
        }
    }
    return false;
}

template <typename FitFn>
uint32_t FindMaxAligned(uint64_t ubSize, uint32_t alignment, FitFn&& fits)
{
    if (alignment == 0) {
        return 0;
    }
    uint64_t maxUnits = std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max() / alignment,
        ubSize / alignment);
    uint64_t low = 1;
    uint64_t high = maxUnits;
    uint64_t best = 0;
    while (low <= high) {
        const uint64_t mid = low + (high - low) / 2;
        const uint32_t candidate = static_cast<uint32_t>(mid * alignment);
        if (fits(candidate)) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return static_cast<uint32_t>(best * alignment);
}

bool BuildLinearTiling(uint64_t ubSize, const DataTypePlan& plan, uint64_t totalElements,
    uint32_t& tileElements)
{
    const uint32_t capacity = FindMaxAligned(
        ubSize, plan.alignment, [&](uint32_t candidate) {
            return LinearFits(ubSize, candidate, plan);
        });
    if (capacity < plan.alignment) {
        return false;
    }
    uint64_t alignedTotal = 0;
    if (!AlignUp(totalElements, plan.alignment, alignedTotal)) {
        return false;
    }
    tileElements = static_cast<uint32_t>(std::min<uint64_t>(
        capacity, std::max<uint64_t>(plan.alignment, alignedTotal)));
    return true;
}

bool BuildRowTiling(uint64_t ubSize, const DataTypePlan& plan, uint64_t inner,
    uint64_t repeat, uint32_t& tileElements, uint32_t& rowBatch)
{
    uint32_t capacity = FindMaxAligned(
        ubSize, plan.alignment, [&](uint32_t candidate) {
            return RowFits(ubSize, candidate, 1, plan);
        });
    if (capacity < plan.alignment) {
        return false;
    }

    uint64_t alignedInner = 0;
    if (!AlignUp(inner, plan.alignment, alignedInner)) {
        return false;
    }
    tileElements = static_cast<uint32_t>(std::min<uint64_t>(
        capacity, std::max<uint64_t>(plan.alignment, alignedInner)));

    // 当整行小于候选 tile 时，尾行的 32B 输出对齐也必须落在 UB 行 pitch 内。
    // 例如 INT32 的 inner=1023 若直接使用 1024 elements，输出需要 1056B，
    // 会使 batch DMA 的源 stride 下溢；缩到最近的合法对齐块即可安全覆盖尾行。
    if (inner != 0 && inner < tileElements &&
        AlignUp(inner, 32, alignedInner) && alignedInner > tileElements) {
        const uint64_t safeTile = AlignDown(inner, plan.alignment);
        tileElements = static_cast<uint32_t>(
            safeTile >= plan.alignment ? safeTile : plan.alignment);
    }

    const uint32_t maxBatch = static_cast<uint32_t>(
        std::min<uint64_t>(std::min<uint64_t>(repeat, MAX_REPEAT_BLOCK_COUNT),
            std::numeric_limits<uint32_t>::max()));
    rowBatch = 1;
    for (uint32_t candidate = maxBatch; candidate > 0; --candidate) {
        if (RowFits(ubSize, tileElements, candidate, plan)) {
            rowBatch = candidate;
            break;
        }
    }
    return RowFits(ubSize, tileElements, rowBatch, plan);
}

bool LessStats(const PartitionStats& lhs, const PartitionStats& rhs)
{
    if (lhs.mte2Bytes != rhs.mte2Bytes) {
        return lhs.mte2Bytes < rhs.mte2Bytes;
    }
    if (lhs.mte3Bytes != rhs.mte3Bytes) {
        return lhs.mte3Bytes < rhs.mte3Bytes;
    }
    if (lhs.vectorElements != rhs.vectorElements) {
        return lhs.vectorElements < rhs.vectorElements;
    }
    if (lhs.dmaCommands != rhs.dmaCommands) {
        return lhs.dmaCommands < rhs.dmaCommands;
    }
    if (lhs.fragments != rhs.fragments) {
        return lhs.fragments < rhs.fragments;
    }
    return lhs.tailBytes < rhs.tailBytes;
}

void MaxStats(PartitionStats& target, const PartitionStats& value)
{
    target.mte2Bytes = std::max(target.mte2Bytes, value.mte2Bytes);
    target.mte3Bytes = std::max(target.mte3Bytes, value.mte3Bytes);
    target.vectorElements = std::max(target.vectorElements, value.vectorElements);
    target.dmaCommands = std::max(target.dmaCommands, value.dmaCommands);
    target.fragments = std::max(target.fragments, value.fragments);
    target.tailBytes = std::max(target.tailBytes, value.tailBytes);
}

uint64_t FragmentBatchCount(uint64_t rowStart, uint64_t rowEnd, uint64_t repeat,
    uint32_t rowBatch)
{
    if (rowStart >= rowEnd || repeat == 0 || rowBatch == 0) {
        return 0;
    }
    const uint64_t firstGroup = rowStart / repeat;
    const uint64_t lastGroup = (rowEnd - 1) / repeat;
    if (firstGroup == lastGroup) {
        return CeilDiv(rowEnd - rowStart, rowBatch);
    }
    const uint64_t firstRows = (firstGroup + 1) * repeat - rowStart;
    const uint64_t lastRows = rowEnd - lastGroup * repeat;
    uint64_t batches = CeilDiv(firstRows, rowBatch) + CeilDiv(lastRows, rowBatch);
    if (lastGroup > firstGroup + 1) {
        batches += (lastGroup - firstGroup - 1) * CeilDiv(repeat, rowBatch);
    }
    return batches;
}

bool BuildPartitionStats(const RowLayout& layout, uint32_t tileElements, uint32_t rowBatch,
    uint32_t activeCores, uint32_t partitionMode, uint32_t rawBytes,
    PartitionStats& stats)
{
    if (activeCores == 0 || tileElements == 0 || rowBatch == 0) {
        return false;
    }
    const uint64_t tileCount = CeilDiv(layout.inner, tileElements);
    const uint64_t inputRowBytes = TileTransferBytes(layout.inner, tileElements, rawBytes);
    const uint64_t outputRowBytes = TileTransferBytes(layout.inner, tileElements, 1);
    if (inputRowBytes == std::numeric_limits<uint64_t>::max() ||
        outputRowBytes == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    uint64_t totalRows = 0;
    if (MulOverflow(layout.outer, layout.repeat, totalRows)) {
        return false;
    }

    for (uint32_t core = 0; core < activeCores; ++core) {
        uint64_t rows = 0;
        uint64_t fragments = 0;
        uint64_t batches = 0;
        if (partitionMode == optiling::GREATER_PARTITION_GROUP_MAJOR) {
            const uint64_t groupStart = layout.outer * core / activeCores;
            const uint64_t groupEnd = layout.outer * (core + 1) / activeCores;
            const uint64_t groups = groupEnd - groupStart;
            if (MulOverflow(groups, layout.repeat, rows)) {
                return false;
            }
            fragments = groups;
            if (MulOverflow(groups, CeilDiv(layout.repeat, rowBatch), batches)) {
                return false;
            }
        } else {
            const uint64_t rowStart = totalRows * core / activeCores;
            const uint64_t rowEnd = totalRows * (core + 1) / activeCores;
            rows = rowEnd - rowStart;
            fragments = rowStart >= rowEnd ? 0 :
                (rowEnd - 1) / layout.repeat - rowStart / layout.repeat + 1;
            batches = FragmentBatchCount(rowStart, rowEnd, layout.repeat, rowBatch);
        }

        PartitionStats current;
        uint64_t fullBytes = 0;
        uint64_t residentBytes = 0;
        uint64_t outputBytes = 0;
        if (MulOverflow(rows, inputRowBytes, fullBytes) ||
            MulOverflow(fragments, tileCount, residentBytes) ||
            MulOverflow(residentBytes, inputRowBytes, residentBytes) ||
            MulOverflow(rows, outputRowBytes, outputBytes)) {
            return false;
        }
        current.mte2Bytes = fullBytes + residentBytes;
        current.mte3Bytes = outputBytes;
        if (MulOverflow(rows, layout.inner, current.vectorElements)) {
            return false;
        }
        current.dmaCommands = (fragments + batches) * tileCount;
        current.fragments = fragments;
        current.tailBytes = rows == 0 ? 0 : (layout.repeat - (rows % layout.repeat)) % layout.repeat;
        MaxStats(stats, current);
    }
    return true;
}

bool InferBroadcastShape(const gert::Shape& selfShape, const gert::Shape& otherShape,
    gert::Shape& outputShape)
{
    const size_t selfRank = selfShape.GetDimNum();
    const size_t otherRank = otherShape.GetDimNum();
    const size_t outputRank = std::max(selfRank, otherRank);
    if (outputRank > optiling::GREATER_MAX_RANK) {
        return false;
    }

    outputShape.SetDimNum(0);
    for (size_t axis = 0; axis < outputRank; ++axis) {
        const size_t selfOffset = outputRank - selfRank;
        const size_t otherOffset = outputRank - otherRank;
        const int64_t selfDim = axis < selfOffset ? 1 : selfShape.GetDim(axis - selfOffset);
        const int64_t otherDim = axis < otherOffset ? 1 : otherShape.GetDim(axis - otherOffset);
        int64_t outputDim = 0;
        if (selfDim == otherDim) {
            outputDim = selfDim;
        } else if (selfDim == 1) {
            outputDim = otherDim;
        } else if (otherDim == 1) {
            outputDim = selfDim;
        } else if (selfDim < 0 && otherDim >= 0) {
            outputDim = otherDim;
        } else if (otherDim < 0 && selfDim >= 0) {
            outputDim = selfDim;
        } else if (selfDim < 0 && otherDim < 0) {
            outputDim = -1;
        } else {
            return false;
        }
        outputShape.AppendDim(outputDim);
    }
    return true;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
    if (dataType != context->GetInputDesc(1)->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    DataTypePlan typePlan;
    if (!GetDataTypePlan(dataType, typePlan)) {
        return ge::GRAPH_FAILED;
    }

    CollapsedTopology topology;
    if (!BuildCollapsedTopology(context->GetInputShape(0)->GetStorageShape(),
        context->GetInputShape(1)->GetStorageShape(), topology)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t platformCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    if (platformCores == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    GreaterTilingData tiling;
    tiling.set_version(TILING_VERSION);
    tiling.set_partitionMode(GREATER_PARTITION_GROUP_MAJOR);
    tiling.set_broadcastMask(0);
    tiling.set_rowBatch(1);
    tiling.set_linearBufferNum(typePlan.linearBufferNum);
    tiling.set_collapsedRank(topology.rank);
    tiling.set_totalElements(topology.totalElements);
    tiling.set_dims(topology.dims.data());
    tiling.set_selfStrides(topology.selfStrides.data());
    tiling.set_otherStrides(topology.otherStrides.data());
    // bool 输出每元素占 1 B。输入是一次性顺序流并保持绕过；仅当整个输出能容纳在
    // 运行时 L2 时，才允许 MTE3 使用默认缓存路径，避免超容量写回替换。
    const bool outputCacheEnabled = l2Size != 0 && topology.totalElements <= l2Size;
    tiling.set_outputCacheEnabled(outputCacheEnabled ? 1U : 0U);
    tiling.set_packedPacketRows(0);

    uint32_t activeCores = 1;
    uint32_t tileElements = typePlan.alignment;
    uint32_t scheduleMode = GREATER_SCHEDULE_AFFINE_ROW;
    uint64_t rowCount = 1;
    uint64_t outer = 1;
    uint64_t repeat = 1;
    uint64_t inner = topology.rank == 0 ? 1 : topology.dims[topology.rank - 1];

    if (IsLinear(topology)) {
        scheduleMode = GREATER_SCHEDULE_LINEAR_SPAN;
        rowCount = topology.totalElements == 0 ? 0 : 1;
        outer = 1;
        repeat = 1;
        inner = topology.totalElements;
        if (!BuildLinearTiling(ubSize, typePlan, topology.totalElements, tileElements)) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t workUnits = CeilDiv(topology.totalElements, typePlan.alignment);
        activeCores = topology.totalElements <= tileElements ? 1U :
            static_cast<uint32_t>(std::min<uint64_t>(
                platformCores, std::max<uint64_t>(1, workUnits)));
    } else {
        RowLayout rowLayout;
        if (MatchResidentRow(topology, rowLayout)) {
            scheduleMode = GREATER_SCHEDULE_RESIDENT_ROW;
            outer = rowLayout.outer;
            repeat = rowLayout.repeat;
            inner = rowLayout.inner;
            if (MulOverflow(outer, repeat, rowCount)) {
                return ge::GRAPH_FAILED;
            }
            uint32_t rowBatch = 1;
            if (!BuildRowTiling(ubSize, typePlan, inner, repeat, tileElements, rowBatch)) {
                return ge::GRAPH_FAILED;
            }
            tiling.set_rowBatch(rowBatch);
            const uint32_t candidateCores = static_cast<uint32_t>(
                std::min<uint64_t>(platformCores, std::max<uint64_t>(1, rowCount)));
            PartitionStats groupStats;
            PartitionStats rowStats;
            if (!BuildPartitionStats(rowLayout, tileElements, rowBatch,
                candidateCores, GREATER_PARTITION_GROUP_MAJOR, typePlan.rawBytes, groupStats) ||
                !BuildPartitionStats(rowLayout, tileElements, rowBatch,
                    candidateCores, GREATER_PARTITION_ROW_MAJOR, typePlan.rawBytes, rowStats)) {
                return ge::GRAPH_FAILED;
            }
            // INT32 行广播的 resident 是半精度零向量/批量计算复用对象；当重复行显著多于
            // 单批容量时，按 group 保持 resident 局部性，避免 row-major 跨 group 碎片化。
            const bool int32ResidentReuse = typePlan.persistentBytes != 0 &&
                repeat >= static_cast<uint64_t>(rowBatch) * 2;
            const bool useGroupPartition = int32ResidentReuse || !LessStats(rowStats, groupStats);
            if (!useGroupPartition) {
                tiling.set_partitionMode(GREATER_PARTITION_ROW_MAJOR);
                activeCores = candidateCores;
            } else {
                tiling.set_partitionMode(GREATER_PARTITION_GROUP_MAJOR);
                activeCores = static_cast<uint32_t>(std::min<uint64_t>(
                    platformCores, std::max<uint64_t>(1, outer)));
            }
            // 连续包需要完整 group 所有权，才能一次构建并复用该 group 的常驻行。
            // 先尝试带物理行距的 resident 路径；其条件仅依赖 INT32、32B 相位、
            // UB 容量与 repeat 整除关系，失败后保留原有紧凑包 fallback。
            if (useGroupPartition) {
                uint32_t residentPacketRows = 0;
                if (scheduleMode == GREATER_SCHEDULE_RESIDENT_ROW &&
                    PitchedResidentRowFits(ubSize, typePlan, inner, repeat, tileElements,
                    residentPacketRows)) {
                    scheduleMode = GREATER_SCHEDULE_RESIDENT_PITCHED;
                    tiling.set_packedPacketRows(residentPacketRows);
                } else {
                    uint32_t packedPacketRows = 0;
                    if (scheduleMode == GREATER_SCHEDULE_RESIDENT_ROW &&
                        PackedResidentRowFits(ubSize, typePlan, inner, repeat, tileElements,
                        packedPacketRows)) {
                        scheduleMode = GREATER_SCHEDULE_RESIDENT_PACKED;
                        tiling.set_packedPacketRows(packedPacketRows);
                    }
                }
            }
            tiling.set_broadcastMask(rowLayout.broadcastFlag);
        } else {
            scheduleMode = GREATER_SCHEDULE_AFFINE_ROW;
            outer = 1;
            for (uint32_t axis = 0; axis + 1 < topology.rank; ++axis) {
                if (MulOverflow(outer, topology.dims[axis], outer)) {
                    return ge::GRAPH_FAILED;
                }
            }
            inner = topology.rank == 0 ? 1 : topology.dims[topology.rank - 1];
            if (MulOverflow(outer, inner, topology.totalElements)) {
                return ge::GRAPH_FAILED;
            }
            rowCount = outer;
            if (!BuildLinearTiling(ubSize, typePlan, inner, tileElements)) {
                return ge::GRAPH_FAILED;
            }
            activeCores = static_cast<uint32_t>(std::min<uint64_t>(
                platformCores, std::max<uint64_t>(1, rowCount)));
        }
    }

    tiling.set_scheduleMode(scheduleMode);
    tiling.set_tileElements(tileElements);
    tiling.set_activeCores(activeCores);
    tiling.set_rowCount(rowCount);
    tiling.set_outer(outer);
    tiling.set_repeat(repeat);
    tiling.set_inner(inner);

    auto* rawTiling = context->GetRawTilingData();
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize() || workspaceSize == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSize[0] = 0;
    const uint64_t tilingKey = scheduleMode == GREATER_SCHEDULE_LINEAR_SPAN &&
        activeCores == 1 && topology.totalElements <= tileElements ?
        GREATER_KEY_V2_WHOLE_LINEAR : GREATER_KEY_V2;
    context->SetBlockDim(activeCores);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return InferBroadcastShape(*context->GetInputShape(0), *context->GetInputShape(1),
        *context->GetOutputShape(0)) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Greater);
}
