#include "square_sum_v1_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr uint32_t TOPOLOGY_AR = 0;
constexpr uint32_t TOPOLOGY_ARA = 1;
constexpr uint32_t TOPOLOGY_RAR = 2;
constexpr uint32_t TOPOLOGY_RA = 3;
constexpr uint32_t TOPOLOGY_GENERIC = 4;
constexpr uint64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t UB_SKEW_BYTES = 256;
constexpr uint32_t MAX_COPY_ROWS = 4095;
constexpr uint32_t SYNC_TILING_KEY = 1;
constexpr uint32_t COMPACT_AR_DIRECT_TILING_KEY = 2;
constexpr uint32_t MAX_VECTOR_REPEAT_STRIDE = 65535;
constexpr uint32_t MAX_VECTOR_REPEATS = 255;
// 紧凑 RA 至少容纳四个 32B 对齐行组，限制 Gather 合并流量不超过输入流量的四分之一。
constexpr uint32_t MIN_COMPACT_RA_GROUPS = 4;
constexpr uint32_t COMPACT_ARA_PHASE_ACCUMULATORS = 2;
constexpr uint32_t COMPACT_RA_TILED = 1;
constexpr uint32_t COMPACT_RA_PHASE = 2;
constexpr uint32_t COMPACT_RA_PHASE_SLABS = 1;
constexpr uint32_t COMPACT_RA_PHASE_ACCUMULATORS = 1;
constexpr uint32_t COMPACT_RA_PHASE_MAX_SEGMENTS = 2;
constexpr uint32_t COMPACT_RA_PHASE_SCHEDULE_WAVE = 1;
constexpr bool ENABLE_COMPACT_RA_PHASE_WAVE = true;
// 非负平方和按一次乘法舍入和 n 层 FP32 加法建模；仅使用 1e-4 相对误差预算的三分之二。
constexpr uint64_t FP32_UNIT_ROUNDOFF_DENOMINATOR = 1ULL << 24;
constexpr uint64_t FP32_PHASE_ERROR_BUDGET_DENOMINATOR = 15000;
constexpr uint32_t MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH =
    static_cast<uint32_t>(
        (FP32_UNIT_ROUNDOFF_DENOMINATOR - FP32_PHASE_ERROR_BUDGET_DENOMINATOR) /
        (FP32_PHASE_ERROR_BUDGET_DENOMINATOR + 1));
constexpr uint32_t COMPACT_RA_PHASE_CORE_DENOMINATOR = 4;
// 大工作集 phase-RA 只连续读取一次输入；保留一个分数核预算给尾部与归并。
constexpr uint32_t COMPACT_RA_PHASE_RESERVED_CORE_UNITS = 1;
constexpr uint32_t COMPACT_RA_PHASE_CORE_NUMERATOR =
    COMPACT_RA_PHASE_CORE_DENOMINATOR - COMPACT_RA_PHASE_RESERVED_CORE_UNITS;
constexpr uint32_t RA_STAGGER_GROUPS = 4;
constexpr uint32_t MIN_RA_STAGGER_ROUNDS = 8;
constexpr uint32_t MIN_RA_STAGGER_SHIFT_ROWS = RA_STAGGER_GROUPS - 1;
constexpr uint32_t RAR_SCHEDULE_WAVE = 1;
constexpr uint32_t MIN_RAR_WAVE_FULL_WAVES = 4;
// 所有 DMA 工作量门槛以 32B block 计数表达，避免把公开 shape 的字节数带入调度。
constexpr uint32_t MIN_DMA_AMORTIZATION_BLOCKS = 1024;
constexpr uint64_t MIN_RAR_WAVE_TILE_BYTES =
    UB_BLOCK_BYTES * MIN_DMA_AMORTIZATION_BLOCKS;
constexpr uint64_t MIN_COMPACT_RA_PHASE_WAVE_TILE_BYTES =
    UB_BLOCK_BYTES * MIN_DMA_AMORTIZATION_BLOCKS;
constexpr uint32_t MIN_COMPACT_RA_PHASE_WAVE_FULL_WAVES = MIN_RA_STAGGER_ROUNDS;
// 小于一个可摊销 DMA tile 的工作集由单核处理，避免多核启动与尾波开销。
// 阈值由搬运粒度和运行时可用的 UB 容量共同限定。
constexpr uint32_t MIN_SINGLE_CORE_BLOCKS = 16;
constexpr uint64_t SINGLE_CORE_DMA_AMORTIZATION_BYTES =
    UB_BLOCK_BYTES * MIN_SINGLE_CORE_BLOCKS;
constexpr uint64_t COMPACT_AR_MIN_INPUT_BYTES =
    UB_BLOCK_BYTES * MIN_DMA_AMORTIZATION_BLOCKS;
constexpr uint32_t MIN_RAR_WAVE_CORES = 4;

template <typename T>
bool MulOverflow(T lhs, T rhs, T& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<T>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
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

uint32_t GcdU32(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

uint32_t LcmU32(uint32_t lhs, uint32_t rhs)
{
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    const uint32_t gcd = GcdU32(lhs, rhs);
    if (gcd == 0) {
        return 0;
    }
    const uint64_t result = static_cast<uint64_t>(lhs / gcd) * rhs;
    return result > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(result);
}

bool ParseAxes(const gert::Shape& shape, const gert::ContinuousVector* axisAttr,
    std::array<bool, optiling::SQUARE_SUM_MAX_RANK>& reduced)
{
    const int32_t rank = shape.GetDimNum();
    reduced.fill(false);
    if (rank <= 0 || rank > static_cast<int32_t>(optiling::SQUARE_SUM_MAX_RANK) || axisAttr == nullptr) {
        return false;
    }
    if (axisAttr->GetSize() == 0) {
        std::fill(reduced.begin(), reduced.begin() + rank, true);
        return true;
    }
    const auto* axes = static_cast<const int64_t*>(axisAttr->GetData());
    if (axes == nullptr) {
        return false;
    }
    for (size_t i = 0; i < axisAttr->GetSize(); ++i) {
        int64_t axis = axes[i];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return false;
        }
        if (reduced[static_cast<size_t>(axis)]) {
            return false;
        }
        reduced[static_cast<size_t>(axis)] = true;
    }
    return true;
}

uint64_t SelectSingleCoreInputBytes(uint64_t ubSize)
{
    return std::min<uint64_t>(SINGLE_CORE_DMA_AMORTIZATION_BYTES, ubSize);
}

bool IsStreamingWorkload(uint64_t inputBytes, uint64_t l2Size)
{
    // 单次读取且工作集超过 L2 时没有跨 tile 复用，按平台实际缓存容量选择流式路径。
    return l2Size > 0 && inputBytes > l2Size;
}

uint32_t SelectCoreNum(uint32_t topology, uint64_t inputBytes, uint64_t dim0, uint64_t dim1,
    uint32_t platformCores, uint64_t singleCoreInputBytes, uint32_t elementSize)
{
    if (platformCores == 0 || inputBytes <= singleCoreInputBytes) {
        return 1;
    }
    uint64_t tasks = 1;
    if (topology == TOPOLOGY_AR || topology == TOPOLOGY_ARA) {
        tasks = dim0;
    } else if (topology == TOPOLOGY_RAR) {
        tasks = dim1;
    } else if (topology == TOPOLOGY_RA) {
        tasks = dim0;
    }
    uint64_t targetCores = platformCores;
    if (topology == TOPOLOGY_AR && elementSize > 0) {
        // 在 AR 拓扑中，每行产生 1 个标量输出。
        // 为保证每个核心的输出写回至少达到一个完整的 32B 硬件对齐块，避免非对齐与跨核写入碎片：
        const uint64_t minTasksPerCore = UB_BLOCK_BYTES / elementSize;
        const uint64_t maxCores = tasks / minTasksPerCore + (tasks % minTasksPerCore != 0 ? 1 : 0);
        targetCores = std::min<uint64_t>(platformCores, maxCores);
    }
    const uint64_t maxTasksPerCore = tasks / targetCores + (tasks % targetCores != 0 ? 1 : 0);
    const uint64_t activeCores = maxTasksPerCore == 0 ? 1 :
        tasks / maxTasksPerCore + (tasks % maxTasksPerCore != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::max<uint64_t>(1, activeCores));
}

uint64_t DivUp(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + (value % divisor == 0 ? 0 : 1);
}

uint32_t SelectCompactRaPhaseCoreNum(uint32_t blockDim, uint32_t platformCores)
{
    if (blockDim == 0 || platformCores < COMPACT_RA_PHASE_CORE_DENOMINATOR) {
        return blockDim;
    }
    const uint32_t coreCap = static_cast<uint32_t>(
        static_cast<uint64_t>(platformCores) * COMPACT_RA_PHASE_CORE_NUMERATOR /
        COMPACT_RA_PHASE_CORE_DENOMINATOR);
    return std::min(blockDim, std::max(1U, coreCap));
}

bool SupportsCompactRaPhase(uint64_t reduceRows, uint32_t inner, uint32_t phaseRows,
    uint32_t blockDim, uint32_t phaseTile)
{
    if (reduceRows == 0 || inner == 0 || phaseRows == 0 || blockDim == 0 || phaseTile == 0) {
        return false;
    }
    const uint64_t phaseElements = static_cast<uint64_t>(phaseRows) * inner;
    if (phaseElements > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        inner > std::numeric_limits<uint32_t>::max() / phaseTile) {
        return false;
    }

    const uint64_t maxCoreRows = DivUp(reduceRows, blockDim);
    const uint64_t fullTiles = maxCoreRows / phaseTile;
    const uint64_t tailRows = maxCoreRows % phaseTile;
    const uint64_t groupsPerFullTile = DivUp(phaseTile, phaseRows);
    const uint64_t tailGroups = DivUp(tailRows, phaseRows);
    uint64_t totalGroups = 0;
    if (MulOverflow(fullTiles, groupsPerFullTile, totalGroups) ||
        totalGroups > std::numeric_limits<uint64_t>::max() - tailGroups) {
        return false;
    }
    totalGroups += tailGroups;
    const uint64_t laneDepth = DivUp(totalGroups, COMPACT_RA_PHASE_ACCUMULATORS);
    const uint64_t mergeDepth =
        (COMPACT_RA_PHASE_ACCUMULATORS - 1U) + (phaseRows - 1U) + (blockDim - 1U);
    if (laneDepth > std::numeric_limits<uint64_t>::max() - mergeDepth) {
        return false;
    }
    const uint64_t phaseRounds = DivUp(maxCoreRows, phaseTile);
    return laneDepth + mergeDepth <= MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH &&
        phaseTile >= phaseRows * 2U && phaseRounds >= MIN_RA_STAGGER_ROUNDS;
}

bool GetCompactRaPhaseAlignedSplitStats(uint64_t reduceRows, uint32_t phaseRows,
    uint32_t blockDim, uint64_t& minCoreRows, uint64_t& maxCoreRows,
    uint64_t& maxCoreGroups)
{
    minCoreRows = 0;
    maxCoreRows = 0;
    maxCoreGroups = 0;
    if (reduceRows == 0 || phaseRows == 0 || blockDim == 0) {
        return false;
    }

    const uint64_t fullGroups = reduceRows / phaseRows;
    const uint64_t residualRows = reduceRows % phaseRows;
    // 完整行组少于核数时会产生空核，不启用分组分核，保持原逐行均分路径。
    if (fullGroups < blockDim) {
        return false;
    }
    const uint64_t baseGroups = fullGroups / blockDim;
    const uint64_t extraGroups = fullGroups % blockDim;
    const uint64_t longCoreGroups = baseGroups + (extraGroups > 0 ? 1U : 0U);
    const uint64_t lastCoreGroups = baseGroups + (residualRows > 0 ? 1U : 0U);
    maxCoreGroups = std::max(longCoreGroups, lastCoreGroups);

    uint64_t baseCoreRows = 0;
    uint64_t longCoreRows = 0;
    uint64_t lastCoreRows = 0;
    if (MulOverflow(baseGroups, static_cast<uint64_t>(phaseRows), baseCoreRows) ||
        MulOverflow(longCoreGroups, static_cast<uint64_t>(phaseRows), longCoreRows) ||
        MulOverflow(baseGroups, static_cast<uint64_t>(phaseRows), lastCoreRows) ||
        lastCoreRows > std::numeric_limits<uint64_t>::max() - residualRows) {
        return false;
    }
    lastCoreRows += residualRows;
    maxCoreRows = std::max(longCoreRows, lastCoreRows);
    minCoreRows = lastCoreRows;
    if (extraGroups > 0) {
        minCoreRows = std::min(minCoreRows, longCoreRows);
    }
    // 除末核外仍存在未领取额外行组的核时，baseCoreRows 也是实际核负载。
    if (extraGroups < static_cast<uint64_t>(blockDim - 1U)) {
        minCoreRows = std::min(minCoreRows, baseCoreRows);
    }
    return minCoreRows > 0 && minCoreRows <= maxCoreRows && maxCoreRows <= reduceRows;
}

bool RequiresCompactRaPhaseSegments(uint64_t reduceRows, uint32_t phaseRows, uint32_t blockDim)
{
    uint64_t minCoreRows = 0;
    uint64_t maxCoreRows = 0;
    uint64_t maxCoreGroups = 0;
    if (!GetCompactRaPhaseAlignedSplitStats(
        reduceRows, phaseRows, blockDim, minCoreRows, maxCoreRows, maxCoreGroups)) {
        return false;
    }
    const uint64_t mergeDepth =
        (COMPACT_RA_PHASE_ACCUMULATORS - 1U) + (phaseRows - 1U) + (blockDim - 1U);
    return maxCoreGroups > MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH ||
        mergeDepth > MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH - maxCoreGroups;
}

bool GetCompactRaPhaseSegmentPlan(uint64_t minCoreGroups, uint64_t maxCoreGroups,
    uint32_t phaseRows, uint32_t blockDim, uint32_t phaseTile,
    uint32_t& segments, uint32_t& checkpointGroups)
{
    segments = 0;
    checkpointGroups = 0;
    if (minCoreGroups == 0 || maxCoreGroups < minCoreGroups || phaseRows == 0 ||
        blockDim == 0 || phaseTile == 0 || phaseTile % phaseRows != 0) {
        return false;
    }
    const uint64_t mergeDepth =
        (COMPACT_RA_PHASE_ACCUMULATORS - 1U) + (phaseRows - 1U) + (blockDim - 1U);
    if (maxCoreGroups <= std::numeric_limits<uint64_t>::max() - mergeDepth &&
        maxCoreGroups + mergeDepth <= MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH) {
        segments = 1;
        return true;
    }

    const uint64_t groupsPerTile = phaseTile / phaseRows;
    const uint64_t minSegmentGroups = 2U * groupsPerTile;
    if (groupsPerTile == 0 || minCoreGroups < 2U * minSegmentGroups) {
        return false;
    }
    const uint64_t halfGroups = DivUp(maxCoreGroups, 2U);
    const uint64_t lowerCheckpoint = halfGroups / groupsPerTile * groupsPerTile;
    const uint64_t upperCheckpoint = lowerCheckpoint == halfGroups ? lowerCheckpoint :
        (lowerCheckpoint <= std::numeric_limits<uint64_t>::max() - groupsPerTile ?
            lowerCheckpoint + groupsPerTile : 0U);
    uint64_t checkpoint = 0;
    uint64_t maxSegmentGroups = std::numeric_limits<uint64_t>::max();
    const uint64_t candidates[2] = {lowerCheckpoint, upperCheckpoint};
    for (uint64_t candidate : candidates) {
        if (candidate < minSegmentGroups || candidate > minCoreGroups - minSegmentGroups ||
            candidate > std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        const uint64_t candidateMax = std::max(candidate, maxCoreGroups - candidate);
        if (candidateMax < maxSegmentGroups) {
            checkpoint = candidate;
            maxSegmentGroups = candidateMax;
        }
    }
    if (checkpoint == 0) {
        return false;
    }
    // 两段都先完成 phase 行归并，再增加一次段间向量加法，最后执行跨核归并。
    const uint64_t segmentedMergeDepth = mergeDepth + 1U;
    if (maxSegmentGroups > std::numeric_limits<uint64_t>::max() - segmentedMergeDepth ||
        maxSegmentGroups + segmentedMergeDepth > MAX_COMPACT_RA_PHASE_ACCUMULATION_DEPTH) {
        return false;
    }
    segments = COMPACT_RA_PHASE_MAX_SEGMENTS;
    checkpointGroups = static_cast<uint32_t>(checkpoint);
    return true;
}

bool SupportsCompactRaPhaseAligned(uint64_t reduceRows, uint32_t inner, uint32_t phaseRows,
    uint32_t blockDim, uint32_t phaseTile, uint32_t* segments = nullptr,
    uint32_t* checkpointGroups = nullptr)
{
    if (inner == 0 || phaseRows == 0 || phaseTile == 0 ||
        phaseRows > std::numeric_limits<uint32_t>::max() / 2U ||
        phaseTile % phaseRows != 0 || inner > std::numeric_limits<uint32_t>::max() / phaseTile) {
        return false;
    }

    uint64_t minCoreRows = 0;
    uint64_t maxCoreRows = 0;
    uint64_t maxCoreGroups = 0;
    if (!GetCompactRaPhaseAlignedSplitStats(
        reduceRows, phaseRows, blockDim, minCoreRows, maxCoreRows, maxCoreGroups)) {
        return false;
    }
    uint64_t tileElements = 0;
    uint64_t tileBytes = 0;
    if (MulOverflow(static_cast<uint64_t>(phaseTile), static_cast<uint64_t>(inner), tileElements) ||
        MulOverflow(tileElements, static_cast<uint64_t>(sizeof(float)), tileBytes) ||
        tileBytes % UB_BLOCK_BYTES != 0) {
        return false;
    }

    const uint64_t minCoreGroups = DivUp(minCoreRows, phaseRows);
    uint32_t selectedSegments = 0;
    uint32_t selectedCheckpointGroups = 0;
    if (!GetCompactRaPhaseSegmentPlan(minCoreGroups, maxCoreGroups, phaseRows,
        blockDim, phaseTile, selectedSegments, selectedCheckpointGroups)) {
        return false;
    }
    const uint64_t phaseRounds = DivUp(maxCoreRows, phaseTile);
    const bool supported = DivUp(minCoreRows, phaseTile) == phaseRounds &&
        phaseTile >= phaseRows * 2U && phaseRounds >= MIN_RA_STAGGER_ROUNDS;
    if (supported) {
        if (segments != nullptr) {
            *segments = selectedSegments;
        }
        if (checkpointGroups != nullptr) {
            *checkpointGroups = selectedCheckpointGroups;
        }
    }
    return supported;
}

bool GetCompactRaPhaseWaveStats(uint64_t reduceRows, uint32_t phaseRows,
    uint32_t blockDim, uint32_t phaseTile, uint64_t& minCoreGroups,
    uint64_t& maxCoreGroups)
{
    minCoreGroups = 0;
    maxCoreGroups = 0;
    if (reduceRows == 0 || phaseRows == 0 || blockDim == 0 || phaseTile == 0 ||
        phaseRows > std::numeric_limits<uint32_t>::max() / 2U ||
        phaseTile % phaseRows != 0 || phaseTile < phaseRows * 2U) {
        return false;
    }
    const uint64_t fullTiles = reduceRows / phaseTile;
    // 每个核至少经历多轮完整 wave，避免为短任务引入额外的跨 wave 控制开销。
    if (fullTiles / blockDim < MIN_COMPACT_RA_PHASE_WAVE_FULL_WAVES) {
        return false;
    }
    const uint64_t groupsPerTile = phaseTile / phaseRows;
    const uint64_t baseTiles = fullTiles / blockDim;
    const uint64_t extraTiles = fullTiles % blockDim;
    if (MulOverflow(baseTiles, groupsPerTile, minCoreGroups) || minCoreGroups == 0) {
        return false;
    }

    uint64_t maxTiles = baseTiles;
    if (extraTiles != 0) {
        if (maxTiles == std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        ++maxTiles;
    }
    uint64_t longCoreGroups = 0;
    if (MulOverflow(maxTiles, groupsPerTile, longCoreGroups)) {
        return false;
    }
    uint64_t tailCoreGroups = 0;
    if (MulOverflow(baseTiles, groupsPerTile, tailCoreGroups)) {
        return false;
    }
    const uint64_t tailRows = reduceRows % phaseTile;
    if (tailRows != 0) {
        const uint64_t tailGroups = DivUp(tailRows, phaseRows);
        if (tailCoreGroups > std::numeric_limits<uint64_t>::max() - tailGroups) {
            return false;
        }
        // 尾 tile 由最后一个不完整 wave 的下一物理核领取，保持该波 GM 访问连续。
        tailCoreGroups += tailGroups;
    }
    maxCoreGroups = std::max(longCoreGroups, tailCoreGroups);
    return minCoreGroups <= maxCoreGroups;
}

bool SupportsCompactRaPhaseWaveSchedule(uint64_t reduceRows, uint32_t inner,
    uint32_t phaseRows, uint32_t blockDim, uint32_t phaseTile,
    uint32_t* segments = nullptr, uint32_t* checkpointGroups = nullptr)
{
    if (inner == 0 || phaseTile == 0 ||
        inner > std::numeric_limits<uint32_t>::max() / phaseTile) {
        return false;
    }
    uint64_t tileElements = 0;
    uint64_t tileBytes = 0;
    if (MulOverflow(static_cast<uint64_t>(phaseTile), static_cast<uint64_t>(inner), tileElements) ||
        MulOverflow(tileElements, static_cast<uint64_t>(sizeof(float)), tileBytes) ||
        tileBytes % UB_BLOCK_BYTES != 0 || tileBytes < MIN_COMPACT_RA_PHASE_WAVE_TILE_BYTES) {
        return false;
    }
    uint64_t minCoreGroups = 0;
    uint64_t maxCoreGroups = 0;
    if (!GetCompactRaPhaseWaveStats(reduceRows, phaseRows, blockDim, phaseTile,
        minCoreGroups, maxCoreGroups)) {
        return false;
    }
    uint32_t selectedSegments = 0;
    uint32_t selectedCheckpointGroups = 0;
    if (!GetCompactRaPhaseSegmentPlan(minCoreGroups, maxCoreGroups, phaseRows,
        blockDim, phaseTile, selectedSegments, selectedCheckpointGroups)) {
        return false;
    }
    if (segments != nullptr) {
        *segments = selectedSegments;
    }
    if (checkpointGroups != nullptr) {
        *checkpointGroups = selectedCheckpointGroups;
    }
    return true;
}

uint32_t SelectCompactRaPhaseAlignedTile(uint32_t maxTile, uint64_t reduceRows,
    uint32_t inner, uint32_t phaseRows, uint32_t blockDim)
{
    if (maxTile == 0 || phaseRows == 0 ||
        phaseRows > std::numeric_limits<uint32_t>::max() / 2U) {
        return 0;
    }
    const uint32_t minTile = phaseRows * 2U;
    uint32_t candidate = maxTile / phaseRows * phaseRows;
    while (candidate >= minTile) {
        if (SupportsCompactRaPhaseAligned(
            reduceRows, inner, phaseRows, blockDim, candidate)) {
            return candidate;
        }
        if (candidate < minTile + phaseRows) {
            break;
        }
        candidate -= phaseRows;
    }
    return 0;
}

uint32_t SelectCompactRaPhaseDmaAlignmentTile(uint32_t maxTile, uint64_t reduceRows,
    uint32_t inner, uint32_t phaseRows, uint32_t blockDim)
{
    if (maxTile == 0 || inner == 0 || phaseRows == 0 || blockDim == 0) {
        return maxTile;
    }
    // 在不把 DMA tile 缩小超过四分之一的前提下，优先缩短连续 tile 起点回到 512B
    // 边界的周期。这样不增加 GM 字节，只改变同一 phase-wave 内请求的地址分布。
    const uint32_t minTile = static_cast<uint32_t>(
        DivUp(static_cast<uint64_t>(maxTile) * 3U, 4U));
    const uint64_t rowBytes = static_cast<uint64_t>(inner) * sizeof(float);
    uint32_t selectedTile = maxTile;
    uint32_t selectedAlignment = GcdU32(
        static_cast<uint32_t>((static_cast<uint64_t>(maxTile) * rowBytes) % 512U), 512U);
    uint32_t candidate = maxTile / phaseRows * phaseRows;
    while (candidate >= phaseRows * 2U) {
        if (candidate < minTile) {
            break;
        }
        uint64_t tileBytes = 0;
        if (!MulOverflow(static_cast<uint64_t>(candidate), rowBytes, tileBytes) &&
            tileBytes >= MIN_COMPACT_RA_PHASE_WAVE_TILE_BYTES &&
            SupportsCompactRaPhaseAligned(
                reduceRows, inner, phaseRows, blockDim, candidate)) {
            const uint32_t alignment = GcdU32(
                static_cast<uint32_t>(tileBytes % 512U), 512U);
            if (alignment > selectedAlignment) {
                selectedTile = candidate;
                selectedAlignment = alignment;
            }
        }
        if (candidate < phaseRows * 3U) {
            break;
        }
        candidate -= phaseRows;
    }
    return selectedTile;
}

uint32_t SelectCompactRaPhaseSafeTile(uint32_t maxTile, uint64_t reduceRows,
    uint32_t inner, uint32_t elementSize, uint32_t phaseRows, uint32_t blockDim)
{
    if (phaseRows == 0 || elementSize == 0) {
        return 0;
    }
    // 满 tile 非 32B 整倍数时所有核统一走 DataCopyPad，避免按 GM 起址分裂成两类快慢核。
    // 若不存在此类安全候选，再回退到任意满足精度与轮数门禁的最大 tile。
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t candidate = maxTile; candidate >= phaseRows * 2U; --candidate) {
            if (!SupportsCompactRaPhase(reduceRows, inner, phaseRows, blockDim, candidate)) {
                continue;
            }
            uint64_t tileElements = 0;
            uint64_t tileBytes = 0;
            const bool byteSizeValid =
                !MulOverflow(static_cast<uint64_t>(candidate), static_cast<uint64_t>(inner), tileElements) &&
                !MulOverflow(tileElements, static_cast<uint64_t>(elementSize), tileBytes);
            if (pass != 0 || (byteSizeValid && tileBytes % UB_BLOCK_BYTES != 0)) {
                return candidate;
            }
        }
    }
    return 0;
}

bool SupportsRaStagger(uint64_t reduceRows, uint32_t blockDim, uint32_t tilePrimary)
{
    if (blockDim < RA_STAGGER_GROUPS || tilePrimary == 0) {
        return false;
    }
    const uint64_t shortCoreRows = reduceRows / blockDim;
    const uint64_t longCoreRows = DivUp(reduceRows, blockDim);
    const uint64_t shortCoreRounds = DivUp(shortCoreRows, tilePrimary);
    const uint64_t longCoreRounds = DivUp(longCoreRows, tilePrimary);
    if (shortCoreRounds != longCoreRounds || shortCoreRounds < MIN_RA_STAGGER_ROUNDS) {
        return false;
    }
    const uint64_t shortTailRows = shortCoreRows - (shortCoreRounds - 1) * tilePrimary;
    const uint64_t longTailRows = longCoreRows - (longCoreRounds - 1) * tilePrimary;
    return tilePrimary - shortTailRows >= MIN_RA_STAGGER_SHIFT_ROWS &&
        tilePrimary - longTailRows >= MIN_RA_STAGGER_SHIFT_ROWS;
}

bool SupportsRarWaveSchedule(uint64_t reduceRows, uint64_t inner, uint32_t elementSize,
    uint32_t blockDim, uint32_t tilePrimary, uint32_t rowQuantum)
{
    if (blockDim < MIN_RAR_WAVE_CORES || rowQuantum == 0 || tilePrimary == 0 ||
        inner == 0 || elementSize == 0 || tilePrimary % rowQuantum != 0) {
        return false;
    }
    uint64_t waveRows = 0;
    uint64_t tileElements = 0;
    uint64_t tileBytes = 0;
    if (MulOverflow(static_cast<uint64_t>(blockDim), static_cast<uint64_t>(tilePrimary), waveRows) ||
        MulOverflow(static_cast<uint64_t>(tilePrimary), inner, tileElements) ||
        MulOverflow(tileElements, static_cast<uint64_t>(elementSize), tileBytes)) {
        return false;
    }
    return waveRows > 0 && reduceRows / waveRows >= MIN_RAR_WAVE_FULL_WAVES &&
        tileBytes >= MIN_RAR_WAVE_TILE_BYTES;
}

uint32_t SelectRaStaggerFriendlyTile(uint32_t maxTile, uint64_t reduceRows, uint32_t blockDim)
{
    // 只允许极小幅度回退；用一两行 UB 容量换取错峰，避免明显增加 DMA 轮数。
    for (uint32_t retreat = 0; retreat < RA_STAGGER_GROUPS && retreat < maxTile; ++retreat) {
        const uint32_t candidate = maxTile - retreat;
        if (SupportsRaStagger(reduceRows, blockDim, candidate)) {
            return candidate;
        }
    }
    return maxTile;
}

uint32_t GetReduceTmpBytes(uint32_t rows, uint32_t cols, AscendC::ReducePattern pattern)
{
    ge::Shape shape({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetReduceSumMaxMinTmpSize(
        shape, ge::DT_FLOAT, pattern, true, true, maxValue, minValue);
    return static_cast<uint32_t>(AlignUp(minValue, UB_BLOCK_BYTES));
}

uint32_t SelectTilePrimary(uint64_t ubSize, uint64_t taskSize, uint32_t alignedInner,
    uint32_t elementSize, uint32_t inputBufferNum, AscendC::ReducePattern pattern,
    uint32_t& sharedTmpBytes)
{
    if (taskSize == 0 || alignedInner == 0) {
        return 0;
    }
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(taskSize, MAX_COPY_ROWS));
    uint32_t best = 0;
    uint32_t bestTmpBytes = 0;
    while (low <= high) {
        const uint32_t primary = low + (high - low) / 2;
        const uint32_t tmpBytes = GetReduceTmpBytes(primary, alignedInner, pattern);
        const uint64_t tileElements = static_cast<uint64_t>(primary) * alignedInner;
        const uint64_t vectorElements = std::max<uint64_t>(primary, alignedInner);
        uint64_t used = inputBufferNum * (tileElements * elementSize + UB_SKEW_BYTES);
        if (elementSize != sizeof(float)) {
            used += tileElements * sizeof(float) + UB_SKEW_BYTES;
        }
        used += 2 * AlignUp(
            vectorElements * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(vectorElements * elementSize, UB_BLOCK_BYTES);
        used += tmpBytes;
        if (used <= ubSize) {
            best = primary;
            bestTmpBytes = tmpBytes;
            low = primary + 1;
        } else {
            high = primary - 1;
        }
    }
    sharedTmpBytes = bestTmpBytes;
    return best;
}

uint32_t SelectCompactTile(uint64_t ubSize, uint64_t taskSize, uint32_t inner,
    uint32_t elementSize, uint32_t inputBufferNum, uint32_t extraTmpBytes = 0)
{
    if (taskSize == 0 || inner == 0) {
        return 0;
    }
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(taskSize, MAX_COPY_ROWS));
    uint32_t best = 0;
    while (low <= high) {
        const uint32_t primary = low + (high - low) / 2;
        const uint64_t tileElements = static_cast<uint64_t>(primary) * inner;
        const uint64_t vectorElements = std::max<uint64_t>(primary, inner);
        uint64_t used = inputBufferNum * (tileElements * elementSize + UB_SKEW_BYTES);
        if (elementSize != sizeof(float)) {
            used += tileElements * sizeof(float) + UB_SKEW_BYTES;
        }
        used += 2 * AlignUp(
            vectorElements * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(vectorElements * elementSize, UB_BLOCK_BYTES);
        used += AlignUp(extraTmpBytes, UB_BLOCK_BYTES);
        if (used <= ubSize) {
            best = primary;
            low = primary + 1;
        } else {
            high = primary - 1;
        }
    }
    return best;
}

uint32_t SelectCompactRaTile(uint64_t ubSize, uint64_t taskSize, uint32_t inner,
    uint32_t alignedInner, uint32_t elementSize, uint32_t extraTmpBytes, uint32_t blockDim)
{
    if (taskSize == 0 || inner == 0 || alignedInner == 0 || blockDim == 0) {
        return 0;
    }
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(taskSize, MAX_COPY_ROWS));
    uint32_t best = 0;
    while (low <= high) {
        const uint32_t primary = low + (high - low) / 2;
        const uint64_t tileElements = static_cast<uint64_t>(primary) * inner;
        const uint64_t vectorElements = std::max<uint64_t>(primary, alignedInner);
        uint64_t used = tileElements * elementSize + UB_SKEW_BYTES;
        used += 2 * AlignUp(
            vectorElements * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(extraTmpBytes, UB_BLOCK_BYTES);
        if (used <= ubSize) {
            best = primary;
            low = primary + 1;
        } else {
            high = primary - 1;
        }
    }
    // core0 会把输入 Queue 复用为 workspace merge 缓冲；未对齐 inner 下需显式计入行 padding。
    while (best > 0) {
        const uint32_t mergeRows = std::min(best, blockDim - 1U);
        const uint64_t queueBytes = static_cast<uint64_t>(best) * inner * elementSize + UB_SKEW_BYTES;
        const uint64_t mergeBytes = static_cast<uint64_t>(mergeRows) * alignedInner * sizeof(float);
        if (queueBytes >= mergeBytes) {
            break;
        }
        --best;
    }
    return best;
}

uint32_t SelectCompactRaPhaseTile(uint64_t ubSize, uint64_t taskSize, uint32_t inner,
    uint32_t alignedInner, uint32_t elementSize, uint32_t inputBufferNum,
    uint32_t blockDim, uint32_t phaseRows, bool reserveSegmentBuffers)
{
    if (taskSize == 0 || inner == 0 || alignedInner == 0 || elementSize == 0 ||
        inputBufferNum == 0 || blockDim == 0 || phaseRows == 0) {
        return 0;
    }
    const uint64_t phaseElements = static_cast<uint64_t>(phaseRows) * inner;
    const uint64_t finalizeElements = 3ULL * alignedInner;
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(taskSize, MAX_COPY_ROWS));
    uint32_t best = 0;
    while (low <= high) {
        const uint32_t primary = low + (high - low) / 2;
        const uint64_t tileElements = static_cast<uint64_t>(primary) * inner;
        // phase accumulator 与输入同时参与 MulAddDst；每个输入槽只保留一个 32B 尾块，
        // 既覆盖 DataCopyPad 的最大补齐量，也避免固定 256B 偏移造成 UB bank 重合。
        uint64_t used = inputBufferNum * AlignUp(
            tileElements * elementSize + UB_BLOCK_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(
            COMPACT_RA_PHASE_ACCUMULATORS * phaseElements * sizeof(float), UB_BLOCK_BYTES);
        if (reserveSegmentBuffers) {
            // 分段候选在输入双缓冲存活期间还要常驻 finalize 工作区和第一段 checkpoint。
            used += AlignUp(finalizeElements * sizeof(float), UB_BLOCK_BYTES);
            used += AlignUp(static_cast<uint64_t>(alignedInner) * sizeof(float), UB_BLOCK_BYTES);
        }
        if (used <= ubSize) {
            best = primary;
            low = primary + 1;
        } else {
            high = primary - 1;
        }
    }
    // 非分段路径仍可在主循环结束后复用输入槽；该容量门禁同时覆盖跨核归并缓冲。
    if (static_cast<uint64_t>(best) * inner < finalizeElements) {
        return 0;
    }
    return best;
}

uint32_t SelectCompactAraPhaseTile(uint64_t ubSize, uint64_t taskSize, uint32_t inner,
    uint32_t elementSize, uint32_t inputBufferNum, uint32_t phaseRows)
{
    if (taskSize == 0 || inner == 0 || phaseRows == 0) {
        return 0;
    }
    const uint64_t phaseElements = static_cast<uint64_t>(phaseRows) * inner;
    const uint64_t phaseWorkElements = std::max<uint64_t>(
        COMPACT_ARA_PHASE_ACCUMULATORS * phaseElements,
        3 * AlignUp(inner, UB_BLOCK_BYTES / sizeof(float)));
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(taskSize, MAX_COPY_ROWS));
    uint32_t best = 0;
    while (low <= high) {
        const uint32_t primary = low + (high - low) / 2;
        const uint64_t tileElements = static_cast<uint64_t>(primary) * inner;
        uint64_t used = inputBufferNum * AlignUp(
            tileElements * elementSize + UB_BLOCK_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(phaseWorkElements * sizeof(float), UB_BLOCK_BYTES);
        used += AlignUp(
            COMPACT_ARA_PHASE_ACCUMULATORS * phaseElements * sizeof(float), UB_BLOCK_BYTES);
        used += AlignUp(inner * elementSize, UB_BLOCK_BYTES);
        if (used <= ubSize) {
            best = primary;
            low = primary + 1;
        } else {
            high = primary - 1;
        }
    }
    if (best < taskSize) {
        best = best / phaseRows * phaseRows;
    }
    return best;
}

uint32_t SelectCompactArTile(uint64_t ubSize, uint64_t taskSize, uint32_t inner,
    uint32_t elementSize, uint32_t inputBufferNum, uint32_t packStride)
{
    if (taskSize == 0 || inner < 16 || packStride == 0) {
        return 0;
    }
    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(
        taskSize, std::min<uint32_t>(MAX_COPY_ROWS, MAX_VECTOR_REPEATS)));
    uint32_t best = 0;
    while (low <= high) {
        const uint32_t rows = low + (high - low) / 2;
        const uint64_t tileElements = static_cast<uint64_t>(rows) * inner;
        const uint64_t alignedRows = AlignUp(rows, UB_BLOCK_BYTES / sizeof(float));
        uint64_t used = inputBufferNum * AlignUp(
            tileElements * elementSize + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(tileElements * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        const uint64_t blockSumElements = (tileElements + 63) / 64 * 8 + 8;
        used += AlignUp(blockSumElements * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(static_cast<uint64_t>(rows) * packStride * sizeof(float) +
            UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += 2 * AlignUp(2 * alignedRows * sizeof(float) + UB_SKEW_BYTES, UB_BLOCK_BYTES);
        used += AlignUp(
            static_cast<uint64_t>(rows) * packStride * sizeof(uint32_t), UB_BLOCK_BYTES);
        used += AlignUp(static_cast<uint64_t>(rows) * elementSize, UB_BLOCK_BYTES);
        if (used <= ubSize) {
            best = rows;
            low = rows + 1;
        } else {
            high = rows - 1;
        }
    }
    return best;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& inputShape = context->GetInputShape(0)->GetStorageShape();
    const int32_t rank = inputShape.GetDimNum();
    const auto* axisAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    std::array<bool, SQUARE_SUM_MAX_RANK> reduced{};
    if (!ParseAxes(inputShape, axisAttr, reduced)) {
        return ge::GRAPH_FAILED;
    }

    std::array<uint32_t, SQUARE_SUM_MAX_RANK> shape{};
    std::array<uint64_t, SQUARE_SUM_MAX_RANK> stride{};
    uint64_t inputSize = 1;
    uint64_t outputSize = 1;
    uint64_t reduceSize = 1;
    uint32_t axisMask = 0;
    for (int32_t i = rank - 1; i >= 0; --i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        shape[static_cast<size_t>(i)] = static_cast<uint32_t>(dim);
        stride[static_cast<size_t>(i)] = inputSize;
        if (MulOverflow(inputSize, static_cast<uint64_t>(dim), inputSize)) {
            return ge::GRAPH_FAILED;
        }
        if (reduced[static_cast<size_t>(i)]) {
            axisMask |= 1U << i;
            if (MulOverflow(reduceSize, static_cast<uint64_t>(dim), reduceSize)) {
                return ge::GRAPH_FAILED;
            }
        } else if (MulOverflow(outputSize, static_cast<uint64_t>(dim), outputSize)) {
            return ge::GRAPH_FAILED;
        }
    }

    std::vector<std::pair<bool, uint64_t>> runs;
    for (int32_t i = 0; i < rank; ++i) {
        const bool isReduce = reduced[static_cast<size_t>(i)];
        if (runs.empty() || runs.back().first != isReduce) {
            runs.emplace_back(isReduce, shape[static_cast<size_t>(i)]);
        } else if (MulOverflow(runs.back().second,
            static_cast<uint64_t>(shape[static_cast<size_t>(i)]), runs.back().second)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t topology = TOPOLOGY_GENERIC;
    uint64_t dim0 = 1;
    uint64_t dim1 = 1;
    uint64_t dim2 = 1;
    if (runs.size() == 1 && runs[0].first) {
        topology = TOPOLOGY_AR;
        dim1 = runs[0].second;
    } else if (runs.size() == 2 && !runs[0].first && runs[1].first) {
        topology = TOPOLOGY_AR;
        dim0 = runs[0].second;
        dim1 = runs[1].second;
    } else if (runs.size() == 3 && !runs[0].first && runs[1].first && !runs[2].first) {
        topology = TOPOLOGY_ARA;
        dim0 = runs[0].second;
        dim1 = runs[1].second;
        dim2 = runs[2].second;
    } else if (runs.size() == 3 && runs[0].first && !runs[1].first && runs[2].first) {
        topology = TOPOLOGY_RAR;
        dim0 = runs[0].second;
        dim1 = runs[1].second;
        dim2 = runs[2].second;
    } else if (runs.size() == 2 && runs[0].first && !runs[1].first) {
        topology = TOPOLOGY_RA;
        dim0 = runs[0].second;
        dim1 = runs[1].second;
    }

    const ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
    const int64_t typeSizeSigned = ge::GetSizeByDataType(dataType);
    if (typeSizeSigned != 2 && typeSizeSigned != 4) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t elementSize = static_cast<uint32_t>(typeSizeSigned);
    uint64_t inputBytes = 0;
    if (MulOverflow(inputSize, static_cast<uint64_t>(elementSize), inputBytes)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t platformCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    if (platformCores == 0 || ubSize < 32 * 1024) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t singleCoreInputBytes = SelectSingleCoreInputBytes(ubSize);
    const bool streamingWorkload = IsStreamingWorkload(inputBytes, l2Size);
    uint32_t blockDim = topology == TOPOLOGY_GENERIC ? 1 :
        SelectCoreNum(topology, inputBytes, dim0, dim1, platformCores, singleCoreInputBytes, elementSize);
    uint64_t inner = 1;
    uint64_t taskSize = 1;
    AscendC::ReducePattern pattern = AscendC::ReducePattern::AR;
    if (topology == TOPOLOGY_AR) {
        inner = dim1;
        taskSize = DivUp(dim0, blockDim);
    } else if (topology == TOPOLOGY_ARA) {
        inner = dim2;
        taskSize = dim1;
        pattern = AscendC::ReducePattern::RA;
    } else if (topology == TOPOLOGY_RAR) {
        inner = dim2;
        taskSize = DivUp(dim1, blockDim);
    } else if (topology == TOPOLOGY_RA) {
        inner = dim1;
        taskSize = DivUp(dim0, blockDim);
        pattern = AscendC::ReducePattern::RA;
    } else {
        inner = shape[static_cast<size_t>(rank - 1)];
        taskSize = 1;
    }
    const uint64_t alignElements = UB_BLOCK_BYTES / elementSize;
    uint64_t alignedInner64 = AlignUp(inner, alignElements);
    if (alignedInner64 == 0 || alignedInner64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t standardAlignedInner = static_cast<uint32_t>(alignedInner64);
    uint32_t alignedInner = standardAlignedInner;
    uint32_t inputBufferNum = 1;
    uint32_t sharedTmpBytes = 0;
    const uint32_t compactStride = static_cast<uint32_t>(inner) /
        GcdU32(static_cast<uint32_t>(inner), UB_BLOCK_BYTES / sizeof(float));
    const uint32_t compactArPackStride = static_cast<uint32_t>(AlignUp(
        inner / (UB_BLOCK_BYTES / sizeof(float)), UB_BLOCK_BYTES / sizeof(float)));
    bool compactAr = topology == TOPOLOGY_AR && elementSize == 2 &&
        inputBytes > COMPACT_AR_MIN_INPUT_BYTES && alignedInner > inner &&
        inner >= UB_BLOCK_BYTES / elementSize &&
        compactStride <= MAX_VECTOR_REPEAT_STRIDE;
    const uint64_t compactRarRowBytes = inner > std::numeric_limits<uint64_t>::max() / elementSize ?
        std::numeric_limits<uint64_t>::max() : inner * elementSize;
    bool compactRar = topology == TOPOLOGY_RAR && dataType == ge::DT_FLOAT16 &&
        compactRarRowBytes < UB_SKEW_BYTES && alignedInner > inner &&
        compactStride <= MAX_VECTOR_REPEAT_STRIDE;
    bool compactRa = topology == TOPOLOGY_RA && dataType == ge::DT_FLOAT &&
        alignedInner > inner && streamingWorkload;
    const uint32_t compactRaPhase = UB_BLOCK_BYTES / sizeof(float) /
        GcdU32(static_cast<uint32_t>(inner), UB_BLOCK_BYTES / sizeof(float));
    uint32_t compactRaMode = 0;
    uint32_t compactRaPhaseRows = 0;
    bool compactRaPhaseAlignedSplit = false;
    uint32_t compactRaPhaseSegments = 0;
    uint32_t compactRaPhaseCheckpointGroups = 0;
    uint32_t compactRaPhaseScheduleMode = 0;
    bool compactAra = topology == TOPOLOGY_ARA && alignedInner > inner;
    bool phaseAra = false;
    uint32_t rarRowQuantum = 1;
    uint32_t tilePrimary = 0;
    if (compactAr) {
        alignedInner = static_cast<uint32_t>(inner);
        inputBufferNum = 1;
        tilePrimary = SelectCompactArTile(ubSize, taskSize, alignedInner, elementSize,
            inputBufferNum, compactArPackStride);
        if (tilePrimary < 2) {
            compactAr = false;
            alignedInner = standardAlignedInner;
            inputBufferNum = 1;
            sharedTmpBytes = 0;
            tilePrimary = 0;
        } else {
            // 紧凑 AR 在平方后立即释放输入槽，下一 tile 的 MTE2 可与当前 tile 的
            // 压缩、行归约和输出重叠。短任务保留单槽，避免第二个队列的启动成本。
            if (taskSize > tilePrimary) {
                const uint32_t doubleBufferTile = SelectCompactArTile(ubSize, taskSize,
                    alignedInner, elementSize, 2, compactArPackStride);
                if (doubleBufferTile >= 2) {
                    inputBufferNum = 2;
                    tilePrimary = doubleBufferTile;
                }
            }
            const bool isTelescopingBf16 = (elementSize == 2 && (alignedInner & 7U) == 1U && ((alignedInner / 8U) % 64U) == 0U);
            if (isTelescopingBf16) {
                if (tilePrimary >= 16U) {
                    tilePrimary = 16U;
                }
                sharedTmpBytes = static_cast<uint32_t>(AlignUp(
                    AlignUp(tilePrimary, UB_BLOCK_BYTES / sizeof(float)) * sizeof(uint32_t), UB_BLOCK_BYTES));
            } else {
                sharedTmpBytes = static_cast<uint32_t>(AlignUp(
                    static_cast<uint64_t>(tilePrimary) * compactArPackStride * sizeof(uint32_t),
                    UB_BLOCK_BYTES));
            }
        }
    }
    if (compactRar) {
        alignedInner = static_cast<uint32_t>(inner);
        inputBufferNum = dim0 > 1 ? 2 : 1;
        tilePrimary = SelectCompactTile(
            ubSize, taskSize, alignedInner, elementSize, inputBufferNum);
        if (tilePrimary > 0) {
            // 同时对齐 FP32 向量相位与 32B 连续搬运长度，且不增加分块轮数。
            const uint32_t vectorRowQuantum = 8U / GcdU32(alignedInner, 8U);
            const uint32_t rowBytesModBlock = static_cast<uint32_t>(
                (static_cast<uint64_t>(alignedInner) * elementSize) % UB_BLOCK_BYTES);
            const uint32_t copyRowQuantum = static_cast<uint32_t>(UB_BLOCK_BYTES) /
                GcdU32(rowBytesModBlock, static_cast<uint32_t>(UB_BLOCK_BYTES));
            const uint32_t rowQuantum = LcmU32(vectorRowQuantum, copyRowQuantum);
            rarRowQuantum = rowQuantum;
            const uint32_t alignedTile = rowQuantum == 0 ? 0 :
                tilePrimary / rowQuantum * rowQuantum;
            const uint64_t originalRounds = DivUp(taskSize, tilePrimary);
            const uint64_t alignedRounds = alignedTile == 0 ? 0 :
                DivUp(taskSize, alignedTile);
            if (alignedTile > 0 && alignedRounds == originalRounds) {
                tilePrimary = alignedTile;
            }
        }
        if (tilePrimary == 0) {
            compactRar = false;
            alignedInner = standardAlignedInner;
            inputBufferNum = 1;
        }
    }
    if (compactRa) {
        inputBufferNum = 1;
        const uint64_t gatherBytes = AlignUp(
            inner * static_cast<uint64_t>(sizeof(uint32_t)), UB_BLOCK_BYTES);
        uint32_t tiledTile = 0;
        if (gatherBytes <= std::numeric_limits<uint32_t>::max()) {
            tiledTile = SelectCompactRaTile(ubSize, taskSize,
                static_cast<uint32_t>(inner), alignedInner, elementSize,
                static_cast<uint32_t>(gatherBytes), blockDim);
        }
        constexpr uint32_t phaseInputBufferNum = 2;
        const uint32_t phaseRows = compactRaPhase * COMPACT_RA_PHASE_SLABS;
        uint32_t phaseTile = SelectCompactRaPhaseTile(ubSize, taskSize,
            static_cast<uint32_t>(inner), alignedInner, elementSize,
            phaseInputBufferNum, blockDim, phaseRows, false);
        uint32_t phaseBlockDim = blockDim;
        uint64_t phaseTaskSize = taskSize;
        bool phaseEligible = SupportsCompactRaPhase(
            dim0, static_cast<uint32_t>(inner), phaseRows, phaseBlockDim, phaseTile);
        // 对需要跨核按行组重分配的 phase 候选，当前 Kernel 还无法证明
        // workspace 中间结果与输入队列的地址生命周期完全不重叠；保留同一
        // 紧凑 RA 的 tiled 归约作为结构化回退。该判断仅依赖运行时切分结果，
        // 不绑定公开 shape 或 case。
        if (phaseEligible && !compactRaPhaseAlignedSplit) {
            const uint32_t cappedBlockDim = SelectCompactRaPhaseCoreNum(blockDim, platformCores);
            if (cappedBlockDim < blockDim) {
                const uint64_t cappedTaskSize = DivUp(dim0, cappedBlockDim);
                const bool reserveSegmentBuffers =
                    RequiresCompactRaPhaseSegments(dim0, phaseRows, cappedBlockDim);
                uint32_t cappedPhaseTile = SelectCompactRaPhaseTile(ubSize, cappedTaskSize,
                    static_cast<uint32_t>(inner), alignedInner, elementSize,
                    phaseInputBufferNum, cappedBlockDim, phaseRows, reserveSegmentBuffers);
                uint32_t alignedPhaseTile = SelectCompactRaPhaseAlignedTile(
                    cappedPhaseTile, dim0, static_cast<uint32_t>(inner), phaseRows, cappedBlockDim);
                if (alignedPhaseTile > 0) {
                    alignedPhaseTile = SelectCompactRaPhaseDmaAlignmentTile(alignedPhaseTile, dim0,
                        static_cast<uint32_t>(inner), phaseRows, cappedBlockDim);
                    uint64_t alignedMinRows = 0;
                    uint64_t alignedTaskSize = 0;
                    uint64_t alignedMaxGroups = 0;
                    if (!GetCompactRaPhaseAlignedSplitStats(dim0, phaseRows, cappedBlockDim,
                        alignedMinRows, alignedTaskSize, alignedMaxGroups)) {
                        return ge::GRAPH_FAILED;
                    }
                    phaseBlockDim = cappedBlockDim;
                    phaseTaskSize = alignedTaskSize;
                    phaseTile = alignedPhaseTile;
                    compactRaPhaseAlignedSplit = true;
                    if (!SupportsCompactRaPhaseAligned(dim0, static_cast<uint32_t>(inner),
                        phaseRows, cappedBlockDim, alignedPhaseTile, &compactRaPhaseSegments,
                        &compactRaPhaseCheckpointGroups)) {
                        return ge::GRAPH_FAILED;
                    }
                } else {
                    if (!SupportsCompactRaPhase(dim0, static_cast<uint32_t>(inner), phaseRows,
                        cappedBlockDim, cappedPhaseTile)) {
                        cappedPhaseTile = SelectCompactRaPhaseSafeTile(cappedPhaseTile, dim0,
                            static_cast<uint32_t>(inner), elementSize, phaseRows, cappedBlockDim);
                    }
                    if (SupportsCompactRaPhase(dim0, static_cast<uint32_t>(inner), phaseRows,
                        cappedBlockDim, cappedPhaseTile)) {
                        phaseBlockDim = cappedBlockDim;
                        phaseTaskSize = cappedTaskSize;
                        phaseTile = cappedPhaseTile;
                    }
                }
            }
        }
        const uint64_t tiledRounds = tiledTile == 0 ? 0 : DivUp(taskSize, tiledTile);
        const bool tiledEligible = tiledTile >= compactRaPhase * MIN_COMPACT_RA_GROUPS &&
            tiledRounds >= MIN_RA_STAGGER_ROUNDS;
        // 持久相位累加优先；不满足其精度/容量门禁时回退到逐 tile 紧凑归约。
        if (phaseEligible) {
            blockDim = phaseBlockDim;
            taskSize = phaseTaskSize;
            compactRaMode = COMPACT_RA_PHASE;
            compactRaPhaseRows = phaseRows;
            if (compactRaPhaseSegments == 0) {
                compactRaPhaseSegments = 1;
            }
            inputBufferNum = phaseInputBufferNum;
            sharedTmpBytes = 0;
            tilePrimary = phaseTile;
            uint32_t waveSegments = 0;
            uint32_t waveCheckpointGroups = 0;
            if (ENABLE_COMPACT_RA_PHASE_WAVE && compactRaPhaseAlignedSplit &&
                streamingWorkload && inputBufferNum == 2 &&
                SupportsCompactRaPhaseWaveSchedule(dim0, static_cast<uint32_t>(inner),
                    compactRaPhaseRows, blockDim, tilePrimary, &waveSegments,
                    &waveCheckpointGroups) && waveSegments == compactRaPhaseSegments) {
                compactRaPhaseScheduleMode = COMPACT_RA_PHASE_SCHEDULE_WAVE;
                compactRaPhaseCheckpointGroups = waveCheckpointGroups;
            }
        } else if (tiledEligible) {
            compactRaMode = COMPACT_RA_TILED;
            sharedTmpBytes = static_cast<uint32_t>(gatherBytes);
            tilePrimary = tiledTile;
        } else {
            compactRa = false;
            inputBufferNum = 1;
            sharedTmpBytes = 0;
            tilePrimary = 0;
        }
    }
    if (compactAra) {
        alignedInner = static_cast<uint32_t>(inner);
        inputBufferNum = dim1 > 1 ? 2 : 1;
        uint64_t sharedTmpBytes64 = 0;
        if (MulOverflow(inner, sizeof(uint32_t), sharedTmpBytes64)) {
            return ge::GRAPH_FAILED;
        }
        sharedTmpBytes64 = AlignUp(sharedTmpBytes64, UB_BLOCK_BYTES);
        if (sharedTmpBytes64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        sharedTmpBytes = static_cast<uint32_t>(sharedTmpBytes64);
        const uint32_t phaseRows = UB_BLOCK_BYTES / elementSize /
            GcdU32(static_cast<uint32_t>(inner), UB_BLOCK_BYTES / elementSize);
        if (elementSize == 2 && taskSize >= phaseRows * MIN_COMPACT_RA_GROUPS) {
            tilePrimary = SelectCompactAraPhaseTile(ubSize, taskSize, alignedInner,
                elementSize, inputBufferNum, phaseRows);
            phaseAra = tilePrimary >= phaseRows;
            if (phaseAra) {
                sharedTmpBytes = 0;
            }
        }
        if (!phaseAra) {
            tilePrimary = SelectCompactTile(
                ubSize, taskSize, alignedInner, elementSize, inputBufferNum, sharedTmpBytes);
        }
        if (tilePrimary < compactRaPhase * MIN_COMPACT_RA_GROUPS) {
            compactAra = false;
            phaseAra = false;
            alignedInner = standardAlignedInner;
            inputBufferNum = 1;
            sharedTmpBytes = 0;
            tilePrimary = 0;
        }
    }
    if (tilePrimary == 0) {
        tilePrimary = SelectTilePrimary(
            ubSize, taskSize, alignedInner, elementSize, inputBufferNum, pattern, sharedTmpBytes);
    }
    if (!compactAr && !compactRar && !compactRa && !compactAra &&
        tilePrimary > 0 && topology != TOPOLOGY_GENERIC) {
        const bool supportsOverlap = topology == TOPOLOGY_AR || topology == TOPOLOGY_ARA ||
            topology == TOPOLOGY_RAR;
        const bool hasMultipleRounds = supportsOverlap && (taskSize > tilePrimary ||
            (topology == TOPOLOGY_RAR && dim0 > 1));
        if (hasMultipleRounds) {
            uint32_t doubleBufferTmpBytes = 0;
            const uint32_t doubleBufferTile = SelectTilePrimary(
                ubSize, taskSize, alignedInner, elementSize, 2, pattern, doubleBufferTmpBytes);
            if (doubleBufferTile > 0) {
                inputBufferNum = 2;
                tilePrimary = doubleBufferTile;
                sharedTmpBytes = doubleBufferTmpBytes;
            }
        }
    }
    if (tilePrimary == 0 && topology != TOPOLOGY_GENERIC) {
        topology = TOPOLOGY_GENERIC;
        compactAr = false;
        compactRar = false;
        compactRa = false;
        compactRaMode = 0;
        compactRaPhaseRows = 0;
        compactRaPhaseAlignedSplit = false;
        compactRaPhaseSegments = 0;
        compactRaPhaseCheckpointGroups = 0;
        compactRaPhaseScheduleMode = 0;
        compactAra = false;
        phaseAra = false;
        blockDim = 1;
        inner = shape[static_cast<size_t>(rank - 1)];
        taskSize = 1;
        pattern = AscendC::ReducePattern::AR;
        alignedInner64 = AlignUp(inner, alignElements);
        if (alignedInner64 == 0 || alignedInner64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        alignedInner = static_cast<uint32_t>(alignedInner64);
        tilePrimary = SelectTilePrimary(
            ubSize, taskSize, alignedInner, elementSize, 1, pattern, sharedTmpBytes);
        inputBufferNum = 1;
    }
    if (tilePrimary == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t raStaggerGroups = 0;
    if (topology == TOPOLOGY_RA && dataType == ge::DT_FLOAT &&
        compactRaMode != COMPACT_RA_PHASE && inputBufferNum == 1 &&
        streamingWorkload && SupportsRaStagger(dim0, blockDim, tilePrimary)) {
        raStaggerGroups = RA_STAGGER_GROUPS;
    }

    const bool streamInputOnce = phaseAra || compactRar ||
        compactRaMode == COMPACT_RA_PHASE;
    const bool arStreaming = topology == TOPOLOGY_AR && l2Size > 0 &&
        inputBytes >= l2Size / 4U;
    const bool bypassL2 = (streamInputOnce && streamingWorkload) || arStreaming;
    uint32_t rarScheduleMode = 0;
    if (compactRar && inputBufferNum == 2 && bypassL2 &&
        SupportsRarWaveSchedule(dim1, inner, elementSize, blockDim, tilePrimary, rarRowQuantum)) {
        rarScheduleMode = RAR_SCHEDULE_WAVE;
    }
    const bool compactArDirect = compactAr && dataType == ge::DT_BF16 &&
        inner <= std::numeric_limits<uint32_t>::max() && (inner % 8U) != 0 &&
        compactArPackStride >= 64U;
    SquareSumV1TilingData tiling;
    tiling.set_shape(shape.data());
    tiling.set_stride(stride.data());
    tiling.set_inputSize(inputSize);
    tiling.set_outputSize(outputSize);
    tiling.set_dim0(dim0);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_reduceSize(reduceSize);
    tiling.set_topology(topology);
    tiling.set_rank(static_cast<uint32_t>(rank));
    tiling.set_axisMask(axisMask);
    tiling.set_blockDim(blockDim);
    tiling.set_tilePrimary(tilePrimary);
    tiling.set_alignedInner(alignedInner);
    tiling.set_sharedTmpBytes(sharedTmpBytes);
    tiling.set_elementSize(elementSize);
    tiling.set_inputBufferNum(inputBufferNum);
    tiling.set_raStaggerGroups(raStaggerGroups);
    tiling.set_compactAr(compactAr ? 1U : 0U);
    tiling.set_compactRar(compactRar ? 1U : 0U);
    tiling.set_compactAra(phaseAra ? 2U : (compactAra ? 1U : 0U));
    // phase-ARA、phase-RA 与紧凑 RAR 均只顺序读取一次大输入，超过 L2 后避免无收益的缓存替换。
    tiling.set_bypassL2(bypassL2 ? 1U : 0U);
    tiling.set_rarScheduleMode(rarScheduleMode);
    tiling.set_compactRa(compactRaMode);
    tiling.set_compactRaPhaseRows(compactRaPhaseRows);
    tiling.set_compactRaPhaseAlignedSplit(compactRaPhaseAlignedSplit ? 1U : 0U);
    tiling.set_compactRaPhaseSegments(compactRaPhaseSegments);
    tiling.set_compactRaPhaseCheckpointGroups(compactRaPhaseCheckpointGroups);
    tiling.set_compactRaPhaseScheduleMode(compactRaPhaseScheduleMode);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    auto* rawTiling = context->GetRawTilingData();
    if (workspaceSize == nullptr || rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    if (topology == TOPOLOGY_RA) {
        const uint64_t workspaceStride = AlignUp(alignedInner * sizeof(float), 512);
        const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();
        uint64_t userWorkspace = 0;
        if (MulOverflow(workspaceStride, static_cast<uint64_t>(blockDim), userWorkspace) ||
            userWorkspace > std::numeric_limits<size_t>::max() - systemWorkspace) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_workspaceStride(workspaceStride);
        workspaceSize[0] = static_cast<size_t>(systemWorkspace + userWorkspace);
        context->SetTilingKey(GET_TPL_TILING_KEY(SYNC_TILING_KEY));
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        tiling.set_workspaceStride(0);
        workspaceSize[0] = 0;
        context->SetTilingKey(GET_TPL_TILING_KEY(compactArDirect ? COMPACT_AR_DIRECT_TILING_KEY : 0));
    }
    context->SetBlockDim(blockDim);
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
    const auto* axisAttr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    const bool* keepDims = context->GetAttrs()->GetAttrPointer<bool>(1);
    std::array<bool, optiling::SQUARE_SUM_MAX_RANK> reduced{};
    if (keepDims == nullptr || !ParseAxes(*inputShape, axisAttr, reduced)) {
        return ge::GRAPH_FAILED;
    }
    const int32_t rank = inputShape->GetDimNum();
    outputShape->SetDimNum(0);
    for (int32_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            if (*keepDims) {
                outputShape->AppendDim(1);
            }
        } else {
            outputShape->AppendDim(inputShape->GetDim(i));
        }
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
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}
