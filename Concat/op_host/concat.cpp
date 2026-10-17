#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t UB_RESERVE_BYTES = 8 * 1024;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t MIN_CORE_WORK_BYTES = 8 * 1024;
constexpr uint64_t DMA_CALL_EQUIVALENT_BYTES = 4 * 1024;
constexpr uint64_t DMA_BLOCK_EQUIVALENT_BYTES = DMA_CALL_EQUIVALENT_BYTES / 2;
// Grid 方案还需计入全体核的 DMA 描述符控制压力；权重小于最慢核项，
// 仅在最慢核代价接近时打破总命令数的平局。
constexpr uint64_t GRID_TOTAL_CALL_EQUIVALENT_BYTES = 128;
constexpr uint64_t TASK_EQUIVALENT_BYTES = 8 * 1024;
constexpr uint64_t GRID_MIN_ROW_BYTES = 64 * 1024;
constexpr uint64_t GRID_COLUMN_ALIGNMENT = 512;
constexpr uint64_t GRID_SPAN_OFFSET_LIMIT = 1ULL << 48;
constexpr uint64_t FLAT_CHUNK_ALIGNMENT = 512;
constexpr uint64_t DATA_COPY_MAX_ROWS = 4095;
constexpr uint64_t DATA_COPY_MAX_BLOCK_LEN = 2097151;
// 静态二维命令头的最高位仅表示输入为一次性流；blockLen 的有效位不超过 bit30。
constexpr uint64_t PREPARED_RECT_INPUT_L2_BYPASS_FLAG = 1ULL << 63;
constexpr uint64_t GATHER_DTYPE_BYTES = 4;
constexpr uint64_t GATHER_ELEMENT_ALIGNMENT = BLOCK_BYTES / GATHER_DTYPE_BYTES;
constexpr uint64_t GATHER_MAX_ELEMENTS_PER_CALL = 255 * 64;
constexpr uint64_t PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL = 255 * 64;
// 输出/offset 别名路径只常驻一行 base 和一行 stride；每批在 VECOUT 内重建完整表。
constexpr uint32_t OFFSET_OUTPUT_ALIAS_SEED_ROWS = 2;
constexpr uint64_t MICRO_GATHER_VF = 64;
constexpr uint64_t COPY_VECTOR_ELEMENTS = 64;
constexpr uint64_t COPY_MAX_REPEAT_TIMES = 255;
constexpr uint64_t COPY_MAX_REPEAT_STRIDE_BLOCKS = std::numeric_limits<uint16_t>::max();
// 二维输入带调度最多让一个连续带由两个行块共同处理；更大的核组会重新把连续 source
// 拆成过多小批，抵消窄段 MTE2 发射减少。
constexpr uint32_t BAND_GATHER_MAX_CORES_PER_BAND = 2;
constexpr uint64_t BAND_GATHER_MAX_WIDTH_FACTOR = 3;
constexpr uint64_t BAND_GATHER_REQUIRED_INPUT_CALL_PERCENT = 80;
constexpr uint64_t BAND_GATHER_MAX_OUTPUT_WORK_PERCENT = 125;
constexpr uint64_t BAND_GATHER_MAX_EXTRA_COMMANDS = 4;
constexpr uint64_t GATHERB_BLOCKS_PER_REPEAT = 8;
constexpr uint64_t GATHERB_ROW_BLOCK_ALIGNMENT = GATHERB_BLOCKS_PER_REPEAT;
constexpr uint64_t GATHERB_SOURCE_GUARD_ELEMENTS = 64;
// RowGather 需要摊销 offset 建表和 Gather 指令；最小工作集按 UB 可容纳的 tile 数推导。
constexpr uint64_t GATHER_MIN_UB_TILES = 4;
constexpr uint32_t GATHER_ROW_SEARCH_DIVISOR = 4;
// 双批平面将两个独立 GM→UB source 批压入同一二维 DMA 描述符。短输入场景中该描述符
// 发射位于关键 MTE2 路径，按比 dense Gather/MTE3 命令更高的控制权重建模。
constexpr uint64_t TWO_PLANE_SOURCE_CALL_WEIGHT = 3;
constexpr size_t MAX_SUPPORTED_RANK = 8;

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

bool AddOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return true;
    }
    result = lhs + rhs;
    return false;
}

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0;
    return AddOverflow(lhs, rhs, result) ? std::numeric_limits<uint64_t>::max() : result;
}

uint64_t SaturatingMul(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0;
    return MulOverflow(lhs, rhs, result) ? std::numeric_limits<uint64_t>::max() : result;
}

uint64_t GetGatherMinTotalBytes(uint64_t ubSize)
{
    // 至少覆盖一个完整的大行 DMA 域，避免在很小的输入上以建表成本替代直接搬运。
    return std::max<uint64_t>(GRID_MIN_ROW_BYTES,
        SaturatingMul(ubSize, GATHER_MIN_UB_TILES));
}

struct InputInfo {
    uint32_t index{0};
    uint64_t rowBytes{0};
    uint64_t outputCol{0};
};

struct LptTask {
    uint16_t input{0};
    uint16_t core{0};
    uint32_t rowStart{0};
    uint32_t rowCount{0};
    uint64_t weight{0};
};

enum class PlanKind : uint32_t {
    FLAT = 1,
    ROW = 2,
    GRID = 3,
    INPUT_LPT = 4,
};

struct Plan {
    PlanKind kind{PlanKind::ROW};
    uint64_t key{optiling::CONCAT_KEY_ROW};
    uint64_t cost{std::numeric_limits<uint64_t>::max()};
    uint32_t blockDim{1};
    uint32_t rowGroups{1};
    uint32_t colGroups{1};
    uint32_t tileBytes{0};
    uint64_t colChunkBytes{0};
    uint64_t coreChunkBytes{0};
    uint64_t maxCoreCalls{std::numeric_limits<uint64_t>::max()};
    uint64_t totalCoreCalls{std::numeric_limits<uint64_t>::max()};
    std::vector<LptTask> tasks;
};

struct GatherSubPlan {
    uint64_t score{std::numeric_limits<uint64_t>::max()};
    uint32_t rowsPerCore{0};
    uint32_t batchRows{0};
    uint32_t sourceBatchRows{0};
    uint32_t sourceBufferBytes{0};
    uint32_t outputBufferBytes{0};
    uint32_t offsetBufferBytes{0};
    // key31 使用两个互补分区；四个容量均是本 subPlan 所有 group 的最大值。
    uint32_t sourcePart0BufferBytes{0};
    uint32_t sourcePart1BufferBytes{0};
    uint32_t outputPart0BufferBytes{0};
    uint32_t outputPart1BufferBytes{0};
    std::vector<uint16_t> groupInputEnd;
    std::vector<uint16_t> groupInputSplit;
    // key38 将 group 按一个 32B 对齐输入边界拆为两段；切点复用 groupInputSplit。
    std::vector<uint16_t> groupInputSplit2;
    std::vector<uint32_t> groupElements;
    std::vector<uint32_t> groupSourceElements;
    // key42 仅在同一个已分配 UB 账本内为窄组增大批行数；普通路径保持统一值。
    std::vector<uint32_t> groupBatchRows;
    // key47 只为满足容量门槛的组记录连续后缀在输入/输出中的起点和 source 槽占用。
    std::vector<uint16_t> groupSuffixInputStart;
    std::vector<uint32_t> groupSuffixOutputStart;
    std::vector<uint32_t> groupSuffixElements;
    std::vector<uint32_t> groupSuffixSourceElements;
    // 仅 key19 的动态收益门槛使用；常规 RowGather 不依赖这些统计。
    uint64_t totalBlocks{0};
    uint64_t interiorBlocks{0};
    uint64_t boundaryPieces{0};
    uint32_t minNonzeroInputsPerGroup{std::numeric_limits<uint32_t>::max()};
};

struct RowGatherPlan {
    uint32_t highCoreCount{0};
    std::array<GatherSubPlan, optiling::CONCAT_GATHER_PLAN_COUNT> subPlans;
};

struct PackedRowPlan {
    uint32_t rowsPerCore{0};
    uint32_t outputRowElements{0};
    uint32_t sourceBufferBytes{0};
    uint32_t outputBufferBytes{0};
    uint32_t offsetBufferBytes{0};
    std::array<uint32_t, optiling::CONCAT_MAX_GRID_INPUTS> inputRowElements{};
};

// 一个二维任务只拥有一段连续输入和一段连续行，输出地址因此彼此不重叠。source 仍按
// 输入连续搬入，随后沿用已验证的 Gather + 整带 MTE3 写回数据流。
struct BandGatherTask {
    uint16_t inputStart{0};
    uint16_t inputEnd{0};
    uint32_t outputStart{0};
    uint32_t groupElements{0};
    uint32_t sourceElements{0};
    uint32_t batchRows{0};
    uint32_t sourceBufferBytes{0};
    uint32_t outputBufferBytes{0};
    uint32_t offsetBufferBytes{0};
    uint64_t rowStart{0};
    uint64_t rowCount{0};
    uint64_t inputCalls{0};
    uint64_t outputCalls{0};
    uint64_t gatherCalls{0};
    uint64_t outputElements{0};
    uint64_t criticalCost{std::numeric_limits<uint64_t>::max()};
};

struct BandGatherPlan {
    std::vector<BandGatherTask> tasks;
    uint64_t maxInputCalls{0};
    uint64_t maxOutputCalls{0};
    uint64_t maxGatherCalls{0};
    uint64_t maxOutputElements{0};
    uint64_t maxCriticalCost{0};
};

struct BandGatherBaseline {
    uint64_t maxInputCalls{0};
    uint64_t maxOutputCalls{0};
    uint64_t maxGatherCalls{0};
    uint64_t maxOutputElements{0};
    uint64_t maxCriticalCost{0};
};

struct PairGatherTask {
    uint16_t inputStart{0};
    uint16_t inputSplit{0};
    uint16_t inputEnd{0};
    uint32_t outputStart{0};
    uint32_t group0Elements{0};
    uint32_t group1Elements{0};
    uint32_t pairElements{0};
    uint32_t sourceElements{0};
    uint32_t batchRows{0};
    uint32_t sourceBufferBytes{0};
    uint32_t outputBufferBytes{0};
    uint32_t offsetBufferBytes{0};
    uint64_t rowStart{0};
    uint64_t rowCount{0};
};

struct PairGatherPlan {
    std::vector<PairGatherTask> tasks;
    uint64_t maxInputCalls{std::numeric_limits<uint64_t>::max()};
    uint64_t maxOutputCalls{std::numeric_limits<uint64_t>::max()};
    uint64_t maxGatherCalls{std::numeric_limits<uint64_t>::max()};
    uint64_t maxOutputElements{std::numeric_limits<uint64_t>::max()};
    uint64_t maxCriticalCost{std::numeric_limits<uint64_t>::max()};
};

struct BandGatherDpState {
    uint64_t maxCost{std::numeric_limits<uint64_t>::max()};
    uint64_t totalCost{std::numeric_limits<uint64_t>::max()};
    uint16_t previousInput{0};
    uint16_t previousCores{0};
    uint8_t allocatedCores{0};
    bool valid{false};
};

uint64_t EstimateSegmentCalls(uint64_t segmentBytes, uint64_t tileBytes)
{
    const uint64_t commandBytes = std::min(tileBytes, DATA_COPY_MAX_BLOCK_LEN);
    return segmentBytes == 0 || commandBytes == 0 ? 0 : CeilDiv(segmentBytes, commandBytes);
}

uint64_t EstimateRectCalls(uint64_t rowBytes, uint64_t rows, uint64_t tileBytes)
{
    if (rowBytes == 0 || rows == 0) {
        return 0;
    }
    const uint64_t alignedRowBytes = AlignUp(rowBytes, BLOCK_BYTES);
    if (alignedRowBytes > tileBytes) {
        return SaturatingMul(rows, CeilDiv(rowBytes, tileBytes));
    }
    const uint64_t rowsPerBatch = std::max<uint64_t>(1,
        std::min<uint64_t>(DATA_COPY_MAX_ROWS, tileBytes / alignedRowBytes));
    return CeilDiv(rows, rowsPerBatch);
}

uint64_t EstimateParallelCost(uint64_t totalBytes, uint64_t calls, uint32_t activeCores)
{
    const uint64_t cores = std::max<uint32_t>(activeCores, 1);
    const uint64_t transfer = CeilDiv(totalBytes, cores);
    const uint64_t launch = CeilDiv(SaturatingMul(calls, DMA_CALL_EQUIVALENT_BYTES), cores);
    return SaturatingAdd(transfer, launch);
}

Plan BuildFlatPlan(uint64_t totalBytes, uint32_t maxCores, uint64_t tileBytes)
{
    Plan plan;
    plan.kind = PlanKind::FLAT;
    plan.key = optiling::CONCAT_KEY_FLAT;
    const uint64_t usefulChunks = std::max<uint64_t>(CeilDiv(totalBytes, FLAT_CHUNK_ALIGNMENT), 1);
    plan.blockDim = static_cast<uint32_t>(std::min<uint64_t>(
        std::min<uint32_t>(maxCores, optiling::CONCAT_MAX_FLAT_CORES), usefulChunks));
    // 对齐可能让尾部若干核变空，收缩核数后重新计算，直到每个有效核都有数据。
    while (true) {
        plan.coreChunkBytes = AlignUp(CeilDiv(totalBytes, plan.blockDim), FLAT_CHUNK_ALIGNMENT);
        const uint32_t actualBlockDim = static_cast<uint32_t>(std::max<uint64_t>(1,
            CeilDiv(totalBytes, plan.coreChunkBytes)));
        if (actualBlockDim >= plan.blockDim) {
            break;
        }
        plan.blockDim = actualBlockDim;
    }
    uint64_t calls = CeilDiv(totalBytes, tileBytes);
    calls = SaturatingAdd(calls, plan.blockDim);
    plan.cost = EstimateParallelCost(totalBytes, calls, plan.blockDim);
    return plan;
}

bool BuildFlatCoreStarts(const std::vector<InputInfo>& inputs, const Plan& plan, uint64_t totalBytes,
    uint32_t inputCount, std::array<uint32_t, optiling::CONCAT_MAX_FLAT_CORES>& coreStartInput,
    std::array<uint64_t, optiling::CONCAT_MAX_FLAT_CORES>& coreStartInputOffset)
{
    coreStartInput.fill(inputCount);
    coreStartInputOffset.fill(0);
    if (totalBytes == 0) {
        return true;
    }

    size_t inputCursor = 0;
    uint64_t inputBegin = 0;
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        const uint64_t coreStart = SaturatingMul(core, plan.coreChunkBytes);
        if (coreStart >= totalBytes) {
            return false;
        }
        while (inputCursor < inputs.size()) {
            uint64_t inputEnd = 0;
            if (AddOverflow(inputBegin, inputs[inputCursor].rowBytes, inputEnd)) {
                return false;
            }
            if (coreStart < inputEnd) {
                break;
            }
            inputBegin = inputEnd;
            ++inputCursor;
        }
        if (inputCursor >= inputs.size()) {
            return false;
        }
        coreStartInput[core] = inputs[inputCursor].index;
        coreStartInputOffset[core] = coreStart - inputBegin;
    }
    return true;
}

uint64_t CountFlatRangeCalls(const std::vector<InputInfo>& inputs, uint64_t begin,
    uint64_t end, uint64_t tileBytes)
{
    uint64_t calls = 0;
    for (const InputInfo& input : inputs) {
        const uint64_t inputEnd = SaturatingAdd(input.outputCol, input.rowBytes);
        const uint64_t segmentBegin = std::max(begin, input.outputCol);
        const uint64_t segmentEnd = std::min(end, inputEnd);
        if (segmentEnd > segmentBegin) {
            calls = SaturatingAdd(calls,
                EstimateSegmentCalls(segmentEnd - segmentBegin, tileBytes));
        }
    }
    return calls;
}

bool BuildGreedyFlatIntervals(const std::vector<InputInfo>& inputs, uint64_t totalBytes,
    uint64_t tileBytes, uint32_t coreLimit, uint64_t maxBytes, uint64_t callLimit,
    std::vector<uint64_t>& starts)
{
    starts.clear();
    starts.push_back(0);
    uint64_t begin = 0;
    while (begin < totalBytes) {
        const uint64_t upper = std::min(totalBytes, SaturatingAdd(begin, maxBytes));
        const uint64_t alignedUpper = AlignDown(upper, FLAT_CHUNK_ALIGNMENT);
        uint64_t bestEnd = begin;
        if (alignedUpper > begin) {
            uint64_t low = 1;
            uint64_t high = (alignedUpper - begin) / FLAT_CHUNK_ALIGNMENT;
            while (low <= high) {
                const uint64_t middle = low + (high - low) / 2;
                const uint64_t candidate = begin + middle * FLAT_CHUNK_ALIGNMENT;
                if (CountFlatRangeCalls(inputs, begin, candidate, tileBytes) <= callLimit) {
                    bestEnd = candidate;
                    low = middle + 1;
                } else {
                    high = middle - 1;
                }
            }
        }
        if (upper == totalBytes &&
            CountFlatRangeCalls(inputs, begin, totalBytes, tileBytes) <= callLimit) {
            bestEnd = totalBytes;
        }
        if (bestEnd == begin) {
            return false;
        }
        starts.push_back(bestEnd);
        begin = bestEnd;
        if (starts.size() - 1 > coreLimit) {
            return false;
        }
    }
    return true;
}

bool BuildPreparedFlatCoreStarts(const std::vector<InputInfo>& inputs, const Plan& plan,
    uint64_t totalBytes, uint64_t tileBytes,
    std::array<uint64_t, optiling::CONCAT_MAX_FLAT_CORES>& coreOutputStart)
{
    coreOutputStart.fill(0);
    if (totalBytes == 0 || plan.blockDim == 0 || plan.blockDim > optiling::CONCAT_MAX_FLAT_CORES) {
        return totalBytes == 0;
    }

    uint64_t baselineMaxCalls = 0;
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        const uint64_t begin = SaturatingMul(core, plan.coreChunkBytes);
        const uint64_t end = std::min(totalBytes, SaturatingAdd(begin, plan.coreChunkBytes));
        baselineMaxCalls = std::max(baselineMaxCalls,
            CountFlatRangeCalls(inputs, begin, end, tileBytes));
    }
    const uint64_t baselineScore = SaturatingAdd(plan.coreChunkBytes,
        SaturatingMul(baselineMaxCalls, DMA_CALL_EQUIVALENT_BYTES));

    uint64_t bestScore = baselineScore;
    std::vector<uint64_t> bestStarts;
    for (uint64_t callLimit = 1; callLimit <= baselineMaxCalls; ++callLimit) {
        uint64_t lowUnits = std::max<uint64_t>(1,
            CeilDiv(CeilDiv(totalBytes, plan.blockDim), FLAT_CHUNK_ALIGNMENT));
        uint64_t highUnits = std::max<uint64_t>(lowUnits,
            CeilDiv(totalBytes, FLAT_CHUNK_ALIGNMENT));
        std::vector<uint64_t> candidateStarts;
        while (lowUnits < highUnits) {
            const uint64_t middleUnits = lowUnits + (highUnits - lowUnits) / 2;
            const uint64_t maxBytes = SaturatingMul(middleUnits, FLAT_CHUNK_ALIGNMENT);
            if (BuildGreedyFlatIntervals(inputs, totalBytes, tileBytes, plan.blockDim,
                maxBytes, callLimit, candidateStarts)) {
                highUnits = middleUnits;
            } else {
                lowUnits = middleUnits + 1;
            }
        }
        const uint64_t maxBytes = SaturatingMul(lowUnits, FLAT_CHUNK_ALIGNMENT);
        if (!BuildGreedyFlatIntervals(inputs, totalBytes, tileBytes, plan.blockDim,
            maxBytes, callLimit, candidateStarts)) {
            continue;
        }
        const uint64_t score = SaturatingAdd(maxBytes,
            SaturatingMul(callLimit, DMA_CALL_EQUIVALENT_BYTES));
        if (score < bestScore) {
            bestScore = score;
            bestStarts = std::move(candidateStarts);
        }
    }

    if (bestStarts.empty()) {
        bestStarts.reserve(plan.blockDim + 1);
        for (uint32_t core = 0; core < plan.blockDim; ++core) {
            bestStarts.push_back(SaturatingMul(core, plan.coreChunkBytes));
        }
        bestStarts.push_back(totalBytes);
    }

    while (bestStarts.size() - 1 < plan.blockDim) {
        size_t splitIndex = bestStarts.size();
        uint64_t splitPoint = 0;
        uint64_t splitBytes = 0;
        for (size_t i = 0; i + 1 < bestStarts.size(); ++i) {
            const uint64_t begin = bestStarts[i];
            const uint64_t end = bestStarts[i + 1];
            const uint64_t middle = AlignDown(begin + (end - begin) / 2, FLAT_CHUNK_ALIGNMENT);
            if (middle > begin && middle < end && end - begin > splitBytes) {
                splitIndex = i + 1;
                splitPoint = middle;
                splitBytes = end - begin;
            }
        }
        if (splitIndex == bestStarts.size()) {
            return false;
        }
        bestStarts.insert(bestStarts.begin() + splitIndex, splitPoint);
    }
    if (bestStarts.size() != static_cast<size_t>(plan.blockDim) + 1) {
        return false;
    }
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        coreOutputStart[core] = bestStarts[core];
    }
    return true;
}

bool BuildPreparedFlatTasks(const std::vector<InputInfo>& inputs, const Plan& plan, uint64_t totalBytes,
    uint64_t tileBytes,
    std::array<uint64_t, optiling::CONCAT_MAX_FLAT_CORES>& coreOutputStart,
    std::array<uint16_t, optiling::CONCAT_MAX_FLAT_CORES>& coreTaskEnd,
    std::array<uint16_t, optiling::CONCAT_MAX_PREPARED_FLAT_TASKS>& taskInput,
    std::array<uint64_t, optiling::CONCAT_MAX_PREPARED_FLAT_TASKS>& taskInputOffset,
    std::array<uint64_t, optiling::CONCAT_MAX_PREPARED_FLAT_TASKS>& taskBytes)
{
    coreTaskEnd.fill(0);
    taskInput.fill(0);
    taskInputOffset.fill(0);
    taskBytes.fill(0);
    if (totalBytes == 0) {
        return true;
    }
    if (!BuildPreparedFlatCoreStarts(inputs, plan, totalBytes, tileBytes, coreOutputStart)) {
        return false;
    }

    size_t inputCursor = 0;
    uint64_t inputBegin = 0;
    uint32_t taskCount = 0;
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        const uint64_t coreStart = coreOutputStart[core];
        const uint64_t coreEnd = core + 1 < plan.blockDim ? coreOutputStart[core + 1] : totalBytes;
        uint64_t outputOffset = coreStart;
        while (outputOffset < coreEnd) {
            uint64_t inputEnd = 0;
            while (inputCursor < inputs.size()) {
                if (AddOverflow(inputBegin, inputs[inputCursor].rowBytes, inputEnd)) {
                    return false;
                }
                if (outputOffset < inputEnd) {
                    break;
                }
                inputBegin = inputEnd;
                ++inputCursor;
            }
            if (inputCursor >= inputs.size() ||
                taskCount >= optiling::CONCAT_MAX_PREPARED_FLAT_TASKS ||
                inputs[inputCursor].index > std::numeric_limits<uint16_t>::max()) {
                return false;
            }

            const uint64_t segmentEnd = std::min(coreEnd, inputEnd);
            const uint64_t copyBytes = segmentEnd - outputOffset;
            if (copyBytes == 0) {
                return false;
            }
            taskInput[taskCount] = static_cast<uint16_t>(inputs[inputCursor].index);
            taskInputOffset[taskCount] = outputOffset - inputBegin;
            taskBytes[taskCount] = copyBytes;
            ++taskCount;
            outputOffset = segmentEnd;
        }
        coreTaskEnd[core] = static_cast<uint16_t>(taskCount);
    }
    return true;
}

bool BuildPreparedFlatTwoCommands(const std::vector<InputInfo>& inputs, const Plan& plan,
    uint64_t totalBytes, uint64_t tileBytes,
    std::array<uint64_t, optiling::CONCAT_MAX_FLAT_CORES>& coreOutputStart,
    std::array<uint16_t, optiling::CONCAT_MAX_PREPARED_FLAT_COMMANDS>& commandInput,
    std::array<uint64_t, optiling::CONCAT_MAX_PREPARED_FLAT_COMMANDS>& commandInputOffset,
    std::array<uint64_t, optiling::CONCAT_MAX_PREPARED_FLAT_COMMANDS>& commandOutputOffset,
    std::array<uint32_t, optiling::CONCAT_MAX_PREPARED_FLAT_COMMANDS>& commandBytes)
{
    commandInput.fill(0);
    commandInputOffset.fill(0);
    commandOutputOffset.fill(0);
    commandBytes.fill(0);
    const uint64_t commandLimit = std::min(tileBytes, DATA_COPY_MAX_BLOCK_LEN);
    if (totalBytes == 0 || commandLimit == 0 || plan.blockDim == 0 ||
        plan.blockDim > optiling::CONCAT_MAX_FLAT_CORES ||
        !BuildPreparedFlatCoreStarts(inputs, plan, totalBytes, tileBytes, coreOutputStart)) {
        return false;
    }
    size_t inputCursor = 0;
    uint64_t inputBegin = 0;
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        const uint64_t coreStart = coreOutputStart[core];
        const uint64_t coreEnd = core + 1 < plan.blockDim ? coreOutputStart[core + 1] : totalBytes;
        uint64_t outputOffset = coreStart;
        uint32_t localCommandCount = 0;
        while (outputOffset < coreEnd) {
            uint64_t inputEnd = 0;
            while (inputCursor < inputs.size()) {
                if (AddOverflow(inputBegin, inputs[inputCursor].rowBytes, inputEnd)) {
                    return false;
                }
                if (outputOffset < inputEnd) {
                    break;
                }
                inputBegin = inputEnd;
                ++inputCursor;
            }
            if (inputCursor >= inputs.size() ||
                inputs[inputCursor].index > std::numeric_limits<uint16_t>::max()) {
                return false;
            }

            const uint64_t segmentEnd = std::min(coreEnd, inputEnd);
            while (outputOffset < segmentEnd) {
                if (localCommandCount >= optiling::CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE) {
                    return false;
                }
                const uint64_t copyBytes = std::min(segmentEnd - outputOffset, commandLimit);
                if (copyBytes == 0 || copyBytes > std::numeric_limits<uint32_t>::max()) {
                    return false;
                }
                const uint32_t command =
                    core * optiling::CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE + localCommandCount;
                commandInput[command] = static_cast<uint16_t>(inputs[inputCursor].index);
                commandInputOffset[command] = outputOffset - inputBegin;
                commandOutputOffset[command] = outputOffset;
                commandBytes[command] = static_cast<uint32_t>(copyBytes);
                outputOffset += copyBytes;
                ++localCommandCount;
            }
        }
        const uint32_t firstCommand = core * optiling::CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE;
        const uint64_t recordedBytes = static_cast<uint64_t>(commandBytes[firstCommand]) +
            commandBytes[firstCommand + 1];
        if (localCommandCount != optiling::CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE ||
            recordedBytes != coreEnd - coreStart) {
            return false;
        }
        if (commandBytes[firstCommand] < commandBytes[firstCommand + 1]) {
            // 两级搬运流水优先发射大块，显式目的偏移保证交换后仍写回原区间。
            std::swap(commandInput[firstCommand], commandInput[firstCommand + 1]);
            std::swap(commandInputOffset[firstCommand], commandInputOffset[firstCommand + 1]);
            std::swap(commandOutputOffset[firstCommand], commandOutputOffset[firstCommand + 1]);
            std::swap(commandBytes[firstCommand], commandBytes[firstCommand + 1]);
        }
    }
    return true;
}

Plan BuildRowPlan(uint64_t outer, uint64_t totalBytes, uint32_t maxCores, uint64_t tileBytes,
    const std::vector<InputInfo>& inputs)
{
    Plan plan;
    plan.kind = PlanKind::ROW;
    plan.key = optiling::CONCAT_KEY_ROW;
    plan.blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(maxCores, outer)));
    const uint64_t maxRowsPerCore = CeilDiv(outer, plan.blockDim);
    uint64_t maxCoreCalls = 0;
    for (const InputInfo& input : inputs) {
        maxCoreCalls = SaturatingAdd(maxCoreCalls,
            EstimateRectCalls(input.rowBytes, maxRowsPerCore, tileBytes));
    }
    const uint64_t maxCoreBytes = SaturatingMul(maxRowsPerCore,
        outer == 0 ? 0 : totalBytes / outer);
    plan.cost = SaturatingAdd(maxCoreBytes,
        SaturatingMul(maxCoreCalls, DMA_CALL_EQUIVALENT_BYTES));
    return plan;
}

bool ShouldBypassRowInputL2(const Plan& plan, uint64_t outer, uint64_t outputRowBytes,
    uint64_t tileBytes)
{
    if (plan.kind != PlanKind::ROW || plan.key != optiling::CONCAT_KEY_ROW ||
        plan.blockDim == 0 || tileBytes == 0) {
        return false;
    }
    // ROW 路径的输入只按行顺序消费一次。单核数据尚未跨越完整双缓冲窗口时，
    // 保留默认 L2 策略，避免 cache hint 的固定代价主导小工作集。
    const uint64_t maxCoreRows = CeilDiv(outer, plan.blockDim);
    const uint64_t maxCoreBytes = SaturatingMul(maxCoreRows, outputRowBytes);
    // 单行任务没有跨行的持续流式消费；此时 cache hint 的固定代价和已有 L2 驻留
    // 往往比绕过收益更明显。至少两个完整行 tile 才启用该输入策略。
    return maxCoreRows >= DOUBLE_BUFFER_NUM &&
        maxCoreBytes >= SaturatingMul(tileBytes, DOUBLE_BUFFER_NUM);
}

bool ShouldUsePrecomputedRowBytes(const Plan& plan, bool bypassRowInputL2,
    ge::DataType dataType, const std::vector<InputInfo>& inputs)
{
    // 仅让带有 key49 内核的 INT32 编译单元命中。门槛完全由动态计划、dtype 与
    // TilingData 容量决定；行宽超过 uint32_t 时保留原始描述符解析路径。
    if (dataType != ge::DT_INT32 || !bypassRowInputL2 || plan.kind != PlanKind::ROW ||
        plan.key != optiling::CONCAT_KEY_ROW || inputs.empty() ||
        inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS) {
        return false;
    }
    for (const InputInfo& input : inputs) {
        if (input.index >= optiling::CONCAT_MAX_GRID_INPUTS ||
            input.rowBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    return true;
}

bool ShouldUseSingleBufferRowBytes(const Plan& plan, bool usePrecomputedRowBytes,
    uint64_t outer, uint32_t tileBytes, uint32_t singleTileBytes,
    const std::vector<InputInfo>& inputs)
{
    if (!usePrecomputedRowBytes || plan.kind != PlanKind::ROW ||
        plan.key != optiling::CONCAT_KEY_ROW || plan.blockDim == 0 || outer == 0 ||
        tileBytes == 0 || singleTileBytes <= tileBytes || inputs.empty()) {
        return false;
    }
    const uint64_t rowsPerCore = CeilDiv(outer, plan.blockDim);
    if (rowsPerCore < 2) {
        return false;
    }
    uint64_t doubleBufferCalls = 0;
    uint64_t singleBufferCalls = 0;
    for (const InputInfo& input : inputs) {
        doubleBufferCalls = SaturatingAdd(doubleBufferCalls,
            EstimateRectCalls(input.rowBytes, rowsPerCore, tileBytes));
        singleBufferCalls = SaturatingAdd(singleBufferCalls,
            EstimateRectCalls(input.rowBytes, rowsPerCore, singleTileBytes));
    }
    // 单槽会放弃跨块 MTE2/MTE3 重叠，只有实际减少二维 DMA 批次才值得切换。
    return singleBufferCalls < doubleBufferCalls;
}

// 对小型、完整均分的 ROW，将一个核负责的所有输入段先搬入连续 source UB，
// 再由 Gather 组装成完整输出行并一次写回。条件只来自运行时 dtype、行/核拓扑、
// UB 容量、对齐和 DMA/Gather 编码范围；不按公开 case 或输入序号分派。
bool BuildPackedRowPlan(const Plan& plan, uint64_t outer, uint64_t outputRowBytes,
    uint32_t blockDim, uint64_t ubSize, uint32_t dtypeSize,
    const std::vector<InputInfo>& inputs, uint32_t tileBytes, PackedRowPlan& packed)
{
    packed = PackedRowPlan{};
    constexpr uint32_t metadataElementBytes = sizeof(int32_t);
    if (dtypeSize != 2 || plan.kind != PlanKind::ROW || plan.key != optiling::CONCAT_KEY_ROW ||
        blockDim == 0 || outer == 0 || outer % blockDim != 0 || inputs.size() >
        optiling::CONCAT_MAX_GRID_INPUTS || outputRowBytes == 0 || outputRowBytes % BLOCK_BYTES != 0 ||
        outputRowBytes % dtypeSize != 0 || outputRowBytes > DATA_COPY_MAX_BLOCK_LEN ||
        tileBytes == 0) {
        return false;
    }

    const uint64_t rowsPerCore64 = outer / blockDim;
    const uint64_t outputElements64 = outputRowBytes / dtypeSize;
    const uint64_t elementAlignment = BLOCK_BYTES / dtypeSize;
    if (rowsPerCore64 < 2 || rowsPerCore64 > DATA_COPY_MAX_ROWS || outputElements64 == 0 ||
        elementAlignment == 0 || outputElements64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t alignedOutputElements = AlignUp(outputElements64, elementAlignment);
    const uint64_t totalGatherElements = SaturatingMul(rowsPerCore64, alignedOutputElements);
    // 保守使用 32-bit Gather 的单次上限，避免不同 dtype 编译单元间的 repeat 语义漂移。
    if (alignedOutputElements > std::numeric_limits<uint32_t>::max() ||
        totalGatherElements == 0 || totalGatherElements > PACKED_ROW_GATHER_MAX_ELEMENTS_PER_CALL) {
        return false;
    }

    uint64_t sourceElements = 0;
    uint32_t nonzeroInputs = 0;
    uint64_t baselineOutputCalls = 0;
    for (const InputInfo& input : inputs) {
        if (input.index >= optiling::CONCAT_MAX_GRID_INPUTS || input.rowBytes % dtypeSize != 0 ||
            input.rowBytes / dtypeSize > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t inputElements = input.rowBytes / dtypeSize;
        packed.inputRowElements[input.index] = static_cast<uint32_t>(inputElements);
        if (inputElements == 0) {
            continue;
        }
        ++nonzeroInputs;
        const uint64_t payloadElements = SaturatingMul(rowsPerCore64, inputElements);
        const uint64_t payloadBytes = SaturatingMul(payloadElements, dtypeSize);
        if (payloadElements == 0 || payloadBytes > DATA_COPY_MAX_BLOCK_LEN ||
            payloadBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        sourceElements = SaturatingAdd(sourceElements, AlignUp(payloadElements, elementAlignment));
        baselineOutputCalls = SaturatingAdd(baselineOutputCalls,
            EstimateRectCalls(input.rowBytes, rowsPerCore64, tileBytes));
    }
    if (nonzeroInputs < 4 || baselineOutputCalls <= 1 || sourceElements == 0 ||
        sourceElements > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const uint64_t sourceBytes = SaturatingMul(sourceElements, dtypeSize);
    const uint64_t outputBytes = SaturatingMul(totalGatherElements, dtypeSize);
    const uint64_t offsetBytes = SaturatingMul(totalGatherElements, metadataElementBytes);
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(sourceBytes, SaturatingAdd(outputBytes, offsetBytes)));
    if (sourceBytes == 0 || outputBytes == 0 || offsetBytes == 0 ||
        sourceBytes > std::numeric_limits<uint32_t>::max() ||
        outputBytes > std::numeric_limits<uint32_t>::max() ||
        offsetBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }

    packed.rowsPerCore = static_cast<uint32_t>(rowsPerCore64);
    packed.outputRowElements = static_cast<uint32_t>(outputElements64);
    packed.sourceBufferBytes = static_cast<uint32_t>(sourceBytes);
    packed.outputBufferBytes = static_cast<uint32_t>(outputBytes);
    packed.offsetBufferBytes = static_cast<uint32_t>(offsetBytes);
    return true;
}

bool ShouldBypassPreparedRectInputL2(const Plan& plan, uint64_t outer, ge::DataType dataType)
{
    if (plan.kind != PlanKind::GRID ||
        plan.key != optiling::CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER ||
        plan.rowGroups == 0 || plan.colGroups < 2 || plan.tileBytes == 0 ||
        // key48 只编译在 FP32 单元中；INT32 虽同为 4 字节，也不能复用该 key。
        dataType != ge::DT_FLOAT) {
        return false;
    }
    // Prepared-rect 的每个核只读一次自己的列区间；工作量达到完整单缓冲 tile 后，
    // 禁止填充 L2 才能避免缓存分配与回写污染后续的独立流。
    const uint64_t maxCoreRows = CeilDiv(outer, plan.rowGroups);
    const uint64_t maxCoreBytes = SaturatingMul(maxCoreRows, plan.colChunkBytes);
    return maxCoreRows != 0 && maxCoreBytes >= plan.tileBytes;
}

bool BuildGatherGroupsWithBufferCounts(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint32_t sourceBatchRows, uint64_t ubSize, uint32_t sourceBufferCount,
    uint32_t matrixBufferCount, GatherSubPlan& plan,
    uint32_t persistentOffsetRows = 0)
{
    if (batchRows == 0 || sourceBatchRows == 0 || sourceBufferCount == 0 ||
        matrixBufferCount == 0) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    plan.groupBatchRows.clear();
    plan.groupSuffixInputStart.clear();
    plan.groupSuffixOutputStart.clear();
    plan.groupSuffixElements.clear();
    plan.groupSuffixSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t sumAlignedElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;
    uint64_t maxOffsetBytes = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t offsetBytes = persistentOffsetRows == 0 ? outputBytes :
            SaturatingMul(persistentOffsetRows,
                SaturatingMul(alignedElements, GATHER_DTYPE_BYTES));
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        plan.groupBatchRows.push_back(batchRows);
        plan.groupSuffixInputStart.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupSuffixOutputStart.push_back(0);
        plan.groupSuffixElements.push_back(0);
        plan.groupSuffixSourceElements.push_back(0);
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        maxOffsetBytes = std::max(maxOffsetBytes, offsetBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputSourceElements = AlignUp(
            SaturatingMul(sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t candidateAligned = AlignUp(candidateElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateMatrixBytes = SaturatingMul(
            SaturatingMul(batchRows, candidateAligned), GATHER_DTYPE_BYTES);
        const uint64_t candidateSourceBytes = SaturatingMul(
            candidateSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t candidateOffsetBytes = persistentOffsetRows == 0 ? 0 :
            SaturatingMul(persistentOffsetRows,
                SaturatingMul(candidateAligned, GATHER_DTYPE_BYTES));
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(SaturatingMul(sourceBufferCount, candidateSourceBytes),
                SaturatingAdd(SaturatingMul(matrixBufferCount, candidateMatrixBytes),
                    candidateOffsetBytes)));
        if (groupElements != 0 && requiredBytes > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t matrixBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t offsetBytes = persistentOffsetRows == 0 ? 0 :
            SaturatingMul(persistentOffsetRows,
                SaturatingMul(alignedElements, GATHER_DTYPE_BYTES));
        const uint64_t singleRequiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(SaturatingMul(sourceBufferCount, sourceBytes),
                SaturatingAdd(SaturatingMul(matrixBufferCount, matrixBytes), offsetBytes)));
        if (singleRequiredBytes > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    const uint64_t sourceFullBatches = plan.rowsPerCore / sourceBatchRows;
    const uint64_t sourceTailRows = plan.rowsPerCore % sourceBatchRows;
    const uint64_t sourceBatches = sourceFullBatches +
        static_cast<uint64_t>(sourceTailRows != 0);
    const uint64_t outputFullBatches = plan.rowsPerCore / batchRows;
    const uint64_t outputTailRows = plan.rowsPerCore % batchRows;
    const uint64_t outputBatches = outputFullBatches +
        static_cast<uint64_t>(outputTailRows != 0);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), outputBatches);
    uint64_t gatherCalls = 0;
    for (uint32_t elements : plan.groupElements) {
        const uint64_t alignedElements = AlignUp(elements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t fullBatchCalls = CeilDiv(
            SaturatingMul(batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
        gatherCalls = SaturatingAdd(gatherCalls,
            SaturatingMul(outputFullBatches, fullBatchCalls));
        if (outputTailRows != 0) {
            gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
                SaturatingMul(outputTailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
        }
    }
    const uint64_t outputBlocks = SaturatingMul(plan.rowsPerCore, plan.groupInputEnd.size());
    const uint64_t offsetWork = SaturatingMul(
        SaturatingMul(sumAlignedElements, batchRows + 1), 3 * sizeof(int32_t));
    plan.score = SaturatingAdd(
        SaturatingAdd(SaturatingMul(
            SaturatingAdd(inputCalls, SaturatingAdd(outputCalls, gatherCalls)),
            DMA_CALL_EQUIVALENT_BYTES),
            SaturatingMul(outputBlocks, DMA_BLOCK_EQUIVALENT_BYTES)), offsetWork);
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(SaturatingMul(sourceBufferCount, maxSourceBytes),
            SaturatingAdd(SaturatingMul(matrixBufferCount, maxOutputBytes),
                persistentOffsetRows == 0 ? 0 : maxOffsetBytes)));
    if (maxSourceBytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() ||
        maxOffsetBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }
    plan.batchRows = batchRows;
    plan.sourceBatchRows = sourceBatchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = static_cast<uint32_t>(maxOffsetBytes);
    return true;
}

bool BuildGatherGroups(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint64_t ubSize, GatherSubPlan& plan)
{
    // 单 source、一个 output 和一个 offset 表是常规 RowGather 的 UB 布局。
    return BuildGatherGroupsWithBufferCounts(inputs, batchRows, batchRows, ubSize, 1, 2, plan);
}

// masked Copy 路径让每个输入在 UB 中按输出段的 32B 相位留出左/右 padding。GM 侧仍按行连续，
// 每行的有效元素可由同一条 Copy repeat 写入对齐的输出行。
bool BuildMaskedCopyGroups(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint64_t ubSize, GatherSubPlan& plan)
{
    if (batchRows == 0 || batchRows > COPY_MAX_REPEAT_TIMES) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;
    uint64_t copyCallsPerBatch = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        if (alignedElements / GATHER_ELEMENT_ALIGNMENT > COPY_MAX_REPEAT_STRIDE_BLOCKS) {
            return false;
        }
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t phase = groupElements % GATHER_ELEMENT_ALIGNMENT;
        const uint64_t sourceStrideElements = inputElements == 0 ? 0 :
            AlignUp(SaturatingAdd(phase, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t inputSourceElements = SaturatingMul(batchRows, sourceStrideElements);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t candidateAligned = AlignUp(candidateElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateSourceBytes = SaturatingMul(
            candidateSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t candidateOutputBytes = SaturatingMul(
            SaturatingMul(batchRows, candidateAligned), GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(candidateSourceBytes, candidateOutputBytes));
        if (groupElements != 0 && requiredBytes > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        const uint64_t groupPhase = groupElements % GATHER_ELEMENT_ALIGNMENT;
        const uint64_t groupStrideElements = inputElements == 0 ? 0 :
            AlignUp(SaturatingAdd(groupPhase, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t groupInputSourceElements = SaturatingMul(batchRows, groupStrideElements);
        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, groupInputSourceElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        if (alignedElements / GATHER_ELEMENT_ALIGNMENT > COPY_MAX_REPEAT_STRIDE_BLOCKS ||
            groupStrideElements / GATHER_ELEMENT_ALIGNMENT > COPY_MAX_REPEAT_STRIDE_BLOCKS ||
            SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(sourceBytes, outputBytes)) > ubSize) {
            return false;
        }
        if (inputElements != 0) {
            copyCallsPerBatch = SaturatingAdd(copyCallsPerBatch,
                CeilDiv(SaturatingAdd(groupPhase, inputElements), COPY_VECTOR_ELEMENTS));
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    const uint64_t batches = CeilDiv(plan.rowsPerCore, batchRows);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t inputCalls = SaturatingMul(nonzeroInputs, batches);
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), batches);
    // Copy 是 V pipe 指令，不把它伪装成 MTE DMA；以一个 32B block 的等价开销参与批次选择。
    plan.score = SaturatingAdd(
        SaturatingMul(SaturatingAdd(inputCalls, outputCalls), DMA_CALL_EQUIVALENT_BYTES),
        SaturatingMul(SaturatingMul(copyCallsPerBatch, batches), BLOCK_BYTES));
    if (maxSourceBytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() ||
        SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(maxSourceBytes, maxOutputBytes)) > ubSize) {
        return false;
    }
    plan.batchRows = batchRows;
    plan.sourceBatchRows = batchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = 0;
    return true;
}

bool BuildMaskedCopySubPlan(const std::vector<InputInfo>& inputs, uint32_t rowsPerCore,
    uint64_t ubSize, GatherSubPlan& plan)
{
    if (rowsPerCore == 0) {
        return false;
    }
    const uint32_t maxBatchRows = static_cast<uint32_t>(std::min<uint64_t>(
        rowsPerCore, COPY_MAX_REPEAT_TIMES));
    GatherSubPlan best;
    for (uint32_t batchRows = maxBatchRows; batchRows != 0; --batchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = rowsPerCore;
        if (!BuildMaskedCopyGroups(inputs, batchRows, ubSize, candidate)) {
            continue;
        }
        if (candidate.score < best.score ||
            (candidate.score == best.score && candidate.batchRows > best.batchRows)) {
            best = std::move(candidate);
        }
    }
    if (best.score == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    plan = std::move(best);
    return true;
}

bool BuildMaskedCopyRowPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildMaskedCopySubPlan(inputs, baseline.subPlans[0].rowsPerCore, ubSize, lowPlan) ||
        !BuildMaskedCopySubPlan(inputs, baseline.subPlans[1].rowsPerCore, ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

// 直接 MTE 路径只保留按输入连续排列的 source。每个输入在 UB 中独占 32B 对齐起点，
// 随后直接使用二维 MTE3 写入输出行，因而不再需要 output/offset 矩阵缓冲。
bool BuildDirectMteGroups(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint64_t ubSize, GatherSubPlan& plan)
{
    if (batchRows == 0) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t maxSourceBytes = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || groupSourceElements == 0 ||
            inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputSourceElements = AlignUp(
            SaturatingMul(batchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements,
            inputSourceElements);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingMul(candidateSourceElements, GATHER_DTYPE_BYTES));
        if (groupElements != 0 && requiredBytes > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t singleRequiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES));
        if (singleRequiredBytes > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }
    if (maxSourceBytes == 0 || maxSourceBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t batches = CeilDiv(plan.rowsPerCore, batchRows);
    // 每个连续 source 搬入都会对应一次直接的 MTE3 输出；最大合法 batch 优先减少两侧
    // 指令发射，同时没有依赖某个 shape 或固定输入分段。
    plan.score = SaturatingMul(SaturatingMul(nonzeroInputs, batches),
        2 * DMA_CALL_EQUIVALENT_BYTES);
    plan.batchRows = batchRows;
    plan.sourceBatchRows = batchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = 0;
    plan.offsetBufferBytes = 0;
    return true;
}

bool BuildDirectMteGatherSubPlan(const std::vector<InputInfo>& inputs, uint32_t rowsPerCore,
    uint64_t ubSize, GatherSubPlan& best)
{
    if (rowsPerCore == 0) {
        return false;
    }
    // 该路径的物理读/写 burst 数固定为 inputCount * rows；增大 batch 仅减少命令和
    // 事件边。因此从最大合法 batch 开始选择，UB 无法容纳时再逐级回退。
    for (uint32_t batchRows = rowsPerCore; batchRows != 0; --batchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = rowsPerCore;
        if (BuildDirectMteGroups(inputs, batchRows, ubSize, candidate)) {
            best = std::move(candidate);
            return true;
        }
    }
    return false;
}

bool BuildDirectMteRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildDirectMteGatherSubPlan(inputs, baseline.subPlans[0].rowsPerCore, ubSize,
        lowPlan) ||
        !BuildDirectMteGatherSubPlan(inputs, baseline.subPlans[1].rowsPerCore, ubSize,
        highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildFullStageOffsetGatherGroups(const std::vector<InputInfo>& inputs,
    uint32_t batchRows, uint32_t sourceBatchRows, uint64_t ubSize, GatherSubPlan& plan)
{
    if (batchRows == 0 || sourceBatchRows <= batchRows) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t sumAlignedElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;
    uint64_t maxOffsetBytes = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t offsetBytes = SaturatingMul(
            SaturatingMul(sourceBatchRows, alignedElements), GATHER_DTYPE_BYTES);
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        maxOffsetBytes = std::max(maxOffsetBytes, offsetBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputSourceElements = AlignUp(
            SaturatingMul(sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements,
            inputSourceElements);
        const uint64_t candidateAligned = AlignUp(candidateElements,
            GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateSourceBytes = SaturatingMul(candidateSourceElements,
            GATHER_DTYPE_BYTES);
        const uint64_t candidateOutputBytes = SaturatingMul(
            SaturatingMul(batchRows, candidateAligned), GATHER_DTYPE_BYTES);
        const uint64_t candidateOffsetBytes = SaturatingMul(
            SaturatingMul(sourceBatchRows, candidateAligned), GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(candidateSourceBytes,
                SaturatingAdd(candidateOutputBytes, candidateOffsetBytes)));
        if (groupElements != 0 && requiredBytes > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t offsetBytes = SaturatingMul(
            SaturatingMul(sourceBatchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t singleRequiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(sourceBytes, SaturatingAdd(outputBytes, offsetBytes)));
        if (singleRequiredBytes > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    const uint64_t sourceFullBatches = plan.rowsPerCore / sourceBatchRows;
    const uint64_t sourceTailRows = plan.rowsPerCore % sourceBatchRows;
    const uint64_t sourceBatches = sourceFullBatches +
        static_cast<uint64_t>(sourceTailRows != 0);
    const uint64_t outputFullBatches = plan.rowsPerCore / batchRows;
    const uint64_t outputTailRows = plan.rowsPerCore % batchRows;
    const uint64_t outputBatches = outputFullBatches +
        static_cast<uint64_t>(outputTailRows != 0);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), outputBatches);
    uint64_t gatherCalls = 0;
    for (uint32_t elements : plan.groupElements) {
        const uint64_t alignedElements = AlignUp(elements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t fullBatchCalls = CeilDiv(
            SaturatingMul(batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
        gatherCalls = SaturatingAdd(gatherCalls,
            SaturatingMul(outputFullBatches, fullBatchCalls));
        if (outputTailRows != 0) {
            gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
                SaturatingMul(outputTailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
        }
    }
    const uint64_t outputBlocks = SaturatingMul(plan.rowsPerCore, plan.groupInputEnd.size());
    // 完整 stage 表只生成一次，并被其中的所有输出子批按行切片复用。
    const uint64_t offsetWork = SaturatingMul(
        SaturatingMul(sumAlignedElements, sourceBatchRows + 1), 3 * sizeof(int32_t));
    plan.score = SaturatingAdd(
        SaturatingAdd(SaturatingMul(
            SaturatingAdd(inputCalls, SaturatingAdd(outputCalls, gatherCalls)),
            DMA_CALL_EQUIVALENT_BYTES),
            SaturatingMul(outputBlocks, DMA_BLOCK_EQUIVALENT_BYTES)), offsetWork);
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(maxSourceBytes, SaturatingAdd(maxOutputBytes, maxOffsetBytes)));
    if (maxSourceBytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() ||
        maxOffsetBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }
    plan.batchRows = batchRows;
    plan.sourceBatchRows = sourceBatchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = static_cast<uint32_t>(maxOffsetBytes);
    return true;
}

bool BuildGatherSubPlanWithBufferCounts(const std::vector<InputInfo>& inputs,
    uint32_t rowsPerCore, uint64_t ubSize, uint32_t sourceBufferCount,
    uint32_t matrixBufferCount, GatherSubPlan& best,
    uint32_t persistentOffsetRows = 0)
{
    best = GatherSubPlan{};
    best.rowsPerCore = rowsPerCore;
    for (uint32_t batchRows = 1;
         batchRows <= std::min<uint32_t>(rowsPerCore, DATA_COPY_MAX_ROWS); ++batchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = rowsPerCore;
        if (BuildGatherGroupsWithBufferCounts(inputs, batchRows, batchRows, ubSize,
            sourceBufferCount, matrixBufferCount, candidate, persistentOffsetRows) &&
            candidate.score < best.score) {
            best = std::move(candidate);
        }
    }
    return best.score != std::numeric_limits<uint64_t>::max();
}

bool BuildGatherSubPlan(const std::vector<InputInfo>& inputs, uint32_t rowsPerCore,
    uint64_t ubSize, GatherSubPlan& best)
{
    return BuildGatherSubPlanWithBufferCounts(inputs, rowsPerCore, ubSize, 1, 2, best);
}

struct GatherCommandStats {
    uint64_t inputCalls{0};
    uint64_t outputCalls{0};
    uint64_t gatherCalls{0};
};

GatherCommandStats GetGatherCommandStats(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& plan)
{
    GatherCommandStats stats;
    if (plan.rowsPerCore == 0 || plan.batchRows == 0) {
        return stats;
    }
    const uint32_t sourceBatchRows = plan.sourceBatchRows == 0 ? plan.batchRows :
        plan.sourceBatchRows;
    const uint64_t sourceFullBatches = plan.rowsPerCore / sourceBatchRows;
    const uint64_t sourceTailRows = plan.rowsPerCore % sourceBatchRows;
    const uint64_t sourceBatches = sourceFullBatches +
        static_cast<uint64_t>(sourceTailRows != 0);
    const uint64_t outputFullBatches = plan.rowsPerCore / plan.batchRows;
    const uint64_t outputTailRows = plan.rowsPerCore % plan.batchRows;
    const uint64_t outputBatches = outputFullBatches +
        static_cast<uint64_t>(outputTailRows != 0);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    stats.inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
    stats.outputCalls = SaturatingMul(plan.groupInputEnd.size(), outputBatches);
    for (uint32_t elements : plan.groupElements) {
        const uint64_t alignedElements = AlignUp(elements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t fullBatchCalls = CeilDiv(
            SaturatingMul(plan.batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
        stats.gatherCalls = SaturatingAdd(stats.gatherCalls,
            SaturatingMul(outputFullBatches, fullBatchCalls));
        if (outputTailRows != 0) {
            stats.gatherCalls = SaturatingAdd(stats.gatherCalls, CeilDiv(
                SaturatingMul(outputTailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
        }
    }
    return stats;
}

// 只有更大的连续 source 批确实减少输入 DMA，且不以更多 output/Gather 指令为代价时，
// 才允许用每批重建 VECOUT offset 的方式换取 UB 空间。所有条件都来自动态计划。
bool IsOffsetOutputAliasRowGatherPlanUseful(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, const RowGatherPlan& candidate)
{
    for (size_t planIndex = 0; planIndex < baseline.subPlans.size(); ++planIndex) {
        const GatherSubPlan& base = baseline.subPlans[planIndex];
        const GatherSubPlan& alias = candidate.subPlans[planIndex];
        const GatherCommandStats baseStats = GetGatherCommandStats(inputs, base);
        const GatherCommandStats aliasStats = GetGatherCommandStats(inputs, alias);
        if (base.rowsPerCore == 0 || alias.rowsPerCore != base.rowsPerCore ||
            alias.batchRows < OFFSET_OUTPUT_ALIAS_SEED_ROWS ||
            alias.sourceBatchRows != alias.batchRows || alias.batchRows <= base.batchRows ||
            alias.rowsPerCore / alias.batchRows < 2 ||
            alias.offsetBufferBytes >= alias.outputBufferBytes ||
            aliasStats.inputCalls >= baseStats.inputCalls ||
            aliasStats.outputCalls > baseStats.outputCalls ||
            aliasStats.gatherCalls > baseStats.gatherCalls) {
            return false;
        }
    }
    return true;
}

// 双 source 槽的目标是让下一批连续 MTE2 与当前批的 Vector/MTE3 重叠，而不是用缩小
// batch 换取更少 DMA。因此只接受不小于基线 batch、且不增加动态命令数的别名计划。
bool IsSourcePingPongOffsetOutputAliasRowGatherPlanUseful(
    const std::vector<InputInfo>& inputs, const RowGatherPlan& baseline,
    const RowGatherPlan& candidate)
{
    for (size_t planIndex = 0; planIndex < baseline.subPlans.size(); ++planIndex) {
        const GatherSubPlan& base = baseline.subPlans[planIndex];
        const GatherSubPlan& combined = candidate.subPlans[planIndex];
        const GatherCommandStats baseStats = GetGatherCommandStats(inputs, base);
        const GatherCommandStats combinedStats = GetGatherCommandStats(inputs, combined);
        if (base.rowsPerCore == 0 || combined.rowsPerCore != base.rowsPerCore ||
            combined.batchRows < base.batchRows ||
            combined.batchRows < OFFSET_OUTPUT_ALIAS_SEED_ROWS ||
            combined.sourceBatchRows != combined.batchRows ||
            combined.rowsPerCore / combined.batchRows < 2 ||
            combined.offsetBufferBytes >= combined.outputBufferBytes ||
            combinedStats.inputCalls > baseStats.inputCalls ||
            combinedStats.outputCalls > baseStats.outputCalls ||
            combinedStats.gatherCalls > baseStats.gatherCalls) {
            return false;
        }
    }
    return true;
}

// 计算一个连续输入 group 在给定 batch 行数下的实际 source 槽和矩阵槽大小。source 仍按
// 输入 Tensor 连续搬入；该函数只改变 UB 内一组的行数，不改变输入/输出的地址所有权。
bool GetGatherGroupBufferLayout(const std::vector<InputInfo>& inputs, uint32_t inputStart,
    uint32_t inputEnd, uint32_t groupElements, uint32_t batchRows,
    uint32_t& sourceElements, uint64_t& sourceBytes, uint64_t& matrixBytes)
{
    sourceElements = 0;
    sourceBytes = 0;
    matrixBytes = 0;
    if (inputStart >= inputEnd || inputEnd > inputs.size() || groupElements == 0 ||
        batchRows == 0 || batchRows > DATA_COPY_MAX_ROWS) {
        return false;
    }
    uint64_t observedElements = 0;
    uint64_t sourceElements64 = 0;
    for (uint32_t input = inputStart; input < inputEnd; ++input) {
        if (inputs[input].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t payloadBytes = SaturatingMul(batchRows, inputs[input].rowBytes);
        // CopyInputs 的每个输入都以一条连续 DMA 发射，单条 blockLen 不能越过硬件上限。
        if (payloadBytes > DATA_COPY_MAX_BLOCK_LEN ||
            payloadBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        observedElements = SaturatingAdd(observedElements, inputElements);
        sourceElements64 = SaturatingAdd(sourceElements64, AlignUp(
            SaturatingMul(batchRows, inputElements), GATHER_ELEMENT_ALIGNMENT));
    }
    const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
    sourceBytes = SaturatingMul(sourceElements64, GATHER_DTYPE_BYTES);
    matrixBytes = SaturatingMul(SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
    if (observedElements != groupElements || sourceElements64 == 0 ||
        sourceElements64 > std::numeric_limits<uint32_t>::max() ||
        sourceBytes > std::numeric_limits<uint32_t>::max() ||
        matrixBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    sourceElements = static_cast<uint32_t>(sourceElements64);
    return true;
}

// group 采用不同 batch 行数后，按实际 group 重新统计命令数。只接受 source 调用严格减少、
// 输出和 Gather 调用不增加的计划，避免把全局扩批的已知回退重新引入调度。
bool GetAdaptiveGroupGatherCommandStats(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& plan, GatherCommandStats& stats)
{
    stats = GatherCommandStats{};
    if (plan.rowsPerCore == 0 || plan.batchRows == 0 ||
        plan.groupInputEnd.empty() ||
        plan.groupInputEnd.size() != plan.groupElements.size() ||
        plan.groupInputEnd.size() != plan.groupSourceElements.size() ||
        plan.groupInputEnd.size() != plan.groupBatchRows.size()) {
        return false;
    }
    uint32_t inputStart = 0;
    for (size_t group = 0; group < plan.groupInputEnd.size(); ++group) {
        const uint32_t inputEnd = plan.groupInputEnd[group];
        const uint32_t groupElements = plan.groupElements[group];
        const uint32_t batchRows = plan.groupBatchRows[group];
        if (inputEnd <= inputStart || inputEnd > inputs.size() || groupElements == 0 ||
            batchRows < plan.batchRows || batchRows > plan.rowsPerCore) {
            return false;
        }
        const uint64_t sourceBatches = CeilDiv(plan.rowsPerCore, batchRows);
        const uint64_t outputBatches = CeilDiv(plan.rowsPerCore, batchRows);
        uint64_t nonzeroInputs = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            nonzeroInputs += static_cast<uint64_t>(inputs[input].rowBytes != 0);
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t fullBatches = plan.rowsPerCore / batchRows;
        const uint64_t tailRows = plan.rowsPerCore % batchRows;
        const uint64_t fullBatchCalls = CeilDiv(
            SaturatingMul(batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
        if (sourceBatches == 0 || outputBatches == 0 || nonzeroInputs == 0 ||
            alignedElements == 0 || fullBatchCalls == 0) {
            return false;
        }
        stats.inputCalls = SaturatingAdd(stats.inputCalls,
            SaturatingMul(nonzeroInputs, sourceBatches));
        stats.outputCalls = SaturatingAdd(stats.outputCalls, outputBatches);
        stats.gatherCalls = SaturatingAdd(stats.gatherCalls,
            SaturatingMul(fullBatches, fullBatchCalls));
        if (tailRows != 0) {
            stats.gatherCalls = SaturatingAdd(stats.gatherCalls, CeilDiv(
                SaturatingMul(tailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
        }
        inputStart = inputEnd;
    }
    return inputStart == inputs.size() && stats.inputCalls != 0 && stats.outputCalls != 0 &&
        stats.gatherCalls != 0;
}

bool BuildAdaptiveGroupBatchGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, GatherSubPlan& plan, bool& improved)
{
    improved = false;
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.sourceBatchRows != baseline.batchRows || baseline.sourceBufferBytes == 0 ||
        baseline.outputBufferBytes == 0 || baseline.offsetBufferBytes == 0 ||
        baseline.groupInputEnd.empty() ||
        baseline.groupInputEnd.size() != baseline.groupElements.size() ||
        baseline.groupInputEnd.size() != baseline.groupSourceElements.size()) {
        return false;
    }
    GatherSubPlan uniform = baseline;
    uniform.groupBatchRows.assign(uniform.groupInputEnd.size(), uniform.batchRows);
    GatherCommandStats baselineStats;
    if (!GetAdaptiveGroupGatherCommandStats(inputs, uniform, baselineStats)) {
        return false;
    }

    plan = uniform;
    uint32_t inputStart = 0;
    for (size_t group = 0; group < plan.groupInputEnd.size(); ++group) {
        const uint32_t inputEnd = plan.groupInputEnd[group];
        const uint32_t groupElements = plan.groupElements[group];
        uint32_t baselineSourceElements = 0;
        uint64_t baselineSourceBytes = 0;
        uint64_t baselineMatrixBytes = 0;
        if (!GetGatherGroupBufferLayout(inputs, inputStart, inputEnd, groupElements,
            plan.batchRows, baselineSourceElements, baselineSourceBytes, baselineMatrixBytes) ||
            baselineSourceElements != plan.groupSourceElements[group] ||
            baselineSourceBytes > plan.sourceBufferBytes ||
            baselineMatrixBytes > plan.outputBufferBytes ||
            baselineMatrixBytes > plan.offsetBufferBytes) {
            return false;
        }

        uint32_t selectedRows = plan.batchRows;
        uint32_t selectedSourceElements = baselineSourceElements;
        uint32_t low = plan.batchRows + 1;
        uint32_t high = std::min<uint32_t>(plan.rowsPerCore, DATA_COPY_MAX_ROWS);
        while (low <= high) {
            const uint32_t middle = low + (high - low) / 2;
            uint32_t sourceElements = 0;
            uint64_t sourceBytes = 0;
            uint64_t matrixBytes = 0;
            const bool fits = GetGatherGroupBufferLayout(inputs, inputStart, inputEnd,
                groupElements, middle, sourceElements, sourceBytes, matrixBytes) &&
                sourceBytes <= plan.sourceBufferBytes &&
                matrixBytes <= plan.outputBufferBytes && matrixBytes <= plan.offsetBufferBytes;
            if (fits) {
                selectedRows = middle;
                selectedSourceElements = sourceElements;
                low = middle + 1;
            } else {
                high = middle - 1;
            }
        }
        plan.groupBatchRows[group] = selectedRows;
        plan.groupSourceElements[group] = selectedSourceElements;
        improved = improved || selectedRows > plan.batchRows;
        inputStart = inputEnd;
    }
    if (inputStart != inputs.size()) {
        return false;
    }

    GatherCommandStats candidateStats;
    if (!GetAdaptiveGroupGatherCommandStats(inputs, plan, candidateStats)) {
        return false;
    }
    if (improved && (candidateStats.inputCalls >= baselineStats.inputCalls ||
        candidateStats.outputCalls > baselineStats.outputCalls ||
        candidateStats.gatherCalls > baselineStats.gatherCalls)) {
        return false;
    }
    return true;
}

bool BuildAdaptiveGroupBatchRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, RowGatherPlan& plan)
{
    bool lowImproved = false;
    bool highImproved = false;
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildAdaptiveGroupBatchGatherSubPlan(inputs, baseline.subPlans[0], lowPlan,
        lowImproved) || !BuildAdaptiveGroupBatchGatherSubPlan(inputs, baseline.subPlans[1],
        highPlan, highImproved) || (!lowImproved && !highImproved)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

uint64_t EstimateOffsetTableWork(const GatherSubPlan& plan, uint32_t tableRows)
{
    if (tableRows == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t sumAlignedElements = 0;
    for (uint32_t elements : plan.groupElements) {
        sumAlignedElements = SaturatingAdd(sumAlignedElements,
            AlignUp(elements, GATHER_ELEMENT_ALIGNMENT));
    }
    return SaturatingMul(SaturatingMul(sumAlignedElements, tableRows + 1),
        3 * sizeof(int32_t));
}

uint64_t CountNonzeroGatherInputs(const std::vector<InputInfo>& inputs)
{
    uint64_t count = 0;
    for (const InputInfo& input : inputs) {
        count += static_cast<uint64_t>(input.rowBytes != 0);
    }
    return count;
}

// 一个双批 stage 由 blockCount=2 的 DMA 覆盖；尾部不能组成等长双平面时保留单平面，
// 以免让 DMA 读取越过输入 Tensor 的合法行范围。
uint64_t GetTwoPlaneSourceStages(uint32_t rowsPerCore, uint32_t batchRows)
{
    if (rowsPerCore == 0 || batchRows == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t planePairRows = SaturatingMul(2, batchRows);
    if (planePairRows == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t fullPairs = rowsPerCore / planePairRows;
    const uint64_t tailRows = rowsPerCore % planePairRows;
    if (tailRows == 0) {
        return fullPairs;
    }
    // tail 小于等于一个平面时只发射一次普通连续搬入；否则依次处理两个单平面。
    return SaturatingAdd(fullPairs, tailRows <= batchRows ? 1 : 2);
}

bool IsTwoPlaneInputPayloadLegal(const std::vector<InputInfo>& inputs, uint32_t batchRows)
{
    if (batchRows == 0) {
        return false;
    }
    for (const InputInfo& input : inputs) {
        if (input.rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t payloadBytes = SaturatingMul(batchRows, input.rowBytes);
        // DataCopyExtParams 的 blockLen 仍受硬件单 block 上限限制；blockCount=2 不会
        // 放宽单个 plane 的长度。
        if (payloadBytes > DATA_COPY_MAX_BLOCK_LEN ||
            payloadBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    return true;
}

GatherCommandStats GetTwoPlaneGatherCommandStats(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& plan)
{
    GatherCommandStats stats = GetGatherCommandStats(inputs, plan);
    const uint64_t sourceStages = GetTwoPlaneSourceStages(plan.rowsPerCore, plan.batchRows);
    if (sourceStages == std::numeric_limits<uint64_t>::max()) {
        return GatherCommandStats{};
    }
    stats.inputCalls = SaturatingMul(CountNonzeroGatherInputs(inputs), sourceStages);
    return stats;
}

uint64_t EstimateSourceCriticalGatherScore(const GatherSubPlan& plan,
    const GatherCommandStats& stats)
{
    const uint64_t weightedInputCalls = SaturatingMul(stats.inputCalls,
        TWO_PLANE_SOURCE_CALL_WEIGHT);
    const uint64_t commandCalls = SaturatingAdd(weightedInputCalls,
        SaturatingAdd(stats.outputCalls, stats.gatherCalls));
    const uint64_t outputBlocks = SaturatingMul(plan.rowsPerCore, plan.groupInputEnd.size());
    return SaturatingAdd(SaturatingAdd(SaturatingMul(commandCalls,
        DMA_CALL_EQUIVALENT_BYTES), SaturatingMul(outputBlocks, DMA_BLOCK_EQUIVALENT_BYTES)),
        EstimateOffsetTableWork(plan, plan.batchRows));
}

bool BuildTwoPlaneGatherSubPlanForBatch(const std::vector<InputInfo>& inputs,
    uint32_t rowsPerCore, uint32_t batchRows, uint64_t ubSize, GatherSubPlan& plan)
{
    if (rowsPerCore == 0 || batchRows == 0 || batchRows > rowsPerCore ||
        !IsTwoPlaneInputPayloadLegal(inputs, batchRows)) {
        return false;
    }
    plan = GatherSubPlan{};
    plan.rowsPerCore = rowsPerCore;
    // 两张 source 平面 + 一张 output 和一张 offset 表；每张 source 平面保持原始输入
    // 的连续 GM→UB 搬入，避免退化为逐行短 burst。
    if (!BuildGatherGroupsWithBufferCounts(inputs, batchRows, batchRows, ubSize, 2, 2, plan) ||
        plan.sourceBufferBytes == 0 ||
        plan.sourceBufferBytes > std::numeric_limits<uint32_t>::max() / 2) {
        return false;
    }
    const GatherCommandStats stats = GetTwoPlaneGatherCommandStats(inputs, plan);
    if (stats.inputCalls == 0 || stats.outputCalls == 0 || stats.gatherCalls == 0) {
        return false;
    }
    plan.score = EstimateSourceCriticalGatherScore(plan, stats);
    return plan.score != std::numeric_limits<uint64_t>::max();
}

uint64_t GetExtraControlCalls(const GatherCommandStats& baseline,
    const GatherCommandStats& candidate)
{
    const uint64_t extraOutput = candidate.outputCalls > baseline.outputCalls ?
        candidate.outputCalls - baseline.outputCalls : 0;
    const uint64_t extraGather = candidate.gatherCalls > baseline.gatherCalls ?
        candidate.gatherCalls - baseline.gatherCalls : 0;
    return SaturatingAdd(extraOutput, extraGather);
}

bool HasTwoPlaneStructuralGain(const GatherCommandStats& baseline,
    const GatherCommandStats& candidate)
{
    if (candidate.inputCalls >= baseline.inputCalls) {
        return false;
    }
    const uint64_t inputSaving = baseline.inputCalls - candidate.inputCalls;
    // 只有输入 descriptor 节省严格覆盖新增的 dense output/Gather 控制命令，才允许用
    // 双 source 平面交换 UB 空间。这个判断只依赖运行时 plan，不引用 shape 或 case。
    return inputSaving > GetExtraControlCalls(baseline, candidate);
}

bool BuildTwoPlaneRowGatherPlan(const std::vector<InputInfo>& inputs, uint32_t blockDim,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    if (blockDim == 0 || baseline.highCoreCount > blockDim ||
        baseline.subPlans[0].rowsPerCore == 0 || baseline.subPlans[1].rowsPerCore == 0) {
        return false;
    }
    const bool lowPlanActive = blockDim > baseline.highCoreCount;
    const bool highPlanActive = baseline.highCoreCount != 0;
    if (!lowPlanActive && !highPlanActive) {
        return false;
    }
    const uint32_t maxBatchRows = std::min(baseline.subPlans[0].rowsPerCore,
        baseline.subPlans[1].rowsPerCore);
    if (maxBatchRows < 2) {
        return false;
    }

    const GatherCommandStats baselineLowStats = GetGatherCommandStats(inputs,
        baseline.subPlans[0]);
    const GatherCommandStats baselineHighStats = GetGatherCommandStats(inputs,
        baseline.subPlans[1]);
    const uint64_t baselineLowScore = EstimateSourceCriticalGatherScore(baseline.subPlans[0],
        baselineLowStats);
    const uint64_t baselineHighScore = EstimateSourceCriticalGatherScore(baseline.subPlans[1],
        baselineHighStats);
    uint64_t bestMaxScore = std::max(lowPlanActive ? baselineLowScore : 0,
        highPlanActive ? baselineHighScore : 0);
    uint64_t bestTotalScore = SaturatingAdd(
        lowPlanActive ? SaturatingMul(blockDim - baseline.highCoreCount, baselineLowScore) : 0,
        highPlanActive ? SaturatingMul(baseline.highCoreCount, baselineHighScore) : 0);
    bool found = false;

    for (uint32_t batchRows = 1; batchRows <= maxBatchRows; ++batchRows) {
        GatherSubPlan candidateLow;
        GatherSubPlan candidateHigh;
        if (!BuildTwoPlaneGatherSubPlanForBatch(inputs, baseline.subPlans[0].rowsPerCore,
                batchRows, ubSize, candidateLow) ||
            !BuildTwoPlaneGatherSubPlanForBatch(inputs, baseline.subPlans[1].rowsPerCore,
                batchRows, ubSize, candidateHigh)) {
            continue;
        }
        const GatherCommandStats candidateLowStats = GetTwoPlaneGatherCommandStats(inputs,
            candidateLow);
        const GatherCommandStats candidateHighStats = GetTwoPlaneGatherCommandStats(inputs,
            candidateHigh);
        if ((lowPlanActive && !HasTwoPlaneStructuralGain(baselineLowStats, candidateLowStats)) ||
            (highPlanActive && !HasTwoPlaneStructuralGain(baselineHighStats, candidateHighStats))) {
            continue;
        }
        const uint64_t candidateMaxScore = std::max(lowPlanActive ? candidateLow.score : 0,
            highPlanActive ? candidateHigh.score : 0);
        const uint64_t candidateTotalScore = SaturatingAdd(
            lowPlanActive ? SaturatingMul(blockDim - baseline.highCoreCount,
                candidateLow.score) : 0,
            highPlanActive ? SaturatingMul(baseline.highCoreCount, candidateHigh.score) : 0);
        const bool improvesMax = candidateMaxScore < bestMaxScore;
        const bool improvesTie = candidateMaxScore == bestMaxScore &&
            candidateTotalScore < bestTotalScore;
        if (improvesMax || improvesTie) {
            plan.highCoreCount = baseline.highCoreCount;
            plan.subPlans[0] = std::move(candidateLow);
            plan.subPlans[1] = std::move(candidateHigh);
            bestMaxScore = candidateMaxScore;
            bestTotalScore = candidateTotalScore;
            found = true;
        }
    }
    return found;
}

bool BuildFullStageOffsetGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    const uint64_t baselineControlCalls = SaturatingAdd(
        baselineStats.outputCalls, baselineStats.gatherCalls);
    const uint64_t baselineOffsetWork = EstimateOffsetTableWork(baseline,
        baseline.batchRows);
    const uint32_t sourceBatchLimit = static_cast<uint32_t>(std::min<uint64_t>(
        baseline.rowsPerCore, SaturatingMul(baseline.batchRows, 2)));
    bool found = false;
    uint64_t bestNetGain = 0;
    for (uint32_t sourceBatchRows = baseline.batchRows + 1;
         sourceBatchRows <= sourceBatchLimit; ++sourceBatchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        if (!BuildFullStageOffsetGatherGroups(inputs, baseline.batchRows, sourceBatchRows,
            ubSize, candidate)) {
            continue;
        }
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        const uint64_t inputCallSaving = baselineStats.inputCalls > candidateStats.inputCalls ?
            baselineStats.inputCalls - candidateStats.inputCalls : 0;
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t controlCallGrowth = candidateControlCalls > baselineControlCalls ?
            candidateControlCalls - baselineControlCalls : 0;
        const uint64_t candidateOffsetWork = EstimateOffsetTableWork(candidate,
            sourceBatchRows);
        const uint64_t offsetWorkGrowth = candidateOffsetWork > baselineOffsetWork ?
            candidateOffsetWork - baselineOffsetWork : 0;
        const uint64_t inputBenefit = SaturatingMul(inputCallSaving,
            DMA_CALL_EQUIVALENT_BYTES);
        const uint64_t controlCost = SaturatingMul(controlCallGrowth,
            DMA_CALL_EQUIVALENT_BYTES);
        const uint64_t candidateCost = SaturatingAdd(controlCost, offsetWorkGrowth);
        // 完整表会占用与 source stage 同行数的 VECCALC。只在少掉一整轮连续 MTE2
        // 发射足以支付新增 Gather/MTE3 控制和建表工作时，才交换这块 UB。
        if (inputBenefit <= candidateCost ||
            candidate.groupInputEnd.size() > baseline.groupInputEnd.size() + 2) {
            continue;
        }
        const uint64_t netGain = inputBenefit - candidateCost;
        if (!found || netGain > bestNetGain ||
            (netGain == bestNetGain &&
             candidate.groupInputEnd.size() < best.groupInputEnd.size()) ||
            (netGain == bestNetGain &&
             candidate.groupInputEnd.size() == best.groupInputEnd.size() &&
             candidate.sourceBatchRows < best.sourceBatchRows)) {
            best = std::move(candidate);
            bestNetGain = netGain;
            found = true;
        }
    }
    return found;
}

bool BuildMergedBatchGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, uint32_t sourceBufferCount,
    uint32_t matrixBufferCount, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    bool found = false;
    GatherCommandStats bestStats;
    for (uint32_t batchRows = baseline.batchRows + 1;
         batchRows <= baseline.rowsPerCore; ++batchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        if (!BuildGatherGroupsWithBufferCounts(inputs, batchRows, batchRows, ubSize,
            sourceBufferCount, matrixBufferCount, candidate)) {
            continue;
        }
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        // 只接受同时减少 source MTE2 发射且不增加 output/Gather 指令数的计划，
        // 避免以更多窄 MTE3 或 Vector 命令交换输入侧收益。
        if (candidateStats.inputCalls >= baselineStats.inputCalls ||
            candidateStats.outputCalls > baselineStats.outputCalls ||
            candidateStats.gatherCalls > baselineStats.gatherCalls) {
            continue;
        }
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t bestControlCalls = SaturatingAdd(
            bestStats.outputCalls, bestStats.gatherCalls);
        if (!found || candidateStats.inputCalls < bestStats.inputCalls ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls < bestControlCalls) ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls == bestControlCalls &&
             candidate.batchRows < best.batchRows)) {
            best = std::move(candidate);
            bestStats = candidateStats;
            found = true;
        }
    }
    return found;
}

bool BuildStagedSourceGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    const uint64_t outputBatches = CeilDiv(baseline.rowsPerCore, baseline.batchRows);
    const uint32_t sourceBatchLimit = static_cast<uint32_t>(std::min<uint64_t>(
        baseline.rowsPerCore, SaturatingMul(baseline.batchRows, 2)));
    bool found = false;
    GatherCommandStats bestStats;
    for (uint32_t sourceBatchRows = baseline.batchRows + 1;
         sourceBatchRows <= sourceBatchLimit; ++sourceBatchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        if (!BuildGatherGroupsWithBufferCounts(inputs, baseline.batchRows, sourceBatchRows,
            ubSize, 1, 2, candidate)) {
            continue;
        }
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t baselineControlCalls = SaturatingAdd(
            baselineStats.outputCalls, baselineStats.gatherCalls);
        // 只容许多出一个 group，限制额外 MTE3/Gather 发射；若 source stage 没有实际
        // 少一次 MTE2，则不以复杂度交换任何收益。
        if (candidateStats.inputCalls >= baselineStats.inputCalls ||
            candidate.groupInputEnd.size() > baseline.groupInputEnd.size() + 1 ||
            candidateControlCalls > SaturatingAdd(baselineControlCalls,
                SaturatingMul(2, outputBatches))) {
            continue;
        }
        const uint64_t bestControlCalls = SaturatingAdd(
            bestStats.outputCalls, bestStats.gatherCalls);
        if (!found || candidateStats.inputCalls < bestStats.inputCalls ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls < bestControlCalls) ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls == bestControlCalls &&
             candidate.sourceBatchRows < best.sourceBatchRows)) {
            best = std::move(candidate);
            bestStats = candidateStats;
            found = true;
        }
    }
    return found;
}

// 连续 source stage 的增量 offset 路径：除常规 source/output/offset 三块矩阵外，
// 只额外保存一行“按输入段行宽展开”的 delta。它可一次平移所有有效 offset 行，避免
// key27 对每一行、每一段分别发射 Adds；delta 也会在同一块 offset buffer 中恢复。
bool BuildDeltaStagedOffsetGatherGroups(const std::vector<InputInfo>& inputs,
    uint32_t batchRows, uint32_t sourceBatchRows, uint64_t ubSize, GatherSubPlan& plan)
{
    // C220 二元向量指令的 repeatTime 与 repeat stride 均为 uint8_t；该路径
    // 不能把非法计划交给 Kernel 再防御回退，否则会留下未写出的输出。
    if (batchRows == 0 || batchRows > 255 || sourceBatchRows <= batchRows) {
        return false;
    }
    const uint32_t rowsPerCore = plan.rowsPerCore;
    plan = GatherSubPlan{};
    plan.rowsPerCore = rowsPerCore;
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t sumAlignedElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;
    uint64_t maxDeltaBytes = 0;

    auto requiredBytes = [&](uint64_t sourceElements, uint64_t outputElements) -> uint64_t {
        const uint64_t sourceBytes = SaturatingMul(sourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, outputElements), GATHER_DTYPE_BYTES);
        const uint64_t deltaBytes = SaturatingMul(outputElements, GATHER_DTYPE_BYTES);
        return SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(sourceBytes,
            SaturatingAdd(SaturatingMul(2, outputBytes), deltaBytes)));
    };
    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        if (alignedElements / GATHER_ELEMENT_ALIGNMENT > 255) {
            return false;
        }
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t deltaBytes = SaturatingMul(alignedElements, GATHER_DTYPE_BYTES);
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        maxDeltaBytes = std::max(maxDeltaBytes, deltaBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputSourceElements = AlignUp(
            SaturatingMul(sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements,
            inputSourceElements);
        const uint64_t candidateAlignedElements = AlignUp(candidateElements,
            GATHER_ELEMENT_ALIGNMENT);
        if (groupElements != 0 && requiredBytes(candidateSourceElements,
            candidateAlignedElements) > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        if (alignedElements / GATHER_ELEMENT_ALIGNMENT > 255) {
            return false;
        }
        if (requiredBytes(groupSourceElements, alignedElements) > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    const uint64_t sourceBatches = CeilDiv(plan.rowsPerCore, sourceBatchRows);
    const uint64_t outputFullBatches = plan.rowsPerCore / batchRows;
    const uint64_t outputTailRows = plan.rowsPerCore % batchRows;
    const uint64_t outputBatches = outputFullBatches +
        static_cast<uint64_t>(outputTailRows != 0);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), outputBatches);
    uint64_t gatherCalls = 0;
    for (uint32_t elements : plan.groupElements) {
        const uint64_t alignedElements = AlignUp(elements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t fullBatchCalls = CeilDiv(
            SaturatingMul(batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
        gatherCalls = SaturatingAdd(gatherCalls,
            SaturatingMul(outputFullBatches, fullBatchCalls));
        if (outputTailRows != 0) {
            gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
                SaturatingMul(outputTailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
        }
    }
    const uint64_t outputBlocks = SaturatingMul(plan.rowsPerCore, plan.groupInputEnd.size());
    // 基准 offset 只建一次；每个 source stage 最多有一个后续子批，需要一行 delta
    // 和两次整行 Add/Sub 恢复。该代价只用于同类计划间排序，最终仍以 PROF 判定。
    const uint64_t offsetWork = SaturatingMul(sumAlignedElements,
        SaturatingMul(SaturatingAdd(SaturatingMul(3, batchRows + 1),
            SaturatingMul(3, sourceBatches)), sizeof(int32_t)));
    plan.score = SaturatingAdd(SaturatingAdd(SaturatingMul(
        SaturatingAdd(inputCalls, SaturatingAdd(outputCalls, gatherCalls)),
        DMA_CALL_EQUIVALENT_BYTES), SaturatingMul(outputBlocks,
            DMA_BLOCK_EQUIVALENT_BYTES)), offsetWork);
    // delta 放在 source UB 尾部，与 offset 表物理分离；总 UB 用量仍为
    // source + delta + output + offset，避免广播源与目标紧邻造成地址歧义。
    const uint64_t sourceBytesWithDelta = SaturatingAdd(maxSourceBytes, maxDeltaBytes);
    const uint64_t offsetBytes = maxOutputBytes;
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(sourceBytesWithDelta, SaturatingAdd(maxOutputBytes, offsetBytes)));
    if (sourceBytesWithDelta > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() ||
        maxDeltaBytes > std::numeric_limits<uint32_t>::max() ||
        offsetBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }
    plan.batchRows = batchRows;
    plan.sourceBatchRows = sourceBatchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(sourceBytesWithDelta);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = static_cast<uint32_t>(offsetBytes);
    return true;
}

bool BuildDeltaStagedOffsetGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    const uint64_t baselineControlCalls = SaturatingAdd(
        baselineStats.outputCalls, baselineStats.gatherCalls);
    const uint32_t sourceBatchLimit = static_cast<uint32_t>(std::min<uint64_t>(
        baseline.rowsPerCore, SaturatingMul(baseline.batchRows, 2)));
    bool found = false;
    GatherCommandStats bestStats;
    for (uint32_t sourceBatchRows = baseline.batchRows + 1;
         sourceBatchRows <= sourceBatchLimit; ++sourceBatchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        if (!BuildDeltaStagedOffsetGatherGroups(inputs, baseline.batchRows, sourceBatchRows,
            ubSize, candidate)) {
            continue;
        }
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        const uint64_t inputCallSaving = baselineStats.inputCalls > candidateStats.inputCalls ?
            baselineStats.inputCalls - candidateStats.inputCalls : 0;
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t controlCallGrowth = candidateControlCalls > baselineControlCalls ?
            candidateControlCalls - baselineControlCalls : 0;
        // 额外 delta 表不会减少 MTE3/Gather 控制命令；只有连续 MTE2 的实际减少仍
        // 大于新增控制命令时，才以更大的 source stage 交换这部分 UB。
        if (inputCallSaving == 0 || inputCallSaving <= controlCallGrowth) {
            continue;
        }
        const uint64_t bestControlCalls = SaturatingAdd(
            bestStats.outputCalls, bestStats.gatherCalls);
        if (!found || candidateStats.inputCalls < bestStats.inputCalls ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls < bestControlCalls) ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls == bestControlCalls &&
             candidate.sourceBatchRows < best.sourceBatchRows)) {
            best = std::move(candidate);
            bestStats = candidateStats;
            found = true;
        }
    }
    return found;
}

bool BuildCachedOffsetStagedGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    // Kernel 只需缓存 source stage 的首、次子批 offset 表，因此 stage 最多容纳两个输出批。
    const uint32_t sourceBatchLimit = static_cast<uint32_t>(std::min<uint64_t>(
        baseline.rowsPerCore, SaturatingMul(baseline.batchRows, 2)));
    bool found = false;
    GatherCommandStats bestStats;
    for (uint32_t sourceBatchRows = baseline.batchRows + 1;
         sourceBatchRows <= sourceBatchLimit; ++sourceBatchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        // source、output 和两张同大小 offset 表；额外一张表必须完整计入 UB 预算。
        if (!BuildGatherGroupsWithBufferCounts(inputs, baseline.batchRows, sourceBatchRows,
            ubSize, 1, 3, candidate) ||
            candidate.outputBufferBytes > std::numeric_limits<uint32_t>::max() / 2) {
            continue;
        }
        candidate.offsetBufferBytes = candidate.outputBufferBytes * 2;
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t baselineControlCalls = SaturatingAdd(
            baselineStats.outputCalls, baselineStats.gatherCalls);
        const uint64_t inputCallSaving = baselineStats.inputCalls > candidateStats.inputCalls ?
            baselineStats.inputCalls - candidateStats.inputCalls : 0;
        const uint64_t controlCallGrowth = candidateControlCalls > baselineControlCalls ?
            candidateControlCalls - baselineControlCalls : 0;
        // 多一张 offset 表可能使 group 多于一组；只要减少的连续 MTE2 发射仍大于新增
        // output/Gather 控制量，就按硬件工作量而非固定 group 数决定是否尝试。
        if (inputCallSaving == 0 || inputCallSaving <= controlCallGrowth) {
            continue;
        }
        const uint64_t bestControlCalls = SaturatingAdd(
            bestStats.outputCalls, bestStats.gatherCalls);
        if (!found || candidateStats.inputCalls < bestStats.inputCalls ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls < bestControlCalls) ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls == bestControlCalls &&
             candidate.sourceBatchRows < best.sourceBatchRows)) {
            best = std::move(candidate);
            bestStats = candidateStats;
            found = true;
        }
    }
    return found;
}

bool BuildRowMajorStagedGatherGroups(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint32_t sourceBatchRows, uint64_t ubSize, GatherSubPlan& plan)
{
    if (batchRows == 0 || sourceBatchRows < batchRows) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t sourceRowElements = 0;
    uint64_t sumAlignedElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        const uint64_t sourceElements = SaturatingMul(sourceBatchRows, sourceRowElements);
        if (groupElements == 0 || sourceRowElements == 0 ||
            inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            sourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(sourceElements, GATHER_DTYPE_BYTES);
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(sourceElements));
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputRowElements = inputElements == 0 ? 0 :
            AlignUp(inputElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceRowElements = SaturatingAdd(sourceRowElements,
            inputRowElements);
        const uint64_t candidateAligned = AlignUp(candidateElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateSourceBytes = SaturatingMul(
            SaturatingMul(sourceBatchRows, candidateSourceRowElements), GATHER_DTYPE_BYTES);
        const uint64_t candidateMatrixBytes = SaturatingMul(
            SaturatingMul(batchRows, candidateAligned), GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(candidateSourceBytes, SaturatingMul(2, candidateMatrixBytes)));
        if (groupElements != 0 && requiredBytes > ubSize) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            sourceRowElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        sourceRowElements = SaturatingAdd(sourceRowElements, inputRowElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(
            SaturatingMul(sourceBatchRows, sourceRowElements), GATHER_DTYPE_BYTES);
        const uint64_t matrixBytes = SaturatingMul(
            SaturatingMul(batchRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t requiredBytesAfterSplit = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(sourceBytes, SaturatingMul(2, matrixBytes)));
        if (requiredBytesAfterSplit > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    const uint64_t sourceBatches = CeilDiv(plan.rowsPerCore, sourceBatchRows);
    const uint64_t outputBatches = CeilDiv(plan.rowsPerCore, batchRows);
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), outputBatches);
    uint64_t gatherCalls = 0;
    for (uint32_t elements : plan.groupElements) {
        gatherCalls = SaturatingAdd(gatherCalls, SaturatingMul(outputBatches, CeilDiv(
            SaturatingMul(batchRows, AlignUp(elements, GATHER_ELEMENT_ALIGNMENT)),
            GATHER_MAX_ELEMENTS_PER_CALL)));
    }
    // 行主序 source 的 offset 只需每组初始化一次，后续小批以 source 子张量为基址复用。
    const uint64_t offsetWork = SaturatingMul(
        SaturatingMul(sumAlignedElements, batchRows + 1), 2 * sizeof(int32_t));
    plan.score = SaturatingAdd(SaturatingMul(
        SaturatingAdd(inputCalls, SaturatingAdd(outputCalls, gatherCalls)),
        DMA_CALL_EQUIVALENT_BYTES), offsetWork);
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(maxSourceBytes, SaturatingMul(2, maxOutputBytes)));
    if (maxSourceBytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }
    plan.batchRows = batchRows;
    plan.sourceBatchRows = sourceBatchRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    return true;
}

bool BuildRowMajorStagedGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& best)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.batchRows >= baseline.rowsPerCore) {
        return false;
    }
    const GatherCommandStats baselineStats = GetGatherCommandStats(inputs, baseline);
    const uint64_t baselineControlCalls = SaturatingAdd(
        baselineStats.outputCalls, baselineStats.gatherCalls);
    const uint32_t sourceBatchLimit = static_cast<uint32_t>(std::min<uint64_t>(
        baseline.rowsPerCore, SaturatingMul(baseline.batchRows, 2)));
    bool found = false;
    GatherCommandStats bestStats;
    for (uint32_t sourceBatchRows = baseline.batchRows + 1;
         sourceBatchRows <= sourceBatchLimit; ++sourceBatchRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = baseline.rowsPerCore;
        if (!BuildRowMajorStagedGatherGroups(inputs, baseline.batchRows, sourceBatchRows,
            ubSize, candidate)) {
            continue;
        }
        const GatherCommandStats candidateStats = GetGatherCommandStats(inputs, candidate);
        const uint64_t candidateControlCalls = SaturatingAdd(
            candidateStats.outputCalls, candidateStats.gatherCalls);
        const uint64_t inputCallSaving = baselineStats.inputCalls > candidateStats.inputCalls ?
            baselineStats.inputCalls - candidateStats.inputCalls : 0;
        const uint64_t controlCallGrowth = candidateControlCalls > baselineControlCalls ?
            candidateControlCalls - baselineControlCalls : 0;
        // 只有 MTE2 发射的实际减少大于额外输出/Gather 控制命令时才切换数据流。
        if (inputCallSaving == 0 || inputCallSaving <= controlCallGrowth) {
            continue;
        }
        const uint64_t bestControlCalls = SaturatingAdd(
            bestStats.outputCalls, bestStats.gatherCalls);
        if (!found || candidateStats.inputCalls < bestStats.inputCalls ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls < bestControlCalls) ||
            (candidateStats.inputCalls == bestStats.inputCalls &&
             candidateControlCalls == bestControlCalls &&
             candidate.sourceBatchRows < best.sourceBatchRows)) {
            best = std::move(candidate);
            bestStats = candidateStats;
            found = true;
        }
    }
    return found;
}

bool BuildStagedSourceRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildStagedSourceGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildStagedSourceGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildDeltaStagedOffsetRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildDeltaStagedOffsetGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildDeltaStagedOffsetGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildFullStageOffsetRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildFullStageOffsetGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildFullStageOffsetGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildCachedOffsetStagedSourceRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildCachedOffsetStagedGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildCachedOffsetStagedGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildRowMajorStagedSourceRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildRowMajorStagedGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildRowMajorStagedGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildMergedBatchRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    // 保持单 source、单 output 和单 offset 布局；该候选只改变每次搬入的行数。
    if (!BuildMergedBatchGatherSubPlan(inputs, baseline.subPlans[0], ubSize, 1, 2, lowPlan) ||
        !BuildMergedBatchGatherSubPlan(inputs, baseline.subPlans[1], ubSize, 1, 2, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

bool BuildRowGatherPlanWithBufferCounts(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim, uint64_t ubSize,
    uint32_t dtypeSize, uint32_t sourceBufferCount, uint32_t matrixBufferCount,
    RowGatherPlan& plan, uint32_t persistentOffsetRows = 0)
{
    if (dtypeSize != GATHER_DTYPE_BYTES || totalBytes < GetGatherMinTotalBytes(ubSize) ||
        inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS || blockDim == 0 || outer < blockDim ||
        outputRowBytes == 0 || outputRowBytes % GATHER_DTYPE_BYTES != 0 ||
        outputRowBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t balancedLowRows = static_cast<uint32_t>(outer / blockDim);
    const uint32_t highRows = static_cast<uint32_t>(CeilDiv(outer, blockDim));
    GatherSubPlan balancedLowPlan;
    GatherSubPlan highPlan;
    if (balancedLowRows == 0 || highRows > DATA_COPY_MAX_ROWS ||
        !BuildGatherSubPlanWithBufferCounts(inputs, balancedLowRows, ubSize,
            sourceBufferCount, matrixBufferCount, balancedLowPlan, persistentOffsetRows) ||
        !BuildGatherSubPlanWithBufferCounts(inputs, highRows, ubSize,
            sourceBufferCount, matrixBufferCount, highPlan, persistentOffsetRows)) {
        return false;
    }
    plan.subPlans[0] = balancedLowPlan;
    plan.subPlans[1] = highPlan;
    plan.highCoreCount = highRows == balancedLowRows ? 0 :
        static_cast<uint32_t>(outer % blockDim);

    auto coreCost = [&](const GatherSubPlan& subPlan) -> uint64_t {
        return SaturatingAdd(SaturatingMul(subPlan.rowsPerCore, outputRowBytes), subPlan.score);
    };
    uint64_t bestMaxCost = std::max(coreCost(plan.subPlans[0]), coreCost(plan.subPlans[1]));
    uint64_t bestTotalCost = SaturatingAdd(
        SaturatingMul(blockDim - plan.highCoreCount, coreCost(plan.subPlans[0])),
        SaturatingMul(plan.highCoreCount, coreCost(plan.subPlans[1])));
    if (highRows != balancedLowRows) {
        const uint32_t searchRows = static_cast<uint32_t>(
            CeilDiv(balancedLowRows, GATHER_ROW_SEARCH_DIVISOR));
        const uint32_t minLowRows = std::max<uint32_t>(1, balancedLowRows - searchRows);
        for (uint32_t lowRows = minLowRows; lowRows < balancedLowRows; ++lowRows) {
            const uint64_t rowDelta = highRows - lowRows;
            const uint64_t highRowsTotal = outer - SaturatingMul(blockDim, lowRows);
            if (rowDelta == 0 || highRowsTotal % rowDelta != 0) {
                continue;
            }
            const uint64_t highCoreCount64 = highRowsTotal / rowDelta;
            if (highCoreCount64 == 0 || highCoreCount64 >= blockDim) {
                continue;
            }
            GatherSubPlan lowPlan;
            if (!BuildGatherSubPlanWithBufferCounts(inputs, lowRows, ubSize,
                sourceBufferCount, matrixBufferCount, lowPlan, persistentOffsetRows)) {
                continue;
            }
            const uint32_t highCoreCount = static_cast<uint32_t>(highCoreCount64);
            const uint64_t lowCost = coreCost(lowPlan);
            const uint64_t highCost = coreCost(highPlan);
            const uint64_t candidateMaxCost = std::max(lowCost, highCost);
            const uint64_t candidateTotalCost = SaturatingAdd(
                SaturatingMul(blockDim - highCoreCount, lowCost),
                SaturatingMul(highCoreCount, highCost));
            const bool improvesMax = candidateMaxCost < bestMaxCost;
            const bool keepsTotal = candidateTotalCost <= bestTotalCost;
            const bool improvesTie = candidateMaxCost == bestMaxCost &&
                candidateTotalCost < bestTotalCost;
            if ((improvesMax && keepsTotal) || improvesTie) {
                plan.subPlans[0] = std::move(lowPlan);
                plan.subPlans[1] = highPlan;
                plan.highCoreCount = highCoreCount;
                bestMaxCost = candidateMaxCost;
                bestTotalCost = candidateTotalCost;
            }
        }
    }
    return true;
}

bool BuildRowGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim,
    uint64_t ubSize, uint32_t dtypeSize, RowGatherPlan& plan)
{
    return BuildRowGatherPlanWithBufferCounts(inputs, outer, totalBytes, outputRowBytes,
        blockDim, ubSize, dtypeSize, 1, 2, plan);
}

// 仅复用常规 RowGather 已分配的 source 尾部。每个可选组在相邻输入的末端
// 选择最长连续后缀：当前批使用普通 [0, G) 布局，下一批后缀在空闲的另一侧
// 预取，随后只补齐前缀即可组成下一批布局。所有门槛均来自 UB 容量、输入拓扑
// 和流水深度，不读取公开 case 或固定 shape。
bool BuildSuffixPrefetchRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    plan = baseline;
    bool sawCandidate = false;
    for (GatherSubPlan& subPlan : plan.subPlans) {
        if (subPlan.rowsPerCore == 0 || subPlan.batchRows == 0 ||
            subPlan.sourceBatchRows != subPlan.batchRows ||
            subPlan.rowsPerCore < 2 * subPlan.batchRows ||
            subPlan.groupInputEnd.size() != subPlan.groupElements.size() ||
            subPlan.groupInputEnd.size() != subPlan.groupSourceElements.size()) {
            return false;
        }
        subPlan.groupSuffixInputStart = subPlan.groupInputEnd;
        subPlan.groupSuffixOutputStart.assign(subPlan.groupInputEnd.size(), 0);
        subPlan.groupSuffixElements.assign(subPlan.groupInputEnd.size(), 0);
        subPlan.groupSuffixSourceElements.assign(subPlan.groupInputEnd.size(), 0);
        const uint64_t sourceCapacityElements = subPlan.sourceBufferBytes / GATHER_DTYPE_BYTES;
        if (sourceCapacityElements == 0) {
            return false;
        }
        uint32_t groupInputStart = 0;
        for (size_t group = 0; group < subPlan.groupInputEnd.size(); ++group) {
            const uint32_t groupInputEnd = subPlan.groupInputEnd[group];
            const uint32_t groupElements = subPlan.groupElements[group];
            const uint32_t groupSourceElements = subPlan.groupSourceElements[group];
            if (groupInputEnd <= groupInputStart || groupInputEnd > inputs.size() ||
                groupElements == 0 || groupSourceElements == 0 ||
                groupSourceElements > sourceCapacityElements) {
                return false;
            }
            const uint64_t spareElements = sourceCapacityElements - groupSourceElements;
            uint64_t suffixSourceElements = 0;
            uint64_t suffixElements = 0;
            uint32_t suffixInputStart = groupInputEnd;
            for (uint32_t input = groupInputEnd; input > groupInputStart; --input) {
                const uint64_t inputElements = inputs[input - 1].rowBytes / GATHER_DTYPE_BYTES;
                const uint64_t inputSourceElements = AlignUp(
                    SaturatingMul(subPlan.batchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
                const uint64_t candidateSource = SaturatingAdd(suffixSourceElements,
                    inputSourceElements);
                if (candidateSource > spareElements) {
                    break;
                }
                suffixInputStart = input - 1;
                suffixSourceElements = candidateSource;
                suffixElements = SaturatingAdd(suffixElements, inputElements);
            }
            const uint64_t prefixElements = groupElements >= suffixElements ?
                groupElements - suffixElements : 0;
            // 预取后缀至少覆盖一个 4KB 等价搬运窗口，且输出仍需保留非空前缀；
            // 否则事件与偏移平移成本会超过可隐藏的 DMA 启动开销。
            if (suffixInputStart > groupInputStart && suffixInputStart < groupInputEnd &&
                prefixElements != 0 && suffixElements != 0 &&
                suffixSourceElements * GATHER_DTYPE_BYTES >= DMA_CALL_EQUIVALENT_BYTES &&
                suffixSourceElements <= spareElements &&
                SaturatingAdd(groupSourceElements, suffixSourceElements) <= sourceCapacityElements) {
                uint64_t outputPrefix = 0;
                for (uint32_t input = groupInputStart; input < suffixInputStart; ++input) {
                    outputPrefix = SaturatingAdd(outputPrefix,
                        inputs[input].rowBytes / GATHER_DTYPE_BYTES);
                }
                if (outputPrefix == prefixElements &&
                    outputPrefix <= std::numeric_limits<uint32_t>::max() &&
                    suffixElements <= std::numeric_limits<uint32_t>::max() &&
                    suffixSourceElements <= std::numeric_limits<uint32_t>::max()) {
                    subPlan.groupSuffixInputStart[group] =
                        static_cast<uint16_t>(suffixInputStart);
                    subPlan.groupSuffixOutputStart[group] =
                        static_cast<uint32_t>(outputPrefix);
                    subPlan.groupSuffixElements[group] =
                        static_cast<uint32_t>(suffixElements);
                    subPlan.groupSuffixSourceElements[group] =
                        static_cast<uint32_t>(suffixSourceElements);
                    sawCandidate = true;
                }
            }
            groupInputStart = groupInputEnd;
        }
        const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(subPlan.sourceBufferBytes,
                SaturatingAdd(subPlan.outputBufferBytes, subPlan.offsetBufferBytes)));
        if (allocatedBytes > ubSize) {
            return false;
        }
    }
    return sawCandidate;
}

// 将完整 output/offset 表复用同一 VECOUT，只保留 base/stride 两行 VECCALC seed。
// 该 UB 布局不依赖输入数或 shape；实际是否启用仍由后续命令数和流水深度门槛决定。
bool BuildOffsetOutputAliasRowGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim,
    uint64_t ubSize, uint32_t dtypeSize, RowGatherPlan& plan)
{
    return BuildRowGatherPlanWithBufferCounts(inputs, outer, totalBytes, outputRowBytes,
        blockDim, ubSize, dtypeSize, 1, 1, plan, OFFSET_OUTPUT_ALIAS_SEED_ROWS);
}

// 与 key43 相同的 output/offset 别名布局，但将节省的一张完整矩阵槽用于第二个 source
// 槽。Host 逐个 group 核算两张 source 槽，避免 key16 的 UB 挤压缩小批行数。
bool BuildSourcePingPongOffsetOutputAliasRowGatherPlan(const std::vector<InputInfo>& inputs,
    uint64_t outer, uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim,
    uint64_t ubSize, uint32_t dtypeSize, RowGatherPlan& plan)
{
    return BuildRowGatherPlanWithBufferCounts(inputs, outer, totalBytes, outputRowBytes,
        blockDim, ubSize, dtypeSize, 2, 1, plan, OFFSET_OUTPUT_ALIAS_SEED_ROWS);
}

bool BuildBandGatherTask(const std::vector<InputInfo>& inputs, uint32_t inputStart,
    uint32_t inputEnd, uint64_t outputStart, uint64_t groupElements, uint64_t rowCount,
    uint64_t ubSize, BandGatherTask& task)
{
    task = BandGatherTask{};
    if (inputStart >= inputEnd || inputEnd > inputs.size() || groupElements == 0 ||
        groupElements > std::numeric_limits<uint32_t>::max() ||
        outputStart > std::numeric_limits<uint32_t>::max() || rowCount == 0) {
        return false;
    }
    const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
    // 保持每个完整 batch 只需一次 Gather，避免用更多 Vector 指令交换 MTE2 节省。
    if (alignedElements == 0 || alignedElements > GATHER_MAX_ELEMENTS_PER_CALL) {
        return false;
    }
    const uint64_t batchLimit = std::min<uint64_t>({rowCount, DATA_COPY_MAX_ROWS,
        GATHER_MAX_ELEMENTS_PER_CALL / alignedElements,
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (batchLimit == 0) {
        return false;
    }

    auto getBufferLayout = [&](uint32_t batchRows, uint64_t& sourceElements,
        uint64_t& sourceBytes, uint64_t& matrixBytes) -> bool {
        sourceElements = 0;
        uint64_t observedElements = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            if (inputs[input].rowBytes % GATHER_DTYPE_BYTES != 0) {
                return false;
            }
            const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
            observedElements = SaturatingAdd(observedElements, inputElements);
            const uint64_t slotElements = AlignUp(SaturatingMul(batchRows, inputElements),
                GATHER_ELEMENT_ALIGNMENT);
            sourceElements = SaturatingAdd(sourceElements, slotElements);
        }
        if (observedElements != groupElements || sourceElements == 0) {
            return false;
        }
        sourceBytes = SaturatingMul(sourceElements, GATHER_DTYPE_BYTES);
        matrixBytes = SaturatingMul(SaturatingMul(batchRows, alignedElements),
            GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(
            sourceBytes, SaturatingMul(2, matrixBytes)));
        return sourceElements <= std::numeric_limits<uint32_t>::max() &&
            sourceBytes <= std::numeric_limits<uint32_t>::max() &&
            matrixBytes <= std::numeric_limits<uint32_t>::max() && requiredBytes <= ubSize;
    };

    uint32_t low = 1;
    uint32_t high = static_cast<uint32_t>(batchLimit);
    uint32_t selectedBatchRows = 0;
    while (low <= high) {
        const uint32_t middle = low + (high - low) / 2;
        uint64_t sourceElements = 0;
        uint64_t sourceBytes = 0;
        uint64_t matrixBytes = 0;
        if (getBufferLayout(middle, sourceElements, sourceBytes, matrixBytes)) {
            selectedBatchRows = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    if (selectedBatchRows == 0) {
        return false;
    }

    uint64_t sourceElements = 0;
    uint64_t sourceBytes = 0;
    uint64_t matrixBytes = 0;
    if (!getBufferLayout(selectedBatchRows, sourceElements, sourceBytes, matrixBytes)) {
        return false;
    }
    uint64_t nonzeroInputs = 0;
    for (uint32_t input = inputStart; input < inputEnd; ++input) {
        nonzeroInputs += static_cast<uint64_t>(inputs[input].rowBytes != 0);
    }
    const uint64_t batches = CeilDiv(rowCount, selectedBatchRows);
    if (nonzeroInputs == 0 || batches == 0) {
        return false;
    }
    task.inputStart = static_cast<uint16_t>(inputStart);
    task.inputEnd = static_cast<uint16_t>(inputEnd);
    task.outputStart = static_cast<uint32_t>(outputStart);
    task.groupElements = static_cast<uint32_t>(groupElements);
    task.sourceElements = static_cast<uint32_t>(sourceElements);
    task.batchRows = selectedBatchRows;
    task.sourceBufferBytes = static_cast<uint32_t>(sourceBytes);
    task.outputBufferBytes = static_cast<uint32_t>(matrixBytes);
    task.offsetBufferBytes = static_cast<uint32_t>(matrixBytes);
    task.rowCount = rowCount;
    task.inputCalls = SaturatingMul(nonzeroInputs, batches);
    task.outputCalls = batches;
    task.gatherCalls = batches;
    task.outputElements = SaturatingMul(rowCount, groupElements);
    const uint64_t offsetWork = SaturatingMul(SaturatingMul(alignedElements,
        static_cast<uint64_t>(selectedBatchRows) + 1), 3 * sizeof(int32_t));
    const uint64_t commandCalls = SaturatingAdd(task.inputCalls,
        SaturatingAdd(task.outputCalls, task.gatherCalls));
    task.criticalCost = SaturatingAdd(SaturatingMul(task.outputElements,
        GATHER_DTYPE_BYTES), SaturatingAdd(SaturatingMul(commandCalls,
        DMA_CALL_EQUIVALENT_BYTES), offsetWork));
    return true;
}

bool BuildBandGatherBaseline(const std::vector<InputInfo>& inputs, uint64_t outputRowBytes,
    uint32_t blockDim, const RowGatherPlan& rowGatherPlan, BandGatherBaseline& baseline)
{
    baseline = BandGatherBaseline{};
    if (blockDim == 0 || outputRowBytes == 0 ||
        outputRowBytes % GATHER_DTYPE_BYTES != 0 ||
        rowGatherPlan.highCoreCount > blockDim) {
        return false;
    }
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        if (input.rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const bool activePlans[optiling::CONCAT_GATHER_PLAN_COUNT] = {
        blockDim > rowGatherPlan.highCoreCount, rowGatherPlan.highCoreCount != 0};
    bool sawActivePlan = false;
    for (size_t planIndex = 0; planIndex < rowGatherPlan.subPlans.size(); ++planIndex) {
        if (!activePlans[planIndex]) {
            continue;
        }
        const GatherSubPlan& subPlan = rowGatherPlan.subPlans[planIndex];
        if (subPlan.rowsPerCore == 0 || subPlan.batchRows == 0 ||
            subPlan.sourceBatchRows == 0 || subPlan.groupInputEnd.empty() ||
            subPlan.groupInputEnd.size() != subPlan.groupElements.size()) {
            return false;
        }
        const uint64_t sourceBatches = CeilDiv(subPlan.rowsPerCore,
            subPlan.sourceBatchRows);
        const uint64_t outputBatches = CeilDiv(subPlan.rowsPerCore, subPlan.batchRows);
        uint64_t gatherCalls = 0;
        uint64_t alignedElements = 0;
        for (uint32_t groupElements : subPlan.groupElements) {
            const uint64_t alignedGroupElements = AlignUp(groupElements,
                GATHER_ELEMENT_ALIGNMENT);
            if (groupElements == 0 || alignedGroupElements == 0) {
                return false;
            }
            alignedElements = SaturatingAdd(alignedElements, alignedGroupElements);
            const uint64_t fullBatchCalls = CeilDiv(SaturatingMul(subPlan.batchRows,
                alignedGroupElements), GATHER_MAX_ELEMENTS_PER_CALL);
            gatherCalls = SaturatingAdd(gatherCalls, SaturatingMul(outputBatches,
                std::max<uint64_t>(fullBatchCalls, 1)));
        }
        const uint64_t inputCalls = SaturatingMul(nonzeroInputs, sourceBatches);
        const uint64_t outputCalls = SaturatingMul(subPlan.groupElements.size(), outputBatches);
        const uint64_t outputElements = SaturatingMul(subPlan.rowsPerCore,
            outputRowBytes / GATHER_DTYPE_BYTES);
        const uint64_t offsetWork = SaturatingMul(SaturatingMul(alignedElements,
            static_cast<uint64_t>(subPlan.batchRows) + 1), 3 * sizeof(int32_t));
        const uint64_t commandCalls = SaturatingAdd(inputCalls,
            SaturatingAdd(outputCalls, gatherCalls));
        const uint64_t criticalCost = SaturatingAdd(SaturatingMul(outputElements,
            GATHER_DTYPE_BYTES), SaturatingAdd(SaturatingMul(commandCalls,
            DMA_CALL_EQUIVALENT_BYTES), offsetWork));
        baseline.maxInputCalls = std::max(baseline.maxInputCalls, inputCalls);
        baseline.maxOutputCalls = std::max(baseline.maxOutputCalls, outputCalls);
        baseline.maxGatherCalls = std::max(baseline.maxGatherCalls, gatherCalls);
        baseline.maxOutputElements = std::max(baseline.maxOutputElements, outputElements);
        baseline.maxCriticalCost = std::max(baseline.maxCriticalCost, criticalCost);
        sawActivePlan = true;
    }
    return sawActivePlan && baseline.maxInputCalls != 0 && baseline.maxOutputCalls != 0 &&
        baseline.maxGatherCalls != 0 && baseline.maxOutputElements != 0 &&
        baseline.maxCriticalCost != 0;
}

bool BuildBandGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t outputRowBytes, uint32_t blockDim, uint64_t ubSize,
    const RowGatherPlan& rowGatherPlan, BandGatherPlan& plan)
{
    plan = BandGatherPlan{};
    if (blockDim == 0 || blockDim > optiling::CONCAT_MAX_BAND_GATHER_TASKS || outer < blockDim ||
        inputs.empty() || inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS ||
        outputRowBytes == 0 || outputRowBytes % GATHER_DTYPE_BYTES != 0) {
        return false;
    }
    std::vector<uint32_t> nonzeroPositions;
    std::vector<uint64_t> nonzeroPrefix;
    nonzeroPrefix.push_back(0);
    uint64_t outputElements = 0;
    for (size_t input = 0; input < inputs.size(); ++input) {
        if (inputs[input].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
        outputElements = SaturatingAdd(outputElements, inputElements);
        if (inputElements != 0) {
            nonzeroPositions.push_back(static_cast<uint32_t>(input));
            nonzeroPrefix.push_back(SaturatingAdd(nonzeroPrefix.back(), inputElements));
        }
    }
    if (outputElements != outputRowBytes / GATHER_DTYPE_BYTES ||
        nonzeroPositions.size() < CeilDiv(blockDim, BAND_GATHER_MAX_CORES_PER_BAND)) {
        return false;
    }
    const uint64_t maxBandElements = CeilDiv(SaturatingMul(outputElements,
        BAND_GATHER_MAX_WIDTH_FACTOR), blockDim);
    if (maxBandElements == 0 || maxBandElements > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    BandGatherBaseline baseline;
    if (!BuildBandGatherBaseline(inputs, outputRowBytes, blockDim, rowGatherPlan, baseline)) {
        return false;
    }

    const size_t inputBandCount = nonzeroPositions.size();
    std::vector<std::vector<BandGatherDpState>> dp(inputBandCount + 1,
        std::vector<BandGatherDpState>(blockDim + 1));
    dp[0][0].maxCost = 0;
    dp[0][0].totalCost = 0;
    dp[0][0].valid = true;
    for (size_t start = 0; start < inputBandCount; ++start) {
        for (uint32_t usedCores = 0; usedCores <= blockDim; ++usedCores) {
            if (!dp[start][usedCores].valid) {
                continue;
            }
            for (size_t end = start + 1; end <= inputBandCount; ++end) {
                const uint64_t groupElements = nonzeroPrefix[end] - nonzeroPrefix[start];
                if (groupElements > maxBandElements) {
                    break;
                }
                const uint32_t inputStart = start == 0 ? 0 : nonzeroPositions[start - 1] + 1;
                const uint32_t inputEnd = end == inputBandCount ?
                    static_cast<uint32_t>(inputs.size()) : nonzeroPositions[end - 1] + 1;
                if (inputStart >= inputEnd) {
                    return false;
                }
                for (uint32_t allocatedCores = 1;
                    allocatedCores <= BAND_GATHER_MAX_CORES_PER_BAND &&
                    usedCores + allocatedCores <= blockDim; ++allocatedCores) {
                    const uint64_t lowRows = outer / allocatedCores;
                    const uint64_t highCoreCount = outer % allocatedCores;
                    uint64_t optionMaxCost = 0;
                    uint64_t optionTotalCost = 0;
                    bool validOption = lowRows != 0;
                    for (uint32_t core = 0; validOption && core < allocatedCores; ++core) {
                        const uint64_t rowCount = lowRows +
                            static_cast<uint64_t>(core < highCoreCount);
                        BandGatherTask task;
                        if (!BuildBandGatherTask(inputs, inputStart, inputEnd,
                            nonzeroPrefix[start], groupElements, rowCount, ubSize, task)) {
                            validOption = false;
                            break;
                        }
                        optionMaxCost = std::max(optionMaxCost, task.criticalCost);
                        optionTotalCost = SaturatingAdd(optionTotalCost, task.criticalCost);
                    }
                    if (!validOption) {
                        continue;
                    }
                    const uint32_t nextCores = usedCores + allocatedCores;
                    BandGatherDpState& next = dp[end][nextCores];
                    const uint64_t candidateMaxCost = std::max(dp[start][usedCores].maxCost,
                        optionMaxCost);
                    const uint64_t candidateTotalCost = SaturatingAdd(
                        dp[start][usedCores].totalCost, optionTotalCost);
                    if (!next.valid || candidateMaxCost < next.maxCost ||
                        (candidateMaxCost == next.maxCost &&
                         candidateTotalCost < next.totalCost)) {
                        next.maxCost = candidateMaxCost;
                        next.totalCost = candidateTotalCost;
                        next.previousInput = static_cast<uint16_t>(start);
                        next.previousCores = static_cast<uint16_t>(usedCores);
                        next.allocatedCores = static_cast<uint8_t>(allocatedCores);
                        next.valid = true;
                    }
                }
            }
        }
    }
    if (!dp[inputBandCount][blockDim].valid) {
        return false;
    }

    struct BandInterval {
        uint16_t start{0};
        uint16_t end{0};
        uint8_t allocatedCores{0};
    };
    std::vector<BandInterval> intervals;
    size_t end = inputBandCount;
    uint32_t usedCores = blockDim;
    while (end != 0) {
        const BandGatherDpState& state = dp[end][usedCores];
        if (!state.valid || state.allocatedCores == 0 || state.previousInput >= end ||
            state.previousCores >= usedCores) {
            return false;
        }
        BandInterval interval;
        interval.start = state.previousInput;
        interval.end = static_cast<uint16_t>(end);
        interval.allocatedCores = state.allocatedCores;
        intervals.push_back(interval);
        end = state.previousInput;
        usedCores = state.previousCores;
    }
    if (usedCores != 0) {
        return false;
    }
    std::reverse(intervals.begin(), intervals.end());
    for (const BandInterval& interval : intervals) {
        const uint32_t inputStart = interval.start == 0 ? 0 :
            nonzeroPositions[interval.start - 1] + 1;
        const uint32_t inputEnd = interval.end == inputBandCount ?
            static_cast<uint32_t>(inputs.size()) : nonzeroPositions[interval.end - 1] + 1;
        const uint64_t groupElements = nonzeroPrefix[interval.end] -
            nonzeroPrefix[interval.start];
        const uint64_t lowRows = outer / interval.allocatedCores;
        const uint64_t highCoreCount = outer % interval.allocatedCores;
        uint64_t rowStart = 0;
        for (uint32_t core = 0; core < interval.allocatedCores; ++core) {
            const uint64_t rowCount = lowRows + static_cast<uint64_t>(core < highCoreCount);
            BandGatherTask task;
            if (!BuildBandGatherTask(inputs, inputStart, inputEnd,
                nonzeroPrefix[interval.start], groupElements, rowCount, ubSize, task)) {
                return false;
            }
            task.rowStart = rowStart;
            rowStart += rowCount;
            plan.maxInputCalls = std::max(plan.maxInputCalls, task.inputCalls);
            plan.maxOutputCalls = std::max(plan.maxOutputCalls, task.outputCalls);
            plan.maxGatherCalls = std::max(plan.maxGatherCalls, task.gatherCalls);
            plan.maxOutputElements = std::max(plan.maxOutputElements, task.outputElements);
            plan.maxCriticalCost = std::max(plan.maxCriticalCost, task.criticalCost);
            plan.tasks.push_back(std::move(task));
        }
        if (rowStart != outer) {
            return false;
        }
    }
    if (plan.tasks.size() != blockDim ||
        plan.tasks.size() > optiling::CONCAT_MAX_BAND_GATHER_TASKS ||
        plan.maxInputCalls == 0 || plan.maxOutputCalls == 0 || plan.maxGatherCalls == 0 ||
        plan.maxOutputElements == 0 || plan.maxCriticalCost == 0) {
        return false;
    }

    // 只有输入连续 DMA 指令明显下降、且输出/Gather 命令与最慢核输出量仍受控时才交换
    // 为二维调度。所有门槛都表达运行时任务代价，不依赖 case、shape 或输入数的固定值。
    if (SaturatingMul(plan.maxInputCalls, 100) > SaturatingMul(baseline.maxInputCalls,
            BAND_GATHER_REQUIRED_INPUT_CALL_PERCENT) ||
        plan.maxOutputCalls > SaturatingAdd(baseline.maxOutputCalls,
            BAND_GATHER_MAX_EXTRA_COMMANDS) ||
        plan.maxGatherCalls > SaturatingAdd(baseline.maxGatherCalls,
            BAND_GATHER_MAX_EXTRA_COMMANDS) ||
        SaturatingMul(plan.maxOutputElements, 100) > SaturatingMul(
            baseline.maxOutputElements, BAND_GATHER_MAX_OUTPUT_WORK_PERCENT) ||
        plan.maxCriticalCost >= baseline.maxCriticalCost) {
        return false;
    }
    return true;
}

struct PairGatherTemplate {
    uint16_t inputStart{0};
    uint16_t inputSplit{0};
    uint16_t inputEnd{0};
    uint32_t outputStart{0};
    uint32_t group0Elements{0};
    uint32_t group1Elements{0};
    uint32_t pairElements{0};
    uint32_t sourceElements{0};
};

bool BuildPairTemplates(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint64_t pairCap, uint64_t groupCap, uint64_t ubSize,
    std::vector<PairGatherTemplate>& templates)
{
    templates.clear();
    if (batchRows == 0 || pairCap < 16 || groupCap < 8 || inputs.empty()) {
        return false;
    }
    std::vector<uint64_t> sourcePrefix(inputs.size() + 1, 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint64_t elements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        sourcePrefix[i + 1] = SaturatingAdd(sourcePrefix[i], AlignUp(
            SaturatingMul(batchRows, elements), GATHER_ELEMENT_ALIGNMENT));
    }
    uint64_t outputStart = 0;
    size_t cursor = 0;
    while (cursor < inputs.size()) {
        bool found = false;
        PairGatherTemplate best;
        uint64_t bestPairElements = 0;
        uint64_t bestSourceElements = std::numeric_limits<uint64_t>::max();
        uint64_t group0Prefix = 0;
        for (size_t split = cursor + 1; split < inputs.size(); ++split) {
            group0Prefix = SaturatingAdd(group0Prefix,
                inputs[split - 1].rowBytes / GATHER_DTYPE_BYTES);
            if (group0Prefix == 0 || group0Prefix > groupCap ||
                group0Prefix % GATHER_ELEMENT_ALIGNMENT != 0) {
                continue;
            }
            uint64_t group1Prefix = 0;
            for (size_t end = split + 1; end <= inputs.size(); ++end) {
                group1Prefix = SaturatingAdd(group1Prefix,
                    inputs[end - 1].rowBytes / GATHER_DTYPE_BYTES);
                if (group1Prefix == 0) {
                    continue;
                }
                const uint64_t pairElements = SaturatingAdd(group0Prefix, group1Prefix);
                if (pairElements > pairCap) {
                    break;
                }
                const uint64_t source0 = sourcePrefix[split] - sourcePrefix[cursor];
                const uint64_t source1 = sourcePrefix[end] - sourcePrefix[split];
                // 两个输入组依次复用同一张 source stage，避免为配对路线复制 source UB。
                const uint64_t sourceElements = std::max(source0, source1);
                const uint64_t outputBytes = SaturatingMul(
                    SaturatingMul(batchRows, AlignUp(pairElements, GATHER_ELEMENT_ALIGNMENT)),
                    GATHER_DTYPE_BYTES);
                const uint64_t maxGroupAligned = AlignUp(std::max(
                    AlignUp(group0Prefix, GATHER_ELEMENT_ALIGNMENT),
                    AlignUp(group1Prefix, GATHER_ELEMENT_ALIGNMENT)), MICRO_GATHER_VF);
                const uint64_t offsetBytes = SaturatingAdd(
                    SaturatingMul(2, SaturatingMul(maxGroupAligned, GATHER_DTYPE_BYTES)),
                    SaturatingMul(SaturatingAdd(batchRows, 1), MICRO_GATHER_VF * GATHER_DTYPE_BYTES));
                const uint64_t required = SaturatingAdd(UB_RESERVE_BYTES,
                    SaturatingAdd(SaturatingMul(sourceElements, GATHER_DTYPE_BYTES),
                        SaturatingAdd(outputBytes, offsetBytes)));
                if (required > ubSize) {
                    continue;
                }
                if (!found || pairElements > bestPairElements ||
                    (pairElements == bestPairElements && sourceElements < bestSourceElements)) {
                    found = true;
                    bestPairElements = pairElements;
                    bestSourceElements = sourceElements;
                    best.inputStart = static_cast<uint16_t>(cursor);
                    best.inputSplit = static_cast<uint16_t>(split);
                    best.inputEnd = static_cast<uint16_t>(end);
                    best.outputStart = static_cast<uint32_t>(outputStart);
                    best.group0Elements = static_cast<uint32_t>(group0Prefix);
                    best.group1Elements = static_cast<uint32_t>(group1Prefix);
                    best.pairElements = static_cast<uint32_t>(pairElements);
                    best.sourceElements = static_cast<uint32_t>(sourceElements);
                }
            }
        }
        if (!found || templates.size() >= optiling::CONCAT_MAX_PAIR_GATHER_TASKS) {
            return false;
        }
        templates.push_back(best);
        outputStart = SaturatingAdd(outputStart, best.pairElements);
        cursor = best.inputEnd;
    }
    return !templates.empty() && outputStart != 0;
}

bool BuildPairGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t outputRowBytes, uint32_t blockDim, uint64_t ubSize,
    const RowGatherPlan& rowGatherPlan, PairGatherPlan& plan)
{
    plan = PairGatherPlan{};
    if (blockDim == 0 || outer < blockDim || outputRowBytes == 0 ||
        outputRowBytes % GATHER_DTYPE_BYTES != 0 || inputs.size() < 2 ||
        inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS) {
        return false;
    }
    for (const InputInfo& input : inputs) {
        if (input.rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
    }
    BandGatherBaseline baseline;
    if (!BuildBandGatherBaseline(inputs, outputRowBytes, blockDim, rowGatherPlan, baseline)) {
        return false;
    }

    const uint64_t available = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : 0;
    PairGatherPlan bestPlan;
    bool foundPlan = false;
    // 逐步尝试可容纳的批行数和配对宽度；候选只由运行时 UB、输入拓扑和 DMA 限制决定。
    for (uint32_t batchRows = 1; batchRows <= std::min<uint64_t>(255, outer); ++batchRows) {
        const uint64_t roughPairCap = AlignDown(
            available / std::max<uint64_t>(1, SaturatingMul(batchRows, GATHER_DTYPE_BYTES)),
            GATHER_ELEMENT_ALIGNMENT);
        if (roughPairCap < 16) {
            continue;
        }
        for (uint32_t divisor = 2; divisor <= 4; ++divisor) {
            const uint64_t groupCap = AlignDown(roughPairCap / divisor,
                GATHER_ELEMENT_ALIGNMENT);
            if (groupCap < GATHER_ELEMENT_ALIGNMENT) {
                continue;
            }
            std::vector<PairGatherTemplate> templates;
            if (!BuildPairTemplates(inputs, batchRows, roughPairCap, groupCap, ubSize,
                templates) || templates.size() > blockDim) {
                continue;
            }
            std::vector<uint32_t> coreCounts(templates.size(), 1);
            uint32_t remaining = blockDim - static_cast<uint32_t>(templates.size());
            while (remaining != 0) {
                size_t best = 0;
                double bestLoad = -1.0;
                for (size_t i = 0; i < templates.size(); ++i) {
                    const double load = static_cast<double>(templates[i].pairElements) /
                        static_cast<double>(coreCounts[i]);
                    if (load > bestLoad) {
                        bestLoad = load;
                        best = i;
                    }
                }
                ++coreCounts[best];
                --remaining;
            }
            PairGatherPlan candidate;
            candidate.maxInputCalls = 0;
            candidate.maxOutputCalls = 0;
            candidate.maxGatherCalls = 0;
            candidate.maxOutputElements = 0;
            candidate.maxCriticalCost = 0;
            uint64_t rowStart = 0;
            bool valid = true;
            for (size_t i = 0; i < templates.size() && valid; ++i) {
                const PairGatherTemplate& item = templates[i];
                const uint32_t cores = coreCounts[i];
                const uint64_t lowRows = outer / cores;
                const uint64_t highCount = outer % cores;
                const uint64_t sourceBytes = SaturatingMul(item.sourceElements,
                    GATHER_DTYPE_BYTES);
                const uint64_t outputBytes = SaturatingMul(
                    SaturatingMul(batchRows, AlignUp(item.pairElements, GATHER_ELEMENT_ALIGNMENT)),
                    GATHER_DTYPE_BYTES);
                const uint64_t maxGroupAligned = AlignUp(std::max(
                    AlignUp(item.group0Elements, GATHER_ELEMENT_ALIGNMENT),
                    AlignUp(item.group1Elements, GATHER_ELEMENT_ALIGNMENT)), MICRO_GATHER_VF);
                const uint64_t offsetBytes = SaturatingAdd(
                    SaturatingMul(2, SaturatingMul(maxGroupAligned, GATHER_DTYPE_BYTES)),
                    SaturatingMul(SaturatingAdd(batchRows, 1), MICRO_GATHER_VF * GATHER_DTYPE_BYTES));
                if (sourceBytes > std::numeric_limits<uint32_t>::max() ||
                    outputBytes > std::numeric_limits<uint32_t>::max() ||
                    offsetBytes > std::numeric_limits<uint32_t>::max()) {
                    valid = false;
                    break;
                }
                for (uint32_t core = 0; core < cores; ++core) {
                    const uint64_t rowCount = lowRows + static_cast<uint64_t>(core < highCount);
                    if (rowCount == 0 || candidate.tasks.size() >=
                        optiling::CONCAT_MAX_PAIR_GATHER_TASKS) {
                        valid = false;
                        break;
                    }
                    const uint64_t batches = CeilDiv(rowCount, batchRows);
                    const uint64_t nonzero0 = std::count_if(inputs.begin() + item.inputStart,
                        inputs.begin() + item.inputSplit,
                        [](const InputInfo& x) { return x.rowBytes != 0; });
                    const uint64_t nonzero1 = std::count_if(inputs.begin() + item.inputSplit,
                        inputs.begin() + item.inputEnd,
                        [](const InputInfo& x) { return x.rowBytes != 0; });
                    const uint64_t inputCalls = SaturatingMul(nonzero0 + nonzero1, batches);
                    const uint64_t outputCalls = batches;
                    const uint64_t gatherCalls = SaturatingMul(batches,
                        CeilDiv(item.group0Elements, MICRO_GATHER_VF) +
                        CeilDiv(item.group1Elements, MICRO_GATHER_VF));
                    const uint64_t outputElements = SaturatingMul(rowCount, item.pairElements);
                    const uint64_t offsetWork = SaturatingMul(
                        SaturatingMul(static_cast<uint64_t>(batchRows + 1), MICRO_GATHER_VF),
                        3 * sizeof(int32_t));
                    const uint64_t critical = SaturatingAdd(
                        SaturatingMul(outputElements, GATHER_DTYPE_BYTES),
                        SaturatingAdd(SaturatingMul(inputCalls + outputCalls + gatherCalls,
                            DMA_CALL_EQUIVALENT_BYTES), offsetWork));
                    PairGatherTask task;
                    task.inputStart = item.inputStart;
                    task.inputSplit = item.inputSplit;
                    task.inputEnd = item.inputEnd;
                    task.outputStart = item.outputStart;
                    task.group0Elements = item.group0Elements;
                    task.group1Elements = item.group1Elements;
                    task.pairElements = item.pairElements;
                    task.sourceElements = item.sourceElements;
                    task.batchRows = batchRows;
                    task.sourceBufferBytes = static_cast<uint32_t>(sourceBytes);
                    task.outputBufferBytes = static_cast<uint32_t>(outputBytes);
                    task.offsetBufferBytes = static_cast<uint32_t>(offsetBytes);
                    task.rowStart = rowStart;
                    task.rowCount = rowCount;
                    candidate.tasks.push_back(task);
                    candidate.maxInputCalls = std::max(candidate.maxInputCalls, inputCalls);
                    candidate.maxOutputCalls = std::max(candidate.maxOutputCalls, outputCalls);
                    candidate.maxGatherCalls = std::max(candidate.maxGatherCalls, gatherCalls);
                    candidate.maxOutputElements = std::max(candidate.maxOutputElements, outputElements);
                    candidate.maxCriticalCost = std::max(candidate.maxCriticalCost, critical);
                    rowStart += rowCount;
                }
            }
            if (!valid || candidate.tasks.size() != blockDim || rowStart != outer) {
                continue;
            }
            // 配对必须同时降低输入和输出命令，且最慢核代价有严格下降，避免把边缘
            // 的单项 sub-metric 改善误当作端到端收益。
            const bool inputRejected = candidate.maxInputCalls * 100 > baseline.maxInputCalls * 80;
            const bool outputRejected = candidate.maxOutputCalls >= baseline.maxOutputCalls;
            const bool criticalRejected = candidate.maxCriticalCost >= baseline.maxCriticalCost;
            if (inputRejected || outputRejected || criticalRejected) {
                continue;
            }
            if (!foundPlan || candidate.maxCriticalCost < bestPlan.maxCriticalCost ||
                (candidate.maxCriticalCost == bestPlan.maxCriticalCost &&
                 candidate.maxGatherCalls < bestPlan.maxGatherCalls)) {
                foundPlan = true;
                bestPlan = std::move(candidate);
            }
        }
    }
    if (!foundPlan) {
        return false;
    }
    plan = std::move(bestPlan);
    return true;
}

// Shared seed 仅保存每个 group 的首行地址和逐行增量。消费者仍在自身 UB 中展开完整
// offset 表，因此不依赖固定 shape、输入切分或某个核的行数。所有可能作为 producer 的
// 核都必须能容纳任意一组的两条 seed 向量，避免两套 subplan 的缓冲容量不同而越界。
bool BuildSharedSeedWorkspace(const RowGatherPlan& plan, uint32_t blockDim,
    uint64_t totalBytes, uint64_t& workspaceBytes)
{
    workspaceBytes = 0;
    if (blockDim == 0 || totalBytes == 0 || plan.highCoreCount > blockDim) {
        return false;
    }
    const bool activePlans[optiling::CONCAT_GATHER_PLAN_COUNT] = {
        blockDim > plan.highCoreCount, plan.highCoreCount != 0};
    uint64_t producerGroups = 0;
    uint64_t maxSeedBytes = 0;
    for (size_t planIndex = 0; planIndex < optiling::CONCAT_GATHER_PLAN_COUNT; ++planIndex) {
        if (!activePlans[planIndex]) {
            continue;
        }
        const GatherSubPlan& subPlan = plan.subPlans[planIndex];
        if (subPlan.rowsPerCore == 0 || subPlan.batchRows < 2 ||
            subPlan.groupInputEnd.empty() ||
            subPlan.groupInputEnd.size() != subPlan.groupElements.size()) {
            return false;
        }
        for (uint32_t groupElements : subPlan.groupElements) {
            if (groupElements == 0) {
                return false;
            }
            const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
            uint64_t seedBytes = 0;
            if (MulOverflow(alignedElements, 2 * sizeof(int32_t), seedBytes) ||
                AddOverflow(workspaceBytes, seedBytes, workspaceBytes)) {
                return false;
            }
            maxSeedBytes = std::max(maxSeedBytes, seedBytes);
            ++producerGroups;
        }
    }
    if (producerGroups == 0 || producerGroups > blockDim ||
        workspaceBytes == 0 || workspaceBytes > totalBytes / 8) {
        return false;
    }
    for (size_t planIndex = 0; planIndex < optiling::CONCAT_GATHER_PLAN_COUNT; ++planIndex) {
        if (activePlans[planIndex] && plan.subPlans[planIndex].offsetBufferBytes < maxSeedBytes) {
            return false;
        }
    }
    return true;
}

bool BuildPartitionedGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& plan)
{
    if (baseline.batchRows == 0 || baseline.sourceBatchRows != baseline.batchRows ||
        baseline.groupInputEnd.empty() || baseline.groupInputEnd.size() != baseline.groupElements.size() ||
        baseline.groupInputEnd.size() != baseline.groupSourceElements.size()) {
        return false;
    }
    plan = baseline;
    plan.groupInputSplit.clear();
    uint32_t inputStart = 0;
    uint64_t maxSourcePart0Bytes = 0;
    uint64_t maxSourcePart1Bytes = 0;
    uint64_t maxOutputPart0Bytes = 0;
    uint64_t maxOutputPart1Bytes = 0;

    for (size_t group = 0; group < baseline.groupInputEnd.size(); ++group) {
        const uint32_t inputEnd = baseline.groupInputEnd[group];
        if (inputEnd <= inputStart + 1 || inputEnd > inputs.size()) {
            return false;
        }

        uint64_t totalSourceElements = 0;
        uint64_t totalOutputElements = 0;
        for (uint32_t input = inputStart; input < inputEnd; ++input) {
            if (inputs[input].rowBytes % GATHER_DTYPE_BYTES != 0) {
                return false;
            }
            const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
            totalOutputElements = SaturatingAdd(totalOutputElements, inputElements);
            totalSourceElements = SaturatingAdd(totalSourceElements, AlignUp(
                SaturatingMul(baseline.sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT));
        }
        if (totalOutputElements != baseline.groupElements[group] || totalSourceElements == 0 ||
            totalSourceElements != baseline.groupSourceElements[group]) {
            return false;
        }

        uint64_t sourcePart0Elements = 0;
        uint64_t outputPart0Elements = 0;
        uint64_t bestBalance = std::numeric_limits<uint64_t>::max();
        uint32_t bestSplit = 0;
        for (uint32_t input = inputStart; input + 1 < inputEnd; ++input) {
            const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
            outputPart0Elements = SaturatingAdd(outputPart0Elements, inputElements);
            sourcePart0Elements = SaturatingAdd(sourcePart0Elements, AlignUp(
                SaturatingMul(baseline.sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT));
            const uint64_t outputPart1Elements = totalOutputElements - outputPart0Elements;
            const uint64_t sourcePart1Elements = totalSourceElements - sourcePart0Elements;
            if (outputPart0Elements == 0 || outputPart1Elements == 0 ||
                sourcePart0Elements == 0 || sourcePart1Elements == 0) {
                continue;
            }
            const uint64_t sourceImbalance = sourcePart0Elements > sourcePart1Elements ?
                sourcePart0Elements - sourcePart1Elements : sourcePart1Elements - sourcePart0Elements;
            const uint64_t outputImbalance = outputPart0Elements > outputPart1Elements ?
                outputPart0Elements - outputPart1Elements : outputPart1Elements - outputPart0Elements;
            // MTE2 仍是主瓶颈，优先均衡连续搬入量；输出量用于打破同等候选。
            const uint64_t balance = SaturatingAdd(SaturatingMul(sourceImbalance, 2), outputImbalance);
            if (balance < bestBalance) {
                bestBalance = balance;
                bestSplit = input + 1;
            }
        }
        if (bestSplit == 0 || bestSplit > std::numeric_limits<uint16_t>::max()) {
            return false;
        }

        uint64_t selectedSourcePart0 = 0;
        uint64_t selectedOutputPart0 = 0;
        for (uint32_t input = inputStart; input < bestSplit; ++input) {
            const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
            selectedOutputPart0 = SaturatingAdd(selectedOutputPart0, inputElements);
            selectedSourcePart0 = SaturatingAdd(selectedSourcePart0, AlignUp(
                SaturatingMul(baseline.sourceBatchRows, inputElements), GATHER_ELEMENT_ALIGNMENT));
        }
        const uint64_t selectedSourcePart1 = totalSourceElements - selectedSourcePart0;
        const uint64_t selectedOutputPart1 = totalOutputElements - selectedOutputPart0;
        if (selectedSourcePart0 == 0 || selectedSourcePart1 == 0 ||
            selectedOutputPart0 == 0 || selectedOutputPart1 == 0) {
            return false;
        }
        plan.groupInputSplit.push_back(static_cast<uint16_t>(bestSplit));
        maxSourcePart0Bytes = std::max(maxSourcePart0Bytes,
            SaturatingMul(selectedSourcePart0, GATHER_DTYPE_BYTES));
        maxSourcePart1Bytes = std::max(maxSourcePart1Bytes,
            SaturatingMul(selectedSourcePart1, GATHER_DTYPE_BYTES));
        maxOutputPart0Bytes = std::max(maxOutputPart0Bytes, SaturatingMul(
            SaturatingMul(baseline.batchRows,
                AlignUp(selectedOutputPart0, GATHER_ELEMENT_ALIGNMENT)), GATHER_DTYPE_BYTES));
        maxOutputPart1Bytes = std::max(maxOutputPart1Bytes, SaturatingMul(
            SaturatingMul(baseline.batchRows,
                AlignUp(selectedOutputPart1, GATHER_ELEMENT_ALIGNMENT)), GATHER_DTYPE_BYTES));
        inputStart = inputEnd;
    }

    const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(
        SaturatingAdd(maxSourcePart0Bytes, maxSourcePart1Bytes), SaturatingAdd(
            SaturatingMul(2, maxOutputPart0Bytes), SaturatingMul(2, maxOutputPart1Bytes))));
    if (inputStart != inputs.size() || maxSourcePart0Bytes == 0 || maxSourcePart1Bytes == 0 ||
        maxOutputPart0Bytes == 0 || maxOutputPart1Bytes == 0 || requiredBytes > ubSize ||
        maxSourcePart0Bytes > std::numeric_limits<uint32_t>::max() ||
        maxSourcePart1Bytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputPart0Bytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputPart1Bytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    plan.sourcePart0BufferBytes = static_cast<uint32_t>(maxSourcePart0Bytes);
    plan.sourcePart1BufferBytes = static_cast<uint32_t>(maxSourcePart1Bytes);
    plan.outputPart0BufferBytes = static_cast<uint32_t>(maxOutputPart0Bytes);
    plan.outputPart1BufferBytes = static_cast<uint32_t>(maxOutputPart1Bytes);
    return true;
}

bool BuildPartitionedRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildPartitionedGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildPartitionedGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

// 将输入重新规划为两个连续分区。两个 source 槽依次搬入；每段先 Gather 到紧凑
// partial output，再用 32B 对齐的 UB Copy 汇入完整 output UB，最终只做一次整行
// MTE3。这样保留 GM 侧连续 source 搬入，同时让第二段 MTE2 与第一段 Gather/Copy
// 重叠；全部门槛由 UB、输入拓扑和 API 对齐约束推导。
bool BuildWavefrontGatherSubPlan(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& baseline, uint64_t ubSize, GatherSubPlan& plan)
{
    if (baseline.rowsPerCore == 0 || baseline.batchRows == 0 ||
        baseline.sourceBatchRows != baseline.batchRows ||
        baseline.batchRows > std::numeric_limits<uint8_t>::max() ||
        baseline.groupInputEnd.empty() ||
        baseline.groupInputEnd.size() != baseline.groupElements.size() ||
        baseline.groupInputEnd.size() != baseline.groupSourceElements.size()) {
        return false;
    }

    struct WavefrontEdge {
        uint16_t inputEnd{0};
        uint16_t inputSplit{0};
        uint32_t groupElements{0};
        uint32_t groupSourceElements{0};
        uint32_t fullOutputBytes{0};
        uint32_t sourceSlotBytes{0};
        uint32_t partialMatrixBytes{0};
        uint64_t score{std::numeric_limits<uint64_t>::max()};
    };
    struct WavefrontState {
        uint32_t groupCount{std::numeric_limits<uint32_t>::max()};
        uint64_t score{std::numeric_limits<uint64_t>::max()};
        uint32_t previousInput{0};
        WavefrontEdge edge{};
        bool valid{false};
    };

    const uint64_t availableUbBytes = ubSize - UB_RESERVE_BYTES;
    if (availableUbBytes < 5 * BLOCK_BYTES ||
        inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS) {
        return false;
    }

    auto scalePermille = [](uint64_t value, uint64_t permille) {
        return SaturatingAdd(SaturatingMul(value / 1000, permille),
            SaturatingMul(value % 1000, permille) / 1000);
    };
    auto planBytes = [](const GatherSubPlan& candidate) {
        return SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(
            SaturatingAdd(candidate.outputBufferBytes,
                SaturatingMul(2, candidate.sourcePart0BufferBytes)),
            SaturatingMul(2, candidate.outputPart0BufferBytes)));
    };
    auto buildWithCaps = [&](uint64_t fullOutputCap, uint64_t sourceSlotCap,
        uint64_t partialMatrixCap, GatherSubPlan& candidate) -> bool {
        std::vector<WavefrontState> states(inputs.size() + 1);
        states[0].valid = true;
        states[0].groupCount = 0;
        states[0].score = 0;

        for (size_t inputStart = 0; inputStart < inputs.size(); ++inputStart) {
            if (!states[inputStart].valid) {
                continue;
            }
            struct SplitPrefix {
                uint16_t input{0};
                uint64_t outputElements{0};
                uint64_t sourceElements{0};
            };
            std::vector<SplitPrefix> splits;
            uint64_t groupElements = 0;
            uint64_t groupSourceElements = 0;
            for (size_t inputEnd = inputStart; inputEnd < inputs.size(); ++inputEnd) {
                if (inputs[inputEnd].rowBytes % GATHER_DTYPE_BYTES != 0) {
                    return false;
                }
                const uint64_t inputElements = inputs[inputEnd].rowBytes / GATHER_DTYPE_BYTES;
                const uint64_t inputSourceElements = AlignUp(SaturatingMul(
                    baseline.batchRows, inputElements), GATHER_ELEMENT_ALIGNMENT);
                groupElements = SaturatingAdd(groupElements, inputElements);
                groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
                const uint64_t fullOutputBytes = SaturatingMul(SaturatingMul(baseline.batchRows,
                    AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT)), GATHER_DTYPE_BYTES);
                if (fullOutputBytes > fullOutputCap) {
                    // 后续 group 只会继续增长；零宽输入不影响此结论。
                    break;
                }

                if (inputEnd > inputStart && groupElements != 0 &&
                    groupElements <= std::numeric_limits<uint32_t>::max() &&
                    groupSourceElements <= std::numeric_limits<uint32_t>::max() &&
                    inputEnd + 1 <= std::numeric_limits<uint16_t>::max()) {
                    WavefrontEdge bestEdge;
                    uint64_t bestSourceImbalance = std::numeric_limits<uint64_t>::max();
                    for (const SplitPrefix& split : splits) {
                        const uint64_t rightOutputElements = groupElements - split.outputElements;
                        const uint64_t rightSourceElements = groupSourceElements - split.sourceElements;
                        if (split.outputElements == 0 || rightOutputElements == 0 ||
                            split.sourceElements == 0 || rightSourceElements == 0) {
                            continue;
                        }
                        const uint64_t sourceSlotBytes = SaturatingMul(std::max(
                            split.sourceElements, rightSourceElements), GATHER_DTYPE_BYTES);
                        const uint64_t partialMatrixBytes = SaturatingMul(SaturatingMul(
                            baseline.batchRows, std::max(AlignUp(split.outputElements,
                            GATHER_ELEMENT_ALIGNMENT), AlignUp(rightOutputElements,
                            GATHER_ELEMENT_ALIGNMENT))), GATHER_DTYPE_BYTES);
                        if (sourceSlotBytes > sourceSlotCap ||
                            partialMatrixBytes > partialMatrixCap ||
                            sourceSlotBytes > std::numeric_limits<uint32_t>::max() ||
                            partialMatrixBytes > std::numeric_limits<uint32_t>::max() ||
                            fullOutputBytes > std::numeric_limits<uint32_t>::max()) {
                            continue;
                        }
                        const uint64_t sourceImbalance = split.sourceElements > rightSourceElements ?
                            split.sourceElements - rightSourceElements :
                            rightSourceElements - split.sourceElements;
                        const uint64_t edgeScore = SaturatingAdd(SaturatingMul(
                            partialMatrixBytes, 2), sourceSlotBytes);
                        if (edgeScore < bestEdge.score ||
                            (edgeScore == bestEdge.score &&
                             sourceImbalance < bestSourceImbalance)) {
                            bestEdge.inputEnd = static_cast<uint16_t>(inputEnd + 1);
                            bestEdge.inputSplit = split.input;
                            bestEdge.groupElements = static_cast<uint32_t>(groupElements);
                            bestEdge.groupSourceElements = static_cast<uint32_t>(groupSourceElements);
                            bestEdge.fullOutputBytes = static_cast<uint32_t>(fullOutputBytes);
                            bestEdge.sourceSlotBytes = static_cast<uint32_t>(sourceSlotBytes);
                            bestEdge.partialMatrixBytes = static_cast<uint32_t>(partialMatrixBytes);
                            bestEdge.score = edgeScore;
                            bestSourceImbalance = sourceImbalance;
                        }
                    }
                    if (bestEdge.score != std::numeric_limits<uint64_t>::max()) {
                        WavefrontState& next = states[inputEnd + 1];
                        const uint32_t groupCount = states[inputStart].groupCount + 1;
                        const uint64_t score = SaturatingAdd(states[inputStart].score,
                            bestEdge.score);
                        if (!next.valid || groupCount < next.groupCount ||
                            (groupCount == next.groupCount && score < next.score)) {
                            next.valid = true;
                            next.groupCount = groupCount;
                            next.score = score;
                            next.previousInput = static_cast<uint32_t>(inputStart);
                            next.edge = bestEdge;
                        }
                    }
                }

                // 只接受真实输入边界；两段 dense output 的起点均因此满足 Copy 的 32B 对齐。
                if (groupElements != 0 && groupElements % GATHER_ELEMENT_ALIGNMENT == 0 &&
                    inputEnd + 1 <= std::numeric_limits<uint16_t>::max()) {
                    SplitPrefix splitPrefix;
                    splitPrefix.input = static_cast<uint16_t>(inputEnd + 1);
                    splitPrefix.outputElements = groupElements;
                    splitPrefix.sourceElements = groupSourceElements;
                    splits.push_back(splitPrefix);
                }
            }
        }
        if (!states.back().valid || states.back().groupCount == 0 ||
            states.back().groupCount > optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }

        std::vector<WavefrontEdge> reversedEdges;
        for (uint32_t inputEnd = static_cast<uint32_t>(inputs.size()); inputEnd != 0;) {
            const WavefrontState& state = states[inputEnd];
            if (!state.valid || state.previousInput >= inputEnd ||
                state.edge.inputEnd != inputEnd) {
                return false;
            }
            reversedEdges.push_back(state.edge);
            inputEnd = state.previousInput;
        }
        std::reverse(reversedEdges.begin(), reversedEdges.end());

        candidate = baseline;
        candidate.groupInputEnd.clear();
        candidate.groupInputSplit.clear();
        candidate.groupInputSplit2.clear();
        candidate.groupElements.clear();
        candidate.groupSourceElements.clear();
        uint64_t maxFullOutputBytes = 0;
        uint64_t maxSourceSlotBytes = 0;
        uint64_t maxPartialMatrixBytes = 0;
        for (const WavefrontEdge& edge : reversedEdges) {
            if (edge.inputEnd == 0 || edge.inputSplit == 0 || edge.inputSplit >= edge.inputEnd ||
                edge.groupElements == 0 || edge.groupSourceElements == 0) {
                return false;
            }
            candidate.groupInputEnd.push_back(edge.inputEnd);
            candidate.groupInputSplit.push_back(edge.inputSplit);
            candidate.groupElements.push_back(edge.groupElements);
            candidate.groupSourceElements.push_back(edge.groupSourceElements);
            maxFullOutputBytes = std::max<uint64_t>(maxFullOutputBytes, edge.fullOutputBytes);
            maxSourceSlotBytes = std::max<uint64_t>(maxSourceSlotBytes, edge.sourceSlotBytes);
            maxPartialMatrixBytes = std::max<uint64_t>(maxPartialMatrixBytes,
                edge.partialMatrixBytes);
        }
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES, SaturatingAdd(
            SaturatingAdd(maxFullOutputBytes, SaturatingMul(2, maxSourceSlotBytes)),
            SaturatingMul(2, maxPartialMatrixBytes)));
        if (maxFullOutputBytes == 0 || maxSourceSlotBytes == 0 ||
            maxPartialMatrixBytes == 0 || requiredBytes > ubSize ||
            maxFullOutputBytes > std::numeric_limits<uint32_t>::max() ||
            maxSourceSlotBytes > std::numeric_limits<uint32_t>::max() ||
            maxPartialMatrixBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        // sourceBuffer/offsetBuffer 仅保持紧凑 TilingData 的基本完整性；key38 实际使用
        // 下方四个专用容量字段。
        candidate.sourceBufferBytes = static_cast<uint32_t>(maxSourceSlotBytes);
        candidate.outputBufferBytes = static_cast<uint32_t>(maxFullOutputBytes);
        candidate.offsetBufferBytes = static_cast<uint32_t>(maxPartialMatrixBytes);
        candidate.sourcePart0BufferBytes = static_cast<uint32_t>(maxSourceSlotBytes);
        candidate.sourcePart1BufferBytes = static_cast<uint32_t>(maxSourceSlotBytes);
        candidate.outputPart0BufferBytes = static_cast<uint32_t>(maxPartialMatrixBytes);
        candidate.outputPart1BufferBytes = static_cast<uint32_t>(maxPartialMatrixBytes);
        candidate.score = states.back().score;
        return true;
    };

    // 每个候选都是完整 UB 账本：一张完整 output、两个等长 source 槽和 partial
    // output/offset 两张矩阵。网格从资源比例推导，而不是来自任何公开 shape；DP 在
    // 每个账本下最小化动态 group 数，再按实际占用择优。
    constexpr std::array<uint16_t, 5> FULL_OUTPUT_PERMILLE = {220, 240, 270, 290, 320};
    constexpr std::array<uint16_t, 3> PARTIAL_MATRIX_PERMILLE = {170, 180, 200};
    GatherSubPlan best;
    uint64_t bestBytes = std::numeric_limits<uint64_t>::max();
    for (uint16_t fullPermille : FULL_OUTPUT_PERMILLE) {
        const uint64_t fullOutputCap = AlignDown(scalePermille(availableUbBytes, fullPermille),
            BLOCK_BYTES);
        for (uint16_t partialPermille : PARTIAL_MATRIX_PERMILLE) {
            const uint64_t partialMatrixCap = AlignDown(scalePermille(availableUbBytes,
                partialPermille), BLOCK_BYTES);
            const uint64_t partialBytes = SaturatingMul(2, partialMatrixCap);
            if (fullOutputCap == 0 || partialMatrixCap == 0 ||
                fullOutputCap >= availableUbBytes ||
                partialBytes >= availableUbBytes - fullOutputCap) {
                continue;
            }
            const uint64_t sourceSlotCap = AlignDown((availableUbBytes - fullOutputCap -
                partialBytes) / 2, BLOCK_BYTES);
            if (sourceSlotCap == 0) {
                continue;
            }
            GatherSubPlan candidate;
            if (!buildWithCaps(fullOutputCap, sourceSlotCap, partialMatrixCap, candidate)) {
                continue;
            }
            const uint64_t candidateBytes = planBytes(candidate);
            const bool fewerGroups = best.groupInputEnd.empty() ||
                candidate.groupInputEnd.size() < best.groupInputEnd.size();
            const bool smallerFootprint = candidate.groupInputEnd.size() ==
                best.groupInputEnd.size() && candidateBytes < bestBytes;
            const bool lowerScore = candidate.groupInputEnd.size() == best.groupInputEnd.size() &&
                candidateBytes == bestBytes && candidate.score < best.score;
            if (fewerGroups || smallerFootprint || lowerScore) {
                best = std::move(candidate);
                bestBytes = candidateBytes;
            }
        }
    }
    if (best.groupInputEnd.empty()) {
        return false;
    }
    plan = std::move(best);
    return true;
}

bool BuildWavefrontRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (!BuildWavefrontGatherSubPlan(inputs, baseline.subPlans[0], ubSize, lowPlan) ||
        !BuildWavefrontGatherSubPlan(inputs, baseline.subPlans[1], ubSize, highPlan)) {
        return false;
    }
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    return true;
}

// key39 复用 key38 的保守双 source 分组，但只保留一张 64-lane × batch 的 staged
// offset 表。Level-0 Gather 的 dstRepStride 以 32B block 表示，因而必须在 Host 侧
// 同时证明行 stride、repeat 和 staged 表容量合法，不能依赖 Kernel 运行时静默截断。
bool BuildDirectWavefrontRowGatherPlan(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& baseline, uint64_t ubSize, RowGatherPlan& plan)
{
    if (!BuildWavefrontRowGatherPlan(inputs, baseline, ubSize, plan)) {
        return false;
    }
    for (const GatherSubPlan& subPlan : plan.subPlans) {
        if (subPlan.rowsPerCore == 0 || subPlan.batchRows == 0 ||
            subPlan.batchRows > std::numeric_limits<uint8_t>::max() ||
            subPlan.sourceBatchRows != subPlan.batchRows ||
            subPlan.sourcePart0BufferBytes == 0 || subPlan.sourcePart1BufferBytes == 0 ||
            subPlan.outputBufferBytes == 0 || subPlan.outputPart1BufferBytes == 0 ||
            subPlan.groupInputEnd.empty() ||
            subPlan.groupInputEnd.size() != subPlan.groupInputSplit.size() ||
            subPlan.groupInputEnd.size() != subPlan.groupElements.size() ||
            subPlan.groupInputEnd.size() != subPlan.groupSourceElements.size()) {
            return false;
        }
        // 每个 64-lane Gather chunk 的各行 offset 之外，额外保留一行 stride，供 Vector
        // Add 从首行生成后续 repeat 的地址。该容量与输入维度无关，只由运行时 batch 推导。
        const uint64_t stagedOffsetBytes = SaturatingMul(SaturatingMul(
            SaturatingAdd(subPlan.batchRows, 1), MICRO_GATHER_VF), GATHER_DTYPE_BYTES);
        if (stagedOffsetBytes > subPlan.outputPart1BufferBytes) {
            return false;
        }
        uint32_t inputStart = 0;
        for (size_t group = 0; group < subPlan.groupInputEnd.size(); ++group) {
            const uint32_t inputEnd = subPlan.groupInputEnd[group];
            const uint32_t inputSplit = subPlan.groupInputSplit[group];
            const uint32_t groupElements = subPlan.groupElements[group];
            const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
            if (inputEnd <= inputStart || inputEnd > inputs.size() ||
                inputSplit <= inputStart || inputSplit >= inputEnd || groupElements == 0 ||
                alignedElements == 0 ||
                alignedElements / GATHER_ELEMENT_ALIGNMENT >
                    std::numeric_limits<uint16_t>::max()) {
                return false;
            }
            inputStart = inputEnd;
        }
        if (inputStart != inputs.size()) {
            return false;
        }
    }
    return true;
}

bool BuildMicroRegisterRowGatherPlan(const RowGatherPlan& baseline, uint64_t ubSize,
    RowGatherPlan& plan)
{
    // MicroAPI 每次从 UB 载入完整 64-lane 的 base/stride 向量；两张表均按向量宽度
    // 补齐，避免尾向量读取越过 offset buffer。仅在原 RowGather 的 UB 账本仍有余量
    // 时启用，其他场景继续走已验证的 key7。
    plan = baseline;
    for (GatherSubPlan& subPlan : plan.subPlans) {
        if (subPlan.rowsPerCore == 0 || subPlan.batchRows < 3 ||
            subPlan.groupElements.empty()) {
            return false;
        }
        uint64_t maxAligned = 0;
        for (uint32_t elements : subPlan.groupElements) {
            maxAligned = std::max(maxAligned, AlignUp(elements, GATHER_ELEMENT_ALIGNMENT));
        }
        const uint64_t paddedElements = AlignUp(maxAligned, MICRO_GATHER_VF);
        const uint64_t requiredOffsetBytes = SaturatingMul(
            SaturatingMul(2, paddedElements), GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(subPlan.sourceBufferBytes,
                SaturatingAdd(subPlan.outputBufferBytes, requiredOffsetBytes)));
        if (requiredOffsetBytes > std::numeric_limits<uint32_t>::max() ||
            requiredBytes > ubSize) {
            return false;
        }
        subPlan.offsetBufferBytes = std::max<uint32_t>(subPlan.offsetBufferBytes,
            static_cast<uint32_t>(requiredOffsetBytes));
    }
    return true;
}

bool BuildSourcePingPongRowGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim,
    uint64_t ubSize, uint32_t dtypeSize, RowGatherPlan& plan)
{
    // source 双槽与 output/offset 各一槽：MTE2 预取不会覆盖当前 Gather 的输入。
    return BuildRowGatherPlanWithBufferCounts(inputs, outer, totalBytes, outputRowBytes,
        blockDim, ubSize, dtypeSize, 2, 2, plan);
}

bool BuildBlockGatherGroups(const std::vector<InputInfo>& inputs, uint32_t batchRows,
    uint64_t ubSize, GatherSubPlan& plan)
{
    if (batchRows == 0 || batchRows > std::numeric_limits<uint8_t>::max()) {
        return false;
    }
    plan = GatherSubPlan{};
    plan.batchRows = batchRows;
    plan.sourceBatchRows = batchRows;
    uint64_t groupElements = 0;
    uint64_t sourceDataElements = 0;
    uint64_t maxSourceElements = 0;
    uint64_t maxOutputElements = 0;
    uint64_t maxOffsetElements = 0;
    size_t groupStart = 0;

    auto calculateStorage = [&](uint64_t elements, uint64_t sourceElements,
        uint64_t& rowBlocks, uint64_t& rowBlockStride, uint64_t& totalSourceElements,
        uint64_t& outputElements, uint64_t& offsetElements) -> bool {
        if (elements == 0) {
            return false;
        }
        rowBlocks = CeilDiv(elements, GATHER_ELEMENT_ALIGNMENT);
        rowBlockStride = AlignUp(rowBlocks, GATHERB_ROW_BLOCK_ALIGNMENT);
        if (rowBlocks == 0 || rowBlockStride == 0 ||
            rowBlockStride > std::numeric_limits<uint16_t>::max()) {
            return false;
        }
        const uint64_t stashElements = SaturatingMul(
            SaturatingMul(batchRows, rowBlockStride), GATHER_ELEMENT_ALIGNMENT);
        totalSourceElements = SaturatingAdd(sourceElements,
            SaturatingAdd(stashElements, GATHERB_SOURCE_GUARD_ELEMENTS));
        offsetElements = SaturatingMul(batchRows, rowBlockStride);
        outputElements = SaturatingMul(offsetElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(SaturatingMul(totalSourceElements, GATHER_DTYPE_BYTES),
                SaturatingAdd(SaturatingMul(outputElements, GATHER_DTYPE_BYTES),
                    SaturatingMul(offsetElements, GATHER_DTYPE_BYTES))));
        return requiredBytes <= ubSize && totalSourceElements <= std::numeric_limits<uint32_t>::max() &&
            outputElements <= std::numeric_limits<uint32_t>::max() &&
            offsetElements <= std::numeric_limits<uint32_t>::max();
    };

    auto saveGroup = [&](size_t inputEnd) -> bool {
        uint64_t rowBlocks = 0;
        uint64_t rowBlockStride = 0;
        uint64_t totalSourceElements = 0;
        uint64_t outputElements = 0;
        uint64_t offsetElements = 0;
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS ||
            !calculateStorage(groupElements, sourceDataElements, rowBlocks, rowBlockStride,
                totalSourceElements, outputElements, offsetElements)) {
            return false;
        }
        uint64_t outputPrefix = 0;
        uint64_t interiorBlocks = 0;
        uint64_t boundaryPieces = 0;
        uint32_t nonzeroInputs = 0;
        for (size_t input = groupStart; input < inputEnd; ++input) {
            const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
            if (inputElements != 0) {
                ++nonzeroInputs;
                const uint64_t firstInterior = CeilDiv(outputPrefix, GATHER_ELEMENT_ALIGNMENT);
                const uint64_t lastInterior = (outputPrefix + inputElements) /
                    GATHER_ELEMENT_ALIGNMENT;
                if (lastInterior > firstInterior) {
                    interiorBlocks = SaturatingAdd(interiorBlocks, lastInterior - firstInterior);
                }
                const uint64_t firstBlock = outputPrefix / GATHER_ELEMENT_ALIGNMENT;
                const uint64_t lastBlock = (outputPrefix + inputElements - 1) /
                    GATHER_ELEMENT_ALIGNMENT;
                const bool hasLeadingBoundary = firstBlock < firstInterior;
                const bool hasTrailingBoundary = lastBlock >= lastInterior;
                boundaryPieces = SaturatingAdd(boundaryPieces,
                    static_cast<uint64_t>(hasLeadingBoundary));
                if (hasTrailingBoundary && (!hasLeadingBoundary || lastBlock != firstBlock)) {
                    boundaryPieces = SaturatingAdd(boundaryPieces, 1);
                }
            }
            outputPrefix = SaturatingAdd(outputPrefix, inputElements);
        }
        if (outputPrefix != groupElements || nonzeroInputs == 0 ||
            interiorBlocks > rowBlocks || boundaryPieces > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(totalSourceElements));
        plan.totalBlocks = SaturatingAdd(plan.totalBlocks, rowBlocks);
        plan.interiorBlocks = SaturatingAdd(plan.interiorBlocks, interiorBlocks);
        plan.boundaryPieces = SaturatingAdd(plan.boundaryPieces, boundaryPieces);
        plan.minNonzeroInputsPerGroup = std::min(plan.minNonzeroInputsPerGroup, nonzeroInputs);
        maxSourceElements = std::max(maxSourceElements, totalSourceElements);
        maxOutputElements = std::max(maxOutputElements, outputElements);
        maxOffsetElements = std::max(maxOffsetElements, offsetElements);
        groupStart = inputEnd;
        return true;
    };

    for (size_t input = 0; input < inputs.size(); ++input) {
        if (inputs[input].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[input].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t phase = groupElements % GATHER_ELEMENT_ALIGNMENT;
        const uint64_t inputStripeElements = inputElements == 0 ? 0 : SaturatingMul(batchRows,
            AlignUp(SaturatingAdd(phase, inputElements), GATHER_ELEMENT_ALIGNMENT));
        uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        uint64_t candidateSourceElements = SaturatingAdd(sourceDataElements, inputStripeElements);
        uint64_t rowBlocks = 0;
        uint64_t rowBlockStride = 0;
        uint64_t totalSourceElements = 0;
        uint64_t outputElements = 0;
        uint64_t offsetElements = 0;
        if (groupElements != 0 && !calculateStorage(candidateElements, candidateSourceElements,
            rowBlocks, rowBlockStride, totalSourceElements, outputElements, offsetElements)) {
            if (!saveGroup(input)) {
                return false;
            }
            groupElements = 0;
            sourceDataElements = 0;
            candidateElements = inputElements;
            candidateSourceElements = inputElements == 0 ? 0 : SaturatingMul(batchRows,
                AlignUp(inputElements, GATHER_ELEMENT_ALIGNMENT));
        }
        if (!calculateStorage(candidateElements, candidateSourceElements, rowBlocks, rowBlockStride,
            totalSourceElements, outputElements, offsetElements)) {
            return false;
        }
        groupElements = candidateElements;
        sourceDataElements = candidateSourceElements;
    }
    if (!saveGroup(inputs.size()) || maxSourceElements == 0 || maxOutputElements == 0 ||
        maxOffsetElements == 0) {
        return false;
    }
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceElements * GATHER_DTYPE_BYTES);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputElements * GATHER_DTYPE_BYTES);
    plan.offsetBufferBytes = static_cast<uint32_t>(maxOffsetElements * GATHER_DTYPE_BYTES);
    // key19 只使用当前批大小；score 用于保留 planner 状态，不参与 key7 的常规选优。
    plan.score = SaturatingAdd(SaturatingMul(plan.totalBlocks, batchRows), plan.boundaryPieces);
    return true;
}

bool BuildBlockGatherRowPlan(const std::vector<InputInfo>& inputs, const RowGatherPlan& baseline,
    uint64_t ubSize, RowGatherPlan& plan)
{
    GatherSubPlan lowPlan;
    GatherSubPlan highPlan;
    if (baseline.subPlans[0].rowsPerCore == 0 || baseline.subPlans[1].rowsPerCore == 0 ||
        !BuildBlockGatherGroups(inputs, baseline.subPlans[0].batchRows, ubSize, lowPlan) ||
        !BuildBlockGatherGroups(inputs, baseline.subPlans[1].batchRows, ubSize, highPlan)) {
        return false;
    }
    lowPlan.rowsPerCore = baseline.subPlans[0].rowsPerCore;
    highPlan.rowsPerCore = baseline.subPlans[1].rowsPerCore;
    plan.highCoreCount = baseline.highCoreCount;
    plan.subPlans[0] = std::move(lowPlan);
    plan.subPlans[1] = std::move(highPlan);
    for (size_t index = 0; index < plan.subPlans.size(); ++index) {
        const GatherSubPlan& candidate = plan.subPlans[index];
        const GatherSubPlan& reference = baseline.subPlans[index];
        // 只有完整块占绝大多数、边界补块不主导且不明显扩大输出组数时，才交换 key7。
        if (candidate.groupInputEnd.empty() || candidate.minNonzeroInputsPerGroup < 2 ||
            candidate.totalBlocks == 0 ||
            SaturatingMul(candidate.interiorBlocks, 4) < SaturatingMul(candidate.totalBlocks, 3) ||
            SaturatingMul(candidate.boundaryPieces, 2) > candidate.interiorBlocks ||
            candidate.groupInputEnd.size() > reference.groupInputEnd.size() + 1) {
            return false;
        }
    }
    return true;
}

bool BuildResidentGatherGroups(const std::vector<InputInfo>& inputs, uint32_t rowsPerCore,
    uint32_t gatherRows, uint64_t ubSize, GatherSubPlan& plan)
{
    if (rowsPerCore == 0 || gatherRows == 0) {
        return false;
    }
    plan.groupInputEnd.clear();
    plan.groupElements.clear();
    plan.groupSourceElements.clear();
    uint64_t groupElements = 0;
    uint64_t groupSourceElements = 0;
    uint64_t sumAlignedElements = 0;
    uint64_t maxSourceBytes = 0;
    uint64_t maxOutputBytes = 0;
    uint64_t maxOffsetBytes = 0;

    auto saveGroup = [&](size_t inputEnd) -> bool {
        if (groupElements == 0 || inputEnd > std::numeric_limits<uint16_t>::max() ||
            groupElements > std::numeric_limits<uint32_t>::max() ||
            groupSourceElements > std::numeric_limits<uint32_t>::max() ||
            SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            plan.groupInputEnd.size() >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t sourceBytes = SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES);
        // source 覆盖整核行数以便跨批复用；output 只保留当前 Gather 批，
        // 避免把整组结果作为一次长跨行 DMA 写回。
        const uint64_t outputBytes = SaturatingMul(
            SaturatingMul(gatherRows, alignedElements), GATHER_DTYPE_BYTES);
        const uint64_t offsetBytes = SaturatingMul(
            SaturatingMul(gatherRows, alignedElements), GATHER_DTYPE_BYTES);
        plan.groupInputEnd.push_back(static_cast<uint16_t>(inputEnd));
        plan.groupElements.push_back(static_cast<uint32_t>(groupElements));
        plan.groupSourceElements.push_back(static_cast<uint32_t>(groupSourceElements));
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        maxSourceBytes = std::max(maxSourceBytes, sourceBytes);
        maxOutputBytes = std::max(maxOutputBytes, outputBytes);
        maxOffsetBytes = std::max(maxOffsetBytes, offsetBytes);
        return true;
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].rowBytes % GATHER_DTYPE_BYTES != 0) {
            return false;
        }
        const uint64_t inputElements = inputs[i].rowBytes / GATHER_DTYPE_BYTES;
        const uint64_t inputPayloadElements = SaturatingMul(rowsPerCore, inputElements);
        if (SaturatingMul(inputPayloadElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN) {
            return false;
        }
        const uint64_t inputSourceElements = AlignUp(inputPayloadElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateElements = SaturatingAdd(groupElements, inputElements);
        const uint64_t candidateSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t candidateAligned = AlignUp(candidateElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t candidateMatrixBytes = SaturatingMul(
            SaturatingMul(2, SaturatingMul(gatherRows, candidateAligned)),
            GATHER_DTYPE_BYTES);
        const uint64_t requiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(SaturatingMul(candidateSourceElements, GATHER_DTYPE_BYTES),
                candidateMatrixBytes));
        const bool candidateOutputTooLarge =
            SaturatingMul(candidateElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN;
        if (groupElements != 0 && (candidateOutputTooLarge || requiredBytes > ubSize)) {
            if (!saveGroup(i)) {
                return false;
            }
            groupElements = 0;
            groupSourceElements = 0;
        }

        groupElements = SaturatingAdd(groupElements, inputElements);
        groupSourceElements = SaturatingAdd(groupSourceElements, inputSourceElements);
        const uint64_t alignedElements = AlignUp(groupElements, GATHER_ELEMENT_ALIGNMENT);
        const uint64_t matrixBytes = SaturatingMul(
            SaturatingMul(2, SaturatingMul(gatherRows, alignedElements)),
            GATHER_DTYPE_BYTES);
        const uint64_t singleRequiredBytes = SaturatingAdd(UB_RESERVE_BYTES,
            SaturatingAdd(SaturatingMul(groupSourceElements, GATHER_DTYPE_BYTES), matrixBytes));
        if (SaturatingMul(groupElements, GATHER_DTYPE_BYTES) > DATA_COPY_MAX_BLOCK_LEN ||
            singleRequiredBytes > ubSize) {
            return false;
        }
    }
    if (!saveGroup(inputs.size())) {
        return false;
    }

    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    uint64_t gatherCalls = 0;
    for (uint32_t elements : plan.groupElements) {
        gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
            SaturatingMul(rowsPerCore, AlignUp(elements, GATHER_ELEMENT_ALIGNMENT)),
            GATHER_MAX_ELEMENTS_PER_CALL));
    }
    const uint64_t gatherBatches = CeilDiv(rowsPerCore, gatherRows);
    // 每个批次都要重建首行 offset；其余行只做相邻行偏移推进。
    const uint64_t offsetRows = SaturatingAdd(rowsPerCore, SaturatingMul(2, gatherBatches));
    const uint64_t offsetWork = SaturatingMul(
        SaturatingMul(sumAlignedElements, offsetRows), 3 * sizeof(int32_t));
    const uint64_t outputCalls = SaturatingMul(plan.groupInputEnd.size(), gatherBatches);
    const uint64_t commandCount = SaturatingAdd(nonzeroInputs,
        SaturatingAdd(outputCalls, gatherCalls));
    plan.score = SaturatingAdd(SaturatingMul(commandCount, DMA_CALL_EQUIVALENT_BYTES), offsetWork);

    // 三块 TBuf 是按各组最大值一次性分配，必须校验最大值组合而非单组组合。
    const uint64_t allocatedBytes = SaturatingAdd(UB_RESERVE_BYTES,
        SaturatingAdd(maxSourceBytes, SaturatingAdd(maxOutputBytes, maxOffsetBytes)));
    if (maxSourceBytes > std::numeric_limits<uint32_t>::max() ||
        maxOutputBytes > std::numeric_limits<uint32_t>::max() ||
        maxOffsetBytes > std::numeric_limits<uint32_t>::max() || allocatedBytes > ubSize) {
        return false;
    }
    plan.batchRows = gatherRows;
    plan.sourceBufferBytes = static_cast<uint32_t>(maxSourceBytes);
    plan.outputBufferBytes = static_cast<uint32_t>(maxOutputBytes);
    plan.offsetBufferBytes = static_cast<uint32_t>(maxOffsetBytes);
    return true;
}

bool BuildResidentGatherSubPlan(const std::vector<InputInfo>& inputs, uint32_t rowsPerCore,
    uint64_t ubSize, GatherSubPlan& best)
{
    best = GatherSubPlan{};
    best.rowsPerCore = rowsPerCore;
    for (uint32_t gatherRows = 1;
         gatherRows <= std::min<uint32_t>(rowsPerCore, DATA_COPY_MAX_ROWS); ++gatherRows) {
        GatherSubPlan candidate;
        candidate.rowsPerCore = rowsPerCore;
        if (BuildResidentGatherGroups(inputs, rowsPerCore, gatherRows, ubSize, candidate) &&
            candidate.score < best.score) {
            best = std::move(candidate);
        }
    }
    return best.score != std::numeric_limits<uint64_t>::max();
}

bool BuildResidentRowGatherPlan(const std::vector<InputInfo>& inputs, uint64_t outer,
    uint64_t totalBytes, uint64_t outputRowBytes, uint32_t blockDim,
    uint64_t ubSize, uint32_t dtypeSize, RowGatherPlan& plan)
{
    if (dtypeSize != GATHER_DTYPE_BYTES || totalBytes < GetGatherMinTotalBytes(ubSize) ||
        inputs.size() > optiling::CONCAT_MAX_GRID_INPUTS || blockDim == 0 || outer < blockDim ||
        outputRowBytes == 0 || outputRowBytes % GATHER_DTYPE_BYTES != 0 ||
        outputRowBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t balancedLowRows = static_cast<uint32_t>(outer / blockDim);
    const uint32_t highRows = static_cast<uint32_t>(CeilDiv(outer, blockDim));
    GatherSubPlan balancedLowPlan;
    GatherSubPlan highPlan;
    if (balancedLowRows == 0 || highRows > DATA_COPY_MAX_ROWS ||
        !BuildResidentGatherSubPlan(inputs, balancedLowRows, ubSize, balancedLowPlan) ||
        !BuildResidentGatherSubPlan(inputs, highRows, ubSize, highPlan)) {
        return false;
    }
    plan.subPlans[0] = balancedLowPlan;
    plan.subPlans[1] = highPlan;
    plan.highCoreCount = highRows == balancedLowRows ? 0 :
        static_cast<uint32_t>(outer % blockDim);

    auto coreCost = [&](const GatherSubPlan& subPlan) -> uint64_t {
        return SaturatingAdd(SaturatingMul(subPlan.rowsPerCore, outputRowBytes), subPlan.score);
    };
    uint64_t bestMaxCost = std::max(coreCost(plan.subPlans[0]), coreCost(plan.subPlans[1]));
    uint64_t bestTotalCost = SaturatingAdd(
        SaturatingMul(blockDim - plan.highCoreCount, coreCost(plan.subPlans[0])),
        SaturatingMul(plan.highCoreCount, coreCost(plan.subPlans[1])));
    if (highRows != balancedLowRows) {
        const uint32_t searchRows = static_cast<uint32_t>(
            CeilDiv(balancedLowRows, GATHER_ROW_SEARCH_DIVISOR));
        const uint32_t minLowRows = std::max<uint32_t>(1, balancedLowRows - searchRows);
        for (uint32_t lowRows = minLowRows; lowRows < balancedLowRows; ++lowRows) {
            const uint64_t rowDelta = highRows - lowRows;
            const uint64_t highRowsTotal = outer - SaturatingMul(blockDim, lowRows);
            if (rowDelta == 0 || highRowsTotal % rowDelta != 0) {
                continue;
            }
            const uint64_t highCoreCount64 = highRowsTotal / rowDelta;
            if (highCoreCount64 == 0 || highCoreCount64 >= blockDim) {
                continue;
            }
            GatherSubPlan lowPlan;
            if (!BuildResidentGatherSubPlan(inputs, lowRows, ubSize, lowPlan)) {
                continue;
            }
            const uint32_t highCoreCount = static_cast<uint32_t>(highCoreCount64);
            const uint64_t lowCost = coreCost(lowPlan);
            const uint64_t highCost = coreCost(highPlan);
            const uint64_t candidateMaxCost = std::max(lowCost, highCost);
            const uint64_t candidateTotalCost = SaturatingAdd(
                SaturatingMul(blockDim - highCoreCount, lowCost),
                SaturatingMul(highCoreCount, highCost));
            const bool improvesMax = candidateMaxCost < bestMaxCost;
            const bool keepsTotal = candidateTotalCost <= bestTotalCost;
            const bool improvesTie = candidateMaxCost == bestMaxCost &&
                candidateTotalCost < bestTotalCost;
            if ((improvesMax && keepsTotal) || improvesTie) {
                plan.subPlans[0] = std::move(lowPlan);
                plan.subPlans[1] = highPlan;
                plan.highCoreCount = highCoreCount;
                bestMaxCost = candidateMaxCost;
                bestTotalCost = candidateTotalCost;
            }
        }
    }
    return true;
}

uint64_t EstimateGatherExecutionCost(const std::vector<InputInfo>& inputs,
    const GatherSubPlan& subPlan, bool sourceResident)
{
    if (subPlan.rowsPerCore == 0 || subPlan.batchRows == 0 ||
        subPlan.groupInputEnd.empty()) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint32_t sourceBatchRows = subPlan.sourceBatchRows == 0 ? subPlan.batchRows :
        subPlan.sourceBatchRows;
    if (sourceBatchRows == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t nonzeroInputs = 0;
    for (const InputInfo& input : inputs) {
        nonzeroInputs += static_cast<uint64_t>(input.rowBytes != 0);
    }
    const uint64_t sourceFullBatches = subPlan.rowsPerCore / sourceBatchRows;
    const uint64_t sourceTailRows = subPlan.rowsPerCore % sourceBatchRows;
    const uint64_t sourceBatches = sourceFullBatches +
        static_cast<uint64_t>(sourceTailRows != 0);
    const uint64_t outputFullBatches = subPlan.rowsPerCore / subPlan.batchRows;
    const uint64_t outputTailRows = subPlan.rowsPerCore % subPlan.batchRows;
    const uint64_t outputBatches = outputFullBatches +
        static_cast<uint64_t>(outputTailRows != 0);
    uint64_t gatherCalls = 0;
    uint64_t sumAlignedElements = 0;
    for (uint32_t elements : subPlan.groupElements) {
        const uint64_t alignedElements = AlignUp(elements, GATHER_ELEMENT_ALIGNMENT);
        sumAlignedElements = SaturatingAdd(sumAlignedElements, alignedElements);
        if (sourceResident) {
            gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
                SaturatingMul(subPlan.rowsPerCore, alignedElements),
                GATHER_MAX_ELEMENTS_PER_CALL));
        } else {
            const uint64_t fullBatchCalls = CeilDiv(
                SaturatingMul(subPlan.batchRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL);
            gatherCalls = SaturatingAdd(gatherCalls,
                SaturatingMul(outputFullBatches, fullBatchCalls));
            if (outputTailRows != 0) {
                gatherCalls = SaturatingAdd(gatherCalls, CeilDiv(
                    SaturatingMul(outputTailRows, alignedElements), GATHER_MAX_ELEMENTS_PER_CALL));
            }
        }
    }
    const uint64_t inputCalls = sourceResident ? nonzeroInputs :
        SaturatingMul(nonzeroInputs, sourceBatches);
    // 即使 source 常驻，每个批次仍需独立沿行 stride 写回输出。
    const uint64_t outputCalls = SaturatingMul(subPlan.groupInputEnd.size(), outputBatches);
    const bool sourceStaged = !sourceResident && sourceBatchRows > subPlan.batchRows;
    const uint64_t offsetRows = sourceStaged ?
        SaturatingMul(outputBatches, SaturatingAdd(subPlan.batchRows, 2)) :
        SaturatingAdd(subPlan.rowsPerCore, SaturatingMul(2, outputBatches));
    const uint64_t offsetWork = SaturatingMul(
        SaturatingMul(sumAlignedElements, offsetRows), 3 * sizeof(int32_t));
    return SaturatingAdd(SaturatingMul(
        SaturatingAdd(inputCalls, SaturatingAdd(outputCalls, gatherCalls)),
        DMA_CALL_EQUIVALENT_BYTES), offsetWork);
}

struct GatherPlanExecutionCost {
    uint64_t maxCoreCost{std::numeric_limits<uint64_t>::max()};
    uint64_t totalCost{std::numeric_limits<uint64_t>::max()};
};

GatherPlanExecutionCost EstimateRowGatherExecutionCost(const std::vector<InputInfo>& inputs,
    const RowGatherPlan& plan, uint32_t blockDim, uint64_t outputRowBytes, bool sourceResident)
{
    GatherPlanExecutionCost cost;
    if (blockDim == 0 || plan.highCoreCount > blockDim) {
        return cost;
    }
    const uint64_t lowCost = SaturatingAdd(
        SaturatingMul(plan.subPlans[0].rowsPerCore, outputRowBytes),
        EstimateGatherExecutionCost(inputs, plan.subPlans[0], sourceResident));
    const uint64_t highCost = SaturatingAdd(
        SaturatingMul(plan.subPlans[1].rowsPerCore, outputRowBytes),
        EstimateGatherExecutionCost(inputs, plan.subPlans[1], sourceResident));
    cost.maxCoreCost = std::max(lowCost, highCost);
    cost.totalCost = SaturatingAdd(
        SaturatingMul(blockDim - plan.highCoreCount, lowCost),
        SaturatingMul(plan.highCoreCount, highCost));
    return cost;
}

bool BuildPreparedRectCommands(const std::vector<InputInfo>& inputs, const Plan& plan,
    uint64_t outer, uint64_t outputRowBytes, bool bypassInputL2,
    std::array<uint64_t, optiling::CONCAT_MAX_PREPARED_RECT_WORDS>& commandData)
{
    commandData.fill(0);
    if (plan.tileBytes == 0 || plan.rowGroups == 0 || plan.colGroups == 0 ||
        plan.blockDim != plan.rowGroups * plan.colGroups ||
        plan.blockDim > optiling::CONCAT_MAX_GRID_COLS) {
        return false;
    }

    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        const uint32_t rowGroup = core / plan.colGroups;
        const uint32_t colGroup = core % plan.colGroups;
        const uint64_t baseRows = outer / plan.rowGroups;
        const uint64_t extraRows = outer % plan.rowGroups;
        const uint64_t rowStart = rowGroup * baseRows + std::min<uint64_t>(rowGroup, extraRows);
        const uint64_t rowCount = baseRows + static_cast<uint64_t>(rowGroup < extraRows);
        const uint64_t colStart = SaturatingMul(colGroup, plan.colChunkBytes);
        const uint64_t colEnd = std::min(outputRowBytes,
            SaturatingAdd(colStart, plan.colChunkBytes));
        if (rowCount == 0 || colStart >= colEnd) {
            return false;
        }

        uint32_t localCommandCount = 0;
        uint64_t inputColStart = 0;
        for (const InputInfo& input : inputs) {
            const uint64_t inputColEnd = SaturatingAdd(inputColStart, input.rowBytes);
            const uint64_t copyStart = std::max(colStart, inputColStart);
            const uint64_t copyEnd = std::min(colEnd, inputColEnd);
            if (copyStart < copyEnd) {
                const uint64_t srcCol = copyStart - inputColStart;
                uint64_t colOffset = 0;
                while (colOffset < copyEnd - copyStart) {
                    const uint64_t blockLen = std::min<uint64_t>(
                        std::min<uint64_t>(copyEnd - copyStart - colOffset, plan.tileBytes),
                        DATA_COPY_MAX_BLOCK_LEN);
                    const uint64_t localRowBytes = AlignUp(blockLen, BLOCK_BYTES);
                    uint64_t rowsPerTile = localRowBytes == 0 ? 0 : plan.tileBytes / localRowBytes;
                    rowsPerTile = std::min<uint64_t>(rowsPerTile, DATA_COPY_MAX_ROWS);
                    const uint64_t srcStride = input.rowBytes - blockLen;
                    const uint64_t dstStride = outputRowBytes - blockLen;
                    if (srcStride > std::numeric_limits<uint32_t>::max() ||
                        dstStride > std::numeric_limits<uint32_t>::max()) {
                        rowsPerTile = 1;
                    }
                    if (blockLen == 0 || rowsPerTile == 0 ||
                        blockLen > std::numeric_limits<uint32_t>::max()) {
                        return false;
                    }

                    uint64_t rowOffset = 0;
                    while (rowOffset < rowCount) {
                        if (localCommandCount >=
                            optiling::CONCAT_PREPARED_RECT_COMMANDS_PER_CORE) {
                            return false;
                        }
                        const uint64_t blockCount = std::min(rowCount - rowOffset, rowsPerTile);
                        const uint32_t command =
                            core * optiling::CONCAT_PREPARED_RECT_COMMANDS_PER_CORE +
                            localCommandCount;
                        const uint32_t word =
                            command * optiling::CONCAT_PREPARED_RECT_WORDS_PER_COMMAND;
                        commandData[word] = static_cast<uint64_t>(input.index) |
                            (blockCount << 16) | (blockLen << 32) |
                            (bypassInputL2 ? PREPARED_RECT_INPUT_L2_BYPASS_FLAG : 0);
                        commandData[word + 1] =
                            (rowStart + rowOffset) * input.rowBytes + srcCol + colOffset;
                        commandData[word + 2] =
                            (rowStart + rowOffset) * outputRowBytes + copyStart + colOffset;
                        const uint64_t packedSrcStride = blockCount == 1 ? 0 : srcStride;
                        const uint64_t packedDstStride = blockCount == 1 ? 0 : dstStride;
                        commandData[word + 3] = packedSrcStride | (packedDstStride << 32);
                        ++localCommandCount;
                        rowOffset += blockCount;
                    }
                    colOffset += blockLen;
                }
            }
            inputColStart = inputColEnd;
            if (inputColStart >= colEnd) {
                break;
            }
        }
        if (localCommandCount < 2 || localCommandCount >
            optiling::CONCAT_PREPARED_RECT_COMMANDS_PER_CORE) {
            return false;
        }
    }
    return true;
}

Plan BuildGridPlan(uint64_t outer, uint64_t outputRowBytes, uint32_t maxCores,
    uint32_t maxColGroups, uint32_t tileBytes, const std::vector<InputInfo>& inputs)
{
    Plan best;
    best.kind = PlanKind::GRID;
    best.key = optiling::CONCAT_KEY_GRID;
    if (tileBytes == 0) {
        return best;
    }
    for (uint32_t cols = 2; cols <= std::min(maxCores, maxColGroups); ++cols) {
        const uint32_t rows = static_cast<uint32_t>(std::min<uint64_t>(outer, maxCores / cols));
        if (rows == 0) {
            continue;
        }
        const uint64_t colChunk = AlignUp(CeilDiv(outputRowBytes, cols), GRID_COLUMN_ALIGNMENT);
        const uint32_t actualCols = static_cast<uint32_t>(CeilDiv(outputRowBytes, colChunk));
        const uint64_t active64 = SaturatingMul(rows, actualCols);
        if (actualCols < 2 || active64 > maxCores) {
            continue;
        }
        const uint32_t active = static_cast<uint32_t>(active64);
        const uint64_t maxRowsPerCore = CeilDiv(outer, rows);
        uint64_t maxCoreCalls = 0;
        uint64_t totalCoreCalls = 0;
        uint64_t inputBegin = 0;
        for (uint32_t col = 0; col < actualCols; ++col) {
            const uint64_t colBegin = SaturatingMul(col, colChunk);
            const uint64_t colEnd = std::min(outputRowBytes, SaturatingAdd(colBegin, colChunk));
            uint64_t calls = 0;
            inputBegin = 0;
            for (const InputInfo& input : inputs) {
                const uint64_t inputEnd = SaturatingAdd(inputBegin, input.rowBytes);
                const uint64_t begin = std::max(inputBegin, colBegin);
                const uint64_t end = std::min(inputEnd, colEnd);
                if (end > begin) {
                    calls = SaturatingAdd(calls,
                        EstimateRectCalls(end - begin, maxRowsPerCore, tileBytes));
                }
                inputBegin = inputEnd;
            }
            maxCoreCalls = std::max(maxCoreCalls, calls);
            totalCoreCalls = SaturatingAdd(totalCoreCalls, calls);
        }
        const uint64_t maxCoreBytes = SaturatingMul(maxRowsPerCore, colChunk);
        const uint64_t cost = SaturatingAdd(maxCoreBytes,
            SaturatingAdd(SaturatingMul(maxCoreCalls, DMA_CALL_EQUIVALENT_BYTES),
                SaturatingMul(totalCoreCalls, GRID_TOTAL_CALL_EQUIVALENT_BYTES)));
        if (cost < best.cost) {
            best.cost = cost;
            best.blockDim = active;
            best.rowGroups = rows;
            best.colGroups = actualCols;
            best.tileBytes = tileBytes;
            best.colChunkBytes = colChunk;
            best.maxCoreCalls = maxCoreCalls;
            best.totalCoreCalls = totalCoreCalls;
        }
    }
    return best;
}

bool BuildGridSpans(const std::vector<InputInfo>& inputs, const Plan& plan,
    uint64_t outputRowBytes, uint32_t inputCount,
    std::array<uint64_t, optiling::CONCAT_MAX_GRID_COLS>& coreSpan,
    std::array<uint64_t, optiling::CONCAT_MAX_GRID_INPUTS>& inputRowBytes)
{
    coreSpan.fill(0);
    inputRowBytes.fill(0);
    if (plan.colGroups == 0 || plan.colGroups > optiling::CONCAT_MAX_GRID_COLS ||
        inputCount > optiling::CONCAT_MAX_GRID_INPUTS ||
        outputRowBytes >= GRID_SPAN_OFFSET_LIMIT) {
        return false;
    }
    for (const InputInfo& input : inputs) {
        if (input.index >= optiling::CONCAT_MAX_GRID_INPUTS) {
            return false;
        }
        inputRowBytes[input.index] = input.rowBytes;
    }

    size_t inputCursor = 0;
    uint64_t inputBegin = 0;
    for (uint32_t col = 0; col < plan.colGroups; ++col) {
        const uint64_t colStart = SaturatingMul(col, plan.colChunkBytes);
        if (colStart >= outputRowBytes) {
            return false;
        }
        while (inputCursor < inputs.size()) {
            uint64_t inputEnd = 0;
            if (AddOverflow(inputBegin, inputs[inputCursor].rowBytes, inputEnd)) {
                return false;
            }
            if (inputEnd > colStart) {
                break;
            }
            inputBegin = inputEnd;
            ++inputCursor;
        }
        if (inputCursor >= inputs.size() || inputBegin >= GRID_SPAN_OFFSET_LIMIT ||
            inputs[inputCursor].index > std::numeric_limits<uint16_t>::max()) {
            return false;
        }
        coreSpan[col] = (inputBegin << 16) | inputs[inputCursor].index;
    }
    return true;
}

uint64_t TaskWeight(uint64_t rowBytes, uint64_t rowCount, uint64_t tileBytes)
{
    const uint64_t bytes = SaturatingMul(rowBytes, rowCount);
    const uint64_t calls = EstimateRectCalls(rowBytes, rowCount, tileBytes);
    return SaturatingAdd(bytes, SaturatingMul(calls, DMA_CALL_EQUIVALENT_BYTES));
}

bool BuildLptTasks(const std::vector<InputInfo>& inputs, uint64_t outer, uint64_t tileBytes,
    uint32_t maxCores, std::vector<LptTask>& tasks, uint64_t& maxCoreLoad)
{
    if (outer > std::numeric_limits<uint32_t>::max() || inputs.size() > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    for (const InputInfo& input : inputs) {
        if (input.rowBytes == 0) {
            continue;
        }
        if (tasks.size() == optiling::CONCAT_MAX_LPT_TASKS) {
            return false;
        }
        LptTask task;
        task.input = static_cast<uint16_t>(input.index);
        task.rowCount = static_cast<uint32_t>(outer);
        task.weight = TaskWeight(input.rowBytes, outer, tileBytes);
        tasks.push_back(task);
    }
    if (tasks.empty()) {
        return false;
    }

    // 每个有效核最多承担一个 LPT 任务，避免把同一核的连续行拆成两次独立
    // MTE2/MTE3 搬运；输入数不足时仍保留每个输入至少一个任务。
    const size_t targetTasks = std::min<size_t>(optiling::CONCAT_MAX_LPT_TASKS,
        std::max<size_t>(tasks.size(), static_cast<size_t>(maxCores) * 2));
    while (tasks.size() < targetTasks) {
        auto splitIt = std::max_element(tasks.begin(), tasks.end(), [](const LptTask& lhs, const LptTask& rhs) {
            const uint64_t lhsWeight = lhs.rowCount > 1 ? lhs.weight : 0;
            const uint64_t rhsWeight = rhs.rowCount > 1 ? rhs.weight : 0;
            return lhsWeight < rhsWeight;
        });
        if (splitIt == tasks.end() || splitIt->rowCount <= 1) {
            break;
        }
        const uint32_t firstRows = splitIt->rowCount / 2;
        const uint32_t secondRows = splitIt->rowCount - firstRows;
        const uint64_t unitWeight = splitIt->weight / splitIt->rowCount;
        LptTask second = *splitIt;
        second.rowStart += firstRows;
        second.rowCount = secondRows;
        second.weight = SaturatingMul(unitWeight, secondRows);
        splitIt->rowCount = firstRows;
        splitIt->weight = SaturatingMul(unitWeight, firstRows);
        tasks.push_back(second);
    }

    std::sort(tasks.begin(), tasks.end(), [](const LptTask& lhs, const LptTask& rhs) {
        return lhs.weight > rhs.weight;
    });
    std::vector<uint64_t> coreLoads(maxCores, 0);
    for (LptTask& task : tasks) {
        const auto minIt = std::min_element(coreLoads.begin(), coreLoads.end());
        const uint32_t core = static_cast<uint32_t>(minIt - coreLoads.begin());
        task.core = static_cast<uint16_t>(core);
        coreLoads[core] = SaturatingAdd(coreLoads[core], task.weight);
    }
    std::stable_sort(tasks.begin(), tasks.end(), [](const LptTask& lhs, const LptTask& rhs) {
        return lhs.core < rhs.core;
    });
    maxCoreLoad = *std::max_element(coreLoads.begin(), coreLoads.end());
    return true;
}

Plan BuildLptPlan(const std::vector<InputInfo>& inputs, uint64_t outer, uint64_t totalBytes,
    uint32_t maxCores, uint64_t tileBytes)
{
    Plan plan;
    plan.kind = PlanKind::INPUT_LPT;
    plan.key = optiling::CONCAT_KEY_INPUT_LPT;
    uint64_t maxCoreLoad = 0;
    if (!BuildLptTasks(inputs, outer, tileBytes, maxCores, plan.tasks, maxCoreLoad)) {
        return plan;
    }
    plan.blockDim = static_cast<uint32_t>(std::min<size_t>(maxCores, plan.tasks.size()));
    plan.cost = SaturatingAdd(maxCoreLoad,
        CeilDiv(SaturatingMul(plan.tasks.size(), TASK_EQUIVALENT_BYTES), plan.blockDim));
    // 极小数据不应被 LPT 的任务表开销吸引。
    if (totalBytes < MIN_CORE_WORK_BYTES) {
        plan.cost = std::numeric_limits<uint64_t>::max();
    }
    return plan;
}

template <typename T>
ge::graphStatus SaveTiling(gert::TilingContext* context, T& tiling, size_t workspaceBytes = 0)
{
    auto* rawTiling = context->GetRawTilingData();
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr || workspaceSize == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    workspaceSize[0] = workspaceBytes;
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* computeNodeInfo = context->GetComputeNodeInfo();
    if (computeNodeInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* inputInstance = computeNodeInfo->GetInputInstanceInfo(0);
    if (inputInstance == nullptr || inputInstance->GetInstanceNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t inputCount = inputInstance->GetInstanceNum();
    const gert::StorageShape* firstStorage = context->GetDynamicInputShape(0, 0);
    const gert::CompileTimeTensorDesc* firstDesc = context->GetDynamicInputDesc(0, 0);
    const int64_t* dimAttr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (firstStorage == nullptr || firstDesc == nullptr || dimAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& firstShape = firstStorage->GetStorageShape();
    const size_t rank = firstShape.GetDimNum();
    int64_t dim = *dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (rank == 0 || rank > MAX_SUPPORTED_RANK || dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t dtypeSizeSigned = ge::GetSizeByDataType(firstDesc->GetDataType());
    if (dtypeSizeSigned != 1 && dtypeSizeSigned != 2 && dtypeSizeSigned != 4) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtypeSize = static_cast<uint32_t>(dtypeSizeSigned);

    uint64_t outer = 1;
    uint64_t innerElements = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t axisSize = firstShape.GetDim(axis);
        if (axisSize < 0) {
            return ge::GRAPH_FAILED;
        }
        if (axis < static_cast<size_t>(dim) && MulOverflow(outer, static_cast<uint64_t>(axisSize), outer)) {
            return ge::GRAPH_FAILED;
        }
        if (axis > static_cast<size_t>(dim) &&
            MulOverflow(innerElements, static_cast<uint64_t>(axisSize), innerElements)) {
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t innerBytes = 0;
    if (MulOverflow(innerElements, dtypeSize, innerBytes)) {
        return ge::GRAPH_FAILED;
    }

    std::vector<InputInfo> inputs;
    inputs.reserve(inputCount);
    uint64_t outputRowBytes = 0;
    uint64_t nonzeroInputs = 0;
    uint64_t maxRowBytes = 0;
    for (uint32_t i = 0; i < inputCount; ++i) {
        const gert::StorageShape* storage = context->GetDynamicInputShape(0, i);
        const gert::CompileTimeTensorDesc* desc = context->GetDynamicInputDesc(0, i);
        if (storage == nullptr || desc == nullptr || desc->GetDataType() != firstDesc->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& shape = storage->GetStorageShape();
        if (shape.GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t axisSize = shape.GetDim(axis);
            if (axisSize < 0 || (axis != static_cast<size_t>(dim) && axisSize != firstShape.GetDim(axis))) {
                return ge::GRAPH_FAILED;
            }
        }
        uint64_t rowBytes = 0;
        uint64_t totalBytes = 0;
        const uint64_t outputCol = outputRowBytes;
        if (MulOverflow(static_cast<uint64_t>(shape.GetDim(dim)), innerBytes, rowBytes) ||
            MulOverflow(rowBytes, outer, totalBytes) || AddOverflow(outputRowBytes, rowBytes, outputRowBytes)) {
            return ge::GRAPH_FAILED;
        }
        InputInfo input;
        input.index = i;
        input.rowBytes = rowBytes;
        input.outputCol = outputCol;
        inputs.push_back(input);
        if (rowBytes != 0) {
            ++nonzeroInputs;
            maxRowBytes = std::max(maxRowBytes, rowBytes);
        }
    }
    uint64_t totalBytes = 0;
    if (MulOverflow(outer, outputRowBytes, totalBytes)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t physicalCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (physicalCores == 0 || ubSize <= UB_RESERVE_BYTES + DOUBLE_BUFFER_NUM * BLOCK_BYTES) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t availableUbBytes = ubSize - UB_RESERVE_BYTES;
    const uint64_t tileBytes64 = AlignDown(availableUbBytes / DOUBLE_BUFFER_NUM, BLOCK_BYTES);
    if (tileBytes64 == 0 || tileBytes64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tileBytes = static_cast<uint32_t>(tileBytes64);
    const uint64_t singleTileBytes64 = AlignDown(availableUbBytes, BLOCK_BYTES);
    const uint32_t singleTileBytes = singleTileBytes64 <= std::numeric_limits<uint32_t>::max() ?
        static_cast<uint32_t>(singleTileBytes64) : 0;
    const uint64_t usefulCores = totalBytes == 0 ? 1 : CeilDiv(totalBytes, MIN_CORE_WORK_BYTES);
    const uint32_t maxCores = static_cast<uint32_t>(std::max<uint64_t>(1,
        std::min<uint64_t>(physicalCores, usefulCores)));

    Plan best = BuildRowPlan(outer, totalBytes, maxCores, tileBytes, inputs);
    if (outer == 1) {
        const Plan flat = BuildFlatPlan(totalBytes, maxCores, tileBytes);
        if (flat.cost <= best.cost) {
            best = flat;
        }
    }
    const bool rowParallelSaturated = outer >= maxCores;
    const uint64_t averageRowBytes = nonzeroInputs == 0 ? 0 : outputRowBytes / nonzeroInputs;
    const bool narrowRowsPreferLocality = rowParallelSaturated && averageRowBytes < GRID_COLUMN_ALIGNMENT;
    const uint32_t gridMaxColGroups = rowParallelSaturated ? 4 : maxCores;
    const uint64_t gridMinRowBytes = rowParallelSaturated ? 2 * GRID_COLUMN_ALIGNMENT : GRID_MIN_ROW_BYTES;
    std::array<uint64_t, CONCAT_MAX_GRID_COLS> singleGridCoreSpan{};
    std::array<uint64_t, CONCAT_MAX_GRID_INPUTS> singleGridInputRowBytes{};
    if (outer != 1 && outputRowBytes >= gridMinRowBytes && nonzeroInputs > 1 &&
        !narrowRowsPreferLocality) {
        const Plan doubleGrid = BuildGridPlan(outer, outputRowBytes, maxCores,
            gridMaxColGroups, tileBytes, inputs);
        if (doubleGrid.cost < best.cost) {
            best = doubleGrid;
        }

        Plan singleGrid = BuildGridPlan(outer, outputRowBytes, maxCores,
            gridMaxColGroups, singleTileBytes, inputs);
        singleGrid.key = CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER;
        const bool singleGridUseful = doubleGrid.cost != std::numeric_limits<uint64_t>::max() &&
            singleGrid.cost < doubleGrid.cost &&
            singleGrid.maxCoreCalls < doubleGrid.maxCoreCalls &&
            singleGrid.cost < best.cost;
        if (singleGridUseful && BuildGridSpans(inputs, singleGrid, outputRowBytes, inputCount,
            singleGridCoreSpan, singleGridInputRowBytes)) {
            // 单缓冲只有在完整 UB tile 确实减少最慢核 DMA 次数时才启用。
            best = std::move(singleGrid);
        }
    }
    const uint32_t lptMaxCores = std::min<uint32_t>(maxCores, CONCAT_MAX_LPT_CORES);
    const bool lptCandidate = averageRowBytes >= GRID_COLUMN_ALIGNMENT && outer > 1 &&
        (nonzeroInputs >= lptMaxCores ||
        (averageRowBytes != 0 && maxRowBytes > SaturatingMul(2, averageRowBytes)));
    if (lptCandidate) {
        const Plan lpt = BuildLptPlan(inputs, outer, totalBytes, lptMaxCores, tileBytes);
        if (lpt.cost < best.cost) {
            best = lpt;
        }
    }
    const bool rowGatherEligible = best.kind == PlanKind::ROW && narrowRowsPreferLocality &&
        nonzeroInputs > 1;
    RowGatherPlan rowGatherPlan;
    const bool rowGatherReady = rowGatherEligible &&
        BuildRowGatherPlan(inputs, outer, totalBytes, outputRowBytes,
            best.blockDim, ubSize, dtypeSize, rowGatherPlan);
    RowGatherPlan offsetOutputAliasRowGatherPlan;
    const bool offsetOutputAliasRowGatherReady = rowGatherReady &&
        BuildOffsetOutputAliasRowGatherPlan(inputs, outer, totalBytes, outputRowBytes,
            best.blockDim, ubSize, dtypeSize, offsetOutputAliasRowGatherPlan);
    RowGatherPlan sourcePingPongOffsetOutputAliasRowGatherPlan;
    const bool sourcePingPongOffsetOutputAliasRowGatherReady = rowGatherReady &&
        BuildSourcePingPongOffsetOutputAliasRowGatherPlan(inputs, outer, totalBytes,
            outputRowBytes, best.blockDim, ubSize, dtypeSize,
            sourcePingPongOffsetOutputAliasRowGatherPlan);
    RowGatherPlan adaptiveGroupBatchRowGatherPlan;
    // 仅复用 key7 已分配的三块 UB 最大容量。窄尾 group 可独立增大 batch，前序较宽
    // group 保持原批行数；Host 还要求 source 命令严格下降且 output/Gather 命令不增加。
    const bool adaptiveGroupBatchRowGatherReady = rowGatherReady &&
        BuildAdaptiveGroupBatchRowGatherPlan(inputs, rowGatherPlan,
            adaptiveGroupBatchRowGatherPlan);
    RowGatherPlan twoPlaneRowGatherPlan;
    // 两个等长 source 批由一次 blockCount=2 DMA 写入相隔 UB 平面。Host 同时核算两张
    // source、output、offset 的完整 UB 账本，以及尾批回退后的最慢核命令代价。
    const bool twoPlaneRowGatherReady = rowGatherReady &&
        BuildTwoPlaneRowGatherPlan(inputs, best.blockDim, rowGatherPlan, ubSize,
            twoPlaneRowGatherPlan);
    BandGatherPlan bandGatherPlan;
    // 二维带计划在 Host 侧同时核算最慢任务的输出量、MTE2、MTE3、Gather 和 UB；未达到
    // 动态收益门槛时严格回退到已验证的 key7。
    const bool bandGatherReady = rowGatherReady &&
        BuildBandGatherPlan(inputs, outer, outputRowBytes, best.blockDim, ubSize,
            rowGatherPlan, bandGatherPlan);
    PairGatherPlan pairGatherPlan;
    const bool pairGatherReady = rowGatherReady &&
        BuildPairGatherPlan(inputs, outer, outputRowBytes, best.blockDim, ubSize,
            rowGatherPlan, pairGatherPlan);
    RowGatherPlan suffixPrefetchRowGatherPlan;
    const bool suffixPrefetchRowGatherReady = rowGatherReady &&
        BuildSuffixPrefetchRowGatherPlan(inputs, rowGatherPlan, ubSize,
            suffixPrefetchRowGatherPlan);
    RowGatherPlan partitionedRowGatherPlan;
    const bool partitionedRowGatherReady = rowGatherReady &&
        BuildPartitionedRowGatherPlan(inputs, rowGatherPlan, ubSize, partitionedRowGatherPlan);
    RowGatherPlan wavefrontRowGatherPlan;
    const bool wavefrontRowGatherReady = rowGatherReady &&
        BuildWavefrontRowGatherPlan(inputs, rowGatherPlan, ubSize, wavefrontRowGatherPlan);
    RowGatherPlan directWavefrontRowGatherPlan;
    const bool directWavefrontRowGatherReady = rowGatherReady &&
        BuildDirectWavefrontRowGatherPlan(inputs, rowGatherPlan, ubSize,
            directWavefrontRowGatherPlan);
    RowGatherPlan maskedCopyRowPlan;
    const bool maskedCopyRowReady = rowGatherReady &&
        BuildMaskedCopyRowPlan(inputs, rowGatherPlan, ubSize, maskedCopyRowPlan);
    RowGatherPlan directMteRowGatherPlan;
    const bool directMteRowGatherReady = rowGatherReady &&
        BuildDirectMteRowGatherPlan(inputs, rowGatherPlan, ubSize, directMteRowGatherPlan);
    RowGatherPlan microRegisterRowGatherPlan;
    const bool microRegisterRowGatherReady = rowGatherReady &&
        BuildMicroRegisterRowGatherPlan(rowGatherPlan, ubSize, microRegisterRowGatherPlan);
    RowGatherPlan blockGatherRowPlan;
    const bool blockGatherRowReady = rowGatherReady &&
        BuildBlockGatherRowPlan(inputs, rowGatherPlan, ubSize, blockGatherRowPlan);
    RowGatherPlan mergedBatchRowGatherPlan;
    // 合并批次只在动态计划已证明减少 source DMA、且不增加 output/Gather 命令时启用。
    const bool mergedBatchRowGatherReady = rowGatherReady &&
        BuildMergedBatchRowGatherPlan(inputs, rowGatherPlan, ubSize, mergedBatchRowGatherPlan);
    RowGatherPlan sourcePingPongRowGatherPlan;
    const bool sourcePingPongRowGatherReady = rowGatherEligible &&
        BuildSourcePingPongRowGatherPlan(inputs, outer, totalBytes, outputRowBytes,
            best.blockDim, ubSize, dtypeSize, sourcePingPongRowGatherPlan);
    RowGatherPlan stagedSourceRowGatherPlan;
    const bool stagedSourceRowGatherReady = rowGatherReady &&
        BuildStagedSourceRowGatherPlan(inputs, rowGatherPlan, ubSize, stagedSourceRowGatherPlan);
    RowGatherPlan deltaStagedOffsetRowGatherPlan;
    const bool deltaStagedOffsetRowGatherReady = rowGatherReady &&
        BuildDeltaStagedOffsetRowGatherPlan(inputs, rowGatherPlan, ubSize,
            deltaStagedOffsetRowGatherPlan);
    RowGatherPlan fullStageOffsetRowGatherPlan;
    const bool fullStageOffsetRowGatherReady = rowGatherReady &&
        BuildFullStageOffsetRowGatherPlan(inputs, rowGatherPlan, ubSize,
            fullStageOffsetRowGatherPlan);
    RowGatherPlan cachedOffsetStagedRowGatherPlan;
    const bool cachedOffsetStagedRowGatherReady = rowGatherReady &&
        BuildCachedOffsetStagedSourceRowGatherPlan(inputs, rowGatherPlan, ubSize,
            cachedOffsetStagedRowGatherPlan);
    RowGatherPlan rowMajorStagedRowGatherPlan;
    const bool rowMajorStagedRowGatherReady = rowGatherReady &&
        BuildRowMajorStagedSourceRowGatherPlan(inputs, rowGatherPlan, ubSize,
            rowMajorStagedRowGatherPlan);
    uint64_t sharedSeedWorkspaceBytes = 0;
    // 每个 group 的首行 offset 与行增量只由动态输入拓扑和 subplan 决定；高扇入、
    // 满流水的 RowGather 才能摊销一次全核同步和 workspace seed 读取。条件不依赖
    // 公开 case、shape 或 split 内容。
    const bool sharedSeedTopology = rowGatherReady && nonzeroInputs >= 64 &&
        rowGatherPlan.subPlans[0].batchRows >= 2 &&
        rowGatherPlan.subPlans[1].batchRows >= 2 &&
        rowGatherPlan.subPlans[0].sourceBatchRows == rowGatherPlan.subPlans[0].batchRows &&
        rowGatherPlan.subPlans[1].sourceBatchRows == rowGatherPlan.subPlans[1].batchRows &&
        rowGatherPlan.subPlans[0].rowsPerCore >= 2 * rowGatherPlan.subPlans[0].batchRows &&
        rowGatherPlan.subPlans[1].rowsPerCore >= 2 * rowGatherPlan.subPlans[1].batchRows;
    const bool sharedSeedRowGatherReady = sharedSeedTopology &&
        BuildSharedSeedWorkspace(rowGatherPlan, best.blockDim, totalBytes, sharedSeedWorkspaceBytes);
    // C5 上的整组与流式常驻输出均稳定回退，保留实现供后续重构复用但不参与调度。
    const bool useResidentRowGather = false;
    // key43 将 offset 表的存储复用为 Gather 输出，仅保留一行 base 和一行 stride。
    // 高扇入门槛对应 64-lane Vector 粒度；候选还必须由动态计划证明连续 source DMA 严格
    // 下降、output/Gather 命令不增加，以及至少两批输出可摊销 seed 构造。
    const bool offsetOutputAliasTopology = offsetOutputAliasRowGatherReady && nonzeroInputs >= 64 &&
        IsOffsetOutputAliasRowGatherPlanUseful(inputs, rowGatherPlan,
            offsetOutputAliasRowGatherPlan);
    // key43 的 C5 A/B/A/B 没有形成超出 run 间漂移的收益；保留实现和动态资源账本供
    // 后续出现新搬运原语时对照，当前统一回退到已验证的 key7。
    const bool useOffsetOutputAliasRowGather = false;
    (void)offsetOutputAliasTopology;
    // key45 将 key43 省出的完整矩阵槽换为第二个 source 槽。只有高扇入、至少两个
    // batch，且 Host 证明不缩 batch、不增任何动态 MTE/Gather 命令时才启用；所有条件
    // 均来自运行时输入拓扑和 UB 账本。
    const bool sourcePingPongOffsetOutputAliasTopology =
        sourcePingPongOffsetOutputAliasRowGatherReady && nonzeroInputs >= 64 &&
        IsSourcePingPongOffsetOutputAliasRowGatherPlanUseful(inputs, rowGatherPlan,
            sourcePingPongOffsetOutputAliasRowGatherPlan);
    // 两次 C5 全核实测均在该数据流的 Vector 路径触发 UB 越界：第一次为独立双 source
    // TBuf，第二次将其改为单个连续 TBuf 后仍复现。保留动态资源账本和实现供后续从
    // 不同数据流重构时对照，但当前必须回退到已验证的 key7，不能让生产调度命中 key45。
    const bool useSourcePingPongOffsetOutputAliasRowGather = false;
    (void)sourcePingPongOffsetOutputAliasTopology;
    // key42 没有扩大 key7 的任一个全局 batch：仅为资源账本比最宽 group
    // 更小的 group 提高行数，以减少连续窄段 MTE2 发射。C5 的 B/A/B/A
    // 相邻采样方向不一致，不能证明稳定收益，当前回退到已验证的 key7。
    const bool useAdaptiveGroupBatchRowGather = false;
    // C5 的独立 35 条采样表明，双平面虽减少输入 MTE2 调用，却稳定增加 Gather/MTE3
    // 关键路径；保留实现供后续数据流复用，当前统一回退到已验证的单平面 RowGather。
    const bool useTwoPlaneRowGather = false;
    // C5 的独立 35 条采样证明，相位 padding 会把连续 source 搬入退化为多行短 burst，
    // MTE2 回退远大于消除 offset/Gather 的收益。保留实现供不同数据流复用，但不参与调度。
    const bool useMaskedCopyRow = false;
    // Copy 指令要求 UB 基址 32B 对齐，任意段宽无法满足，当前路径不可调度。
    const bool useDirectRowGather = false;
    // key26 的二维 MTE 行写回无法表达当前非 32B 对齐的紧密 source 行布局；其
    // correctness 已被严格校验否定，保留实现作对照但不参与常规调度。
    const bool useDirectMteRowGather = false;
    // C5 实测表明拆分输出槽增加了 MTE3 发射，未形成有效重叠；保留 key 15 实现供后续
    // 数据流重构复用，但不让常规 RowGather 调度到该实验路径。
    const bool useOutputPingPongRowGather = false;
    // source 双槽只在原本已判定为窄段 RowGather 的动态拓扑中启用；其 UB 预算由独立
    // plan 校验，避免预取槽挤占 output/offset 后越界。
    // C5 的 source 双槽虽可重叠部分 Gather，却因 UB 挤压而缩小了 source 批次；保留 key16
    // 供后续“搬入批次与输出批次解耦”的数据流复用，不参与当前常规调度。
    const bool useSourcePingPongRowGather = false;
    // C5 的 35 条同进程采样证明，单纯扩批虽减少输入 DMA 发射，却使跨行输出 DMA
    // 显著变长；保留 key17 实现供后续数据流重构对照，不参与常规调度。
    const bool useMergedBatchRowGather = false;
    // key18 虽能减少 MTE2，但 C5 实测因每个 source stage 重建 offset 表而稳定增加
    // Vector 关键路径；保留实现供后续 offset 常驻数据流复用，当前不参与调度。
    const bool useStagedSourceRowGather = false;
    // key23 为完整 source stage 建一次 offset 表，再由全部输出子批按行切片复用。
    // 仅接受动态 UB 账本、MTE2 发射节省和 offset 工作量共同证明有净收益的计划。
    const bool useFullStageOffsetRowGather = false;
    // key24 保持 key18 的连续 source stage 与较少分组；source stage 的尾子批只生成
    // 实际参与 Gather 的 offset 行数，不再为未使用行重复执行 CreateVecIndex 和 Add。
    const bool useCompactStageOffsetRowGather = false;
    // key25 依赖 C220 不提供的 MicroAPI Vector scope；实现保留给受支持架构的后续验证，
    // 但 910B4 绝不调度到该 key。
    const bool useMicroRegisterRowGather = false;
    // key20 把每个窄输入改为多行短 burst，实测 MTE2 严重回退；保留实现作对照，
    // 不能再由常规路径调度。
    const bool useRowMajorStagedRowGather = false;
    // key21 的分半写回没有降低大 batch 的 MTE3 关键路径，保留实现作对照，不参与调度。
    const bool useSplitOutputRowGather = false;
    // key22 虽复用了 offset 表，却因额外矩阵缓冲挤压 UB 并拉长 Vector/MTE3 路径；
    // 保留实现作后续数据流对照，不参与常规调度。
    const bool useCachedOffsetStagedRowGather = false;
    // key27 的逐行逐段平移把减少的建表工作重新放大成 Scalar/Vector 热循环；严格
    // 正确但 C5 35 条样本回退，保留实现作反例而不参与调度。
    const bool useRollingStageOffsetRowGather = false;
    // key28 虽以向量 Add/Sub 取代 key27 的逐行更新，但 C5 的 35 条采样仍因更长的
    // Vector/MTE3 关键路径回退；保留实现供后续不同数据流复用，不参与常规调度。
    const bool useDeltaStagedOffsetRowGather = false;
    // key19 的 boundary stash 与块级 Gather 在高输入数窄段拓扑中稳定拉长 Vector 路径；
    // 保留实现供后续数据流重构对照，但不参与常规调度。
    const bool useBlockGatherRow = false;
    // key31 不复制整块 source/output，而是将每个 group 分为互补输入分区：一侧 Gather 后
    // 立即归还其 source 槽给下一批 MTE2，另一侧 Gather 同时继续。高扇入且至少有两个完整
    // batch 才能摊销额外的两次 MTE3/offset 表；门槛只表达流水深度和资源前提。
    const bool partitionedPipelineTopology = nonzeroInputs >= 64 &&
        rowGatherPlan.subPlans[0].batchRows != 0 && rowGatherPlan.subPlans[1].batchRows != 0 &&
        rowGatherPlan.subPlans[0].rowsPerCore >= 2 * rowGatherPlan.subPlans[0].batchRows &&
        rowGatherPlan.subPlans[1].rowsPerCore >= 2 * rowGatherPlan.subPlans[1].batchRows;
    // C220 的 35 条 C5 样本表明，显式互补分区没有改变 MTE2/Vector 的串行关键路径；
    // 保留 key31 实现作事件语义对照，但不进入常规调度。
    const bool usePartitionedRowGather = false;
    // key38 仍保持每核拥有完整输出行：两个输入连续分区分别以 dense Gather 重排，
    // 再经 32B 对齐 UB Copy 汇入完整 output。因此它不会重走 key31 的双 MTE3
    // 抽象；仅在高扇入、至少两个完整 batch 且动态 UB 账本/对齐 DP 都成立时启用。
    const bool wavefrontTopology = rowGatherReady && nonzeroInputs >= 64 &&
        rowGatherPlan.subPlans[0].rowsPerCore >= 2 * rowGatherPlan.subPlans[0].batchRows &&
        rowGatherPlan.subPlans[1].rowsPerCore >= 2 * rowGatherPlan.subPlans[1].batchRows;
    // key38 的 C5 同条件 35 条采样稳定将端到端时间拉长到约 243 us；保留实现和
    // 动态资源规划供后续不同重排路径复用，但不让该数据流进入常规调度。
    const bool useWavefrontRowGather = false;
    // key39 仍沿用 key38 的连续双分区，区别是 Level-0 Gather 直接跨行距写入完整
    // output，省去 partial output 的 UB Copy。门槛只表达高扇入、可形成波前的行数和
    // Host 已验证的 repeat/stride/UB 约束，不读取公开 shape、case 或分片内容。
    const bool directWavefrontTopology = directWavefrontRowGatherReady && nonzeroInputs >= 64 &&
        directWavefrontRowGatherPlan.subPlans[0].rowsPerCore >=
            2 * directWavefrontRowGatherPlan.subPlans[0].batchRows &&
        directWavefrontRowGatherPlan.subPlans[1].rowsPerCore >=
            2 * directWavefrontRowGatherPlan.subPlans[1].batchRows;
    // key39 的完整 35 条 C5 采样虽将 MTE2 降至约 51 us，却使每个 64-lane chunk
    // 重建跨行 offset 表，Vector/Scalar 控制路径升至约 254/278 us；严格正确但稳定回退。
    // 保留 Host/Kernel 实现供未来受支持的 GM 直写路径对照，不参与常规调度。
    const bool useDirectWavefrontRowGather = false;
    // key33 虽减少了 offset 建表的 Vector 时间，但 C5 的 A/B/A 显示全核同步和 seed DMA
    // 稳定拉长端到端耗时；保留实现作后续无同步数据流的对照，不参与常规调度。
    const bool useSharedSeedRowGather = false;
    // 保持每核逻辑行块连续，仅将偶/奇物理核交错映射到前/后半逻辑块；不改变 UB、
    // DMA 调用数或输入分组。该候选用于验证高扇入读写是否受物理核与 bank 对应关系影响。
    const bool useCoreSwizzleRowGather = false;
    // 高扇入且至少有多个动态 group 时，各核若锁步处理同一组输入会集中争用相同 GM 域。
    // C5 的 35 条采样表明仅错峰 group 顺序会同时拉长 MTE2/MTE3，保留 key35 实现作
    // 跨核时序对照，但不进入常规调度。
    const bool groupPhaseTopology = rowGatherReady && best.blockDim > 1 && nonzeroInputs >= 64 &&
        rowGatherPlan.subPlans[0].groupInputEnd.size() > 1 &&
        rowGatherPlan.subPlans[1].groupInputEnd.size() > 1;
    const bool useGroupPhaseRowGather = false;
    // key36 的逐列二维 MTE3 写回在高扇入场景稳定回退；保留 Host/Kernal 对照实现，
    // 但不允许常规调度命中，直到出现同时保持整行写回的新数据流。
    const bool useBandGatherRowGather = false;
    // key46 的配对计划尚未完成逐用例硬件验收，当前保持已验证的 RowGather 主路径。
    const bool usePairGather = false;
    // key47 只在既有 source 尾部可容纳至少一个连续 4KB 后缀、每核至少两批时启用。
    // 它不增加 source TBuf，后缀位置在 normal/mixed 布局间交替，前缀仍由当前批结束
    // 后补齐；未满足容量门槛的 group 继续由同一 key47 内的 key7 逻辑处理。
    const bool useSuffixPrefetchRowGather = false;
    // key44 的首次探针把未使用的 workspace 地址作为 AIV 执行前提，无法区分该地址
    // 是否为空与 AIV 编号是否正确。C220 的 GetBlockIdx 定义已明确为物理 block 与
    // subblock 的全局逻辑编号；本轮只验证该最小语义，条件完全由运行时行切分推导。
    // key44 已在 C5 严格逐元素校验中确认全量输出错误。混合任务的逻辑 AIV 编号与
    // 启动语义尚未被最小独立实验精确证明，因此生产调度必须保持已验证的 key7。
    const bool useMixedAivProbe = false;
    (void)sharedSeedRowGatherReady;
    (void)adaptiveGroupBatchRowGatherReady;
    (void)blockGatherRowReady;
    (void)maskedCopyRowReady;
    (void)directMteRowGatherReady;
    (void)mergedBatchRowGatherReady;
    (void)sourcePingPongRowGatherReady;
    (void)stagedSourceRowGatherReady;
    (void)deltaStagedOffsetRowGatherReady;
    (void)microRegisterRowGatherReady;
    (void)fullStageOffsetRowGatherReady;
    (void)cachedOffsetStagedRowGatherReady;
    (void)rowMajorStagedRowGatherReady;
    (void)partitionedRowGatherReady;
    (void)wavefrontRowGatherReady;
    (void)directWavefrontTopology;
    (void)bandGatherReady;
    (void)pairGatherReady;
    (void)suffixPrefetchRowGatherReady;
    (void)partitionedPipelineTopology;
    (void)groupPhaseTopology;
    (void)useDirectMteRowGather;
    const bool useRowGather = rowGatherReady || useBlockGatherRow || useSourcePingPongRowGather ||
        useSourcePingPongOffsetOutputAliasRowGather ||
        useOffsetOutputAliasRowGather || useAdaptiveGroupBatchRowGather || useMergedBatchRowGather ||
        useStagedSourceRowGather || useRowMajorStagedRowGather ||
        useSplitOutputRowGather || useCachedOffsetStagedRowGather ||
        useFullStageOffsetRowGather || useCompactStageOffsetRowGather ||
        useMicroRegisterRowGather || useDirectMteRowGather ||
        useRollingStageOffsetRowGather || useDeltaStagedOffsetRowGather || usePartitionedRowGather ||
        useWavefrontRowGather || useDirectWavefrontRowGather ||
        useSharedSeedRowGather || useCoreSwizzleRowGather || useGroupPhaseRowGather ||
        useBandGatherRowGather || useTwoPlaneRowGather || useSuffixPrefetchRowGather;
    const RowGatherPlan* selectedRowGatherPlanPtr = &rowGatherPlan;
    if (useSourcePingPongOffsetOutputAliasRowGather) {
        selectedRowGatherPlanPtr = &sourcePingPongOffsetOutputAliasRowGatherPlan;
    }
    if (useOffsetOutputAliasRowGather) {
        selectedRowGatherPlanPtr = &offsetOutputAliasRowGatherPlan;
    }
    if (useAdaptiveGroupBatchRowGather) {
        selectedRowGatherPlanPtr = &adaptiveGroupBatchRowGatherPlan;
    }
    if (useTwoPlaneRowGather) {
        selectedRowGatherPlanPtr = &twoPlaneRowGatherPlan;
    }
    if (useMaskedCopyRow) {
        selectedRowGatherPlanPtr = &maskedCopyRowPlan;
    }
    if (useSourcePingPongRowGather) {
        selectedRowGatherPlanPtr = &sourcePingPongRowGatherPlan;
    }
    if (useMergedBatchRowGather || useSplitOutputRowGather) {
        selectedRowGatherPlanPtr = &mergedBatchRowGatherPlan;
    }
    if (useStagedSourceRowGather) {
        selectedRowGatherPlanPtr = &stagedSourceRowGatherPlan;
    }
    if (useBlockGatherRow) {
        selectedRowGatherPlanPtr = &blockGatherRowPlan;
    }
    if (useRowMajorStagedRowGather) {
        selectedRowGatherPlanPtr = &rowMajorStagedRowGatherPlan;
    }
    if (useCachedOffsetStagedRowGather) {
        selectedRowGatherPlanPtr = &cachedOffsetStagedRowGatherPlan;
    }
    if (useFullStageOffsetRowGather) {
        selectedRowGatherPlanPtr = &fullStageOffsetRowGatherPlan;
    }
    if (useCompactStageOffsetRowGather) {
        selectedRowGatherPlanPtr = &stagedSourceRowGatherPlan;
    }
    if (useMicroRegisterRowGather) {
        selectedRowGatherPlanPtr = &microRegisterRowGatherPlan;
    }
    if (useDirectMteRowGather) {
        selectedRowGatherPlanPtr = &directMteRowGatherPlan;
    }
    if (useRollingStageOffsetRowGather) {
        selectedRowGatherPlanPtr = &stagedSourceRowGatherPlan;
    }
    if (useDeltaStagedOffsetRowGather) {
        selectedRowGatherPlanPtr = &deltaStagedOffsetRowGatherPlan;
    }
    if (usePartitionedRowGather) {
        selectedRowGatherPlanPtr = &partitionedRowGatherPlan;
    }
    if (useWavefrontRowGather) {
        selectedRowGatherPlanPtr = &wavefrontRowGatherPlan;
    }
    if (useDirectWavefrontRowGather) {
        selectedRowGatherPlanPtr = &directWavefrontRowGatherPlan;
    }
    if (useSuffixPrefetchRowGather) {
        selectedRowGatherPlanPtr = &suffixPrefetchRowGatherPlan;
    }
    const RowGatherPlan& selectedRowGatherPlan = *selectedRowGatherPlanPtr;
    const bool bypassRowInputL2 = ShouldBypassRowInputL2(best, outer, outputRowBytes,
        tileBytes);
    const bool usePrecomputedRowBytes = ShouldUsePrecomputedRowBytes(best, bypassRowInputL2,
        firstDesc->GetDataType(), inputs);
    const bool useSingleBufferRowBytes = ShouldUseSingleBufferRowBytes(best,
        usePrecomputedRowBytes, outer, tileBytes, singleTileBytes, inputs);
    PackedRowPlan packedRowPlan;
    uint32_t packedBlockDim = best.blockDim;
    bool packedRowReady = BuildPackedRowPlan(best, outer, outputRowBytes,
        packedBlockDim, ubSize, dtypeSize, inputs, tileBytes, packedRowPlan);
    context->SetBlockDim(best.blockDim);
    context->SetTilingKey(bypassRowInputL2 ? CONCAT_KEY_ROW_STREAM_INPUT_L2_BYPASS : best.key);
    if (packedRowReady) {
        context->SetBlockDim(packedBlockDim);
        ConcatPackedRowTilingData tiling;
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_rowsPerCore(packedRowPlan.rowsPerCore);
        tiling.set_outputRowElements(packedRowPlan.outputRowElements);
        tiling.set_sourceBufferBytes(packedRowPlan.sourceBufferBytes);
        tiling.set_outputBufferBytes(packedRowPlan.outputBufferBytes);
        tiling.set_offsetBufferBytes(packedRowPlan.offsetBufferBytes);
        tiling.set_outer(outer);
        tiling.set_outputRowBytes(outputRowBytes);
        tiling.set_inputRowElements(packedRowPlan.inputRowElements.data());
        context->SetTilingKey(CONCAT_KEY_ROW_PACKED_UB);
        return SaveTiling(context, tiling);
    }
    if (usePairGather) {
        if (pairGatherPlan.tasks.empty() ||
            pairGatherPlan.tasks.size() > CONCAT_MAX_PAIR_GATHER_TASKS) {
            return ge::GRAPH_FAILED;
        }
        std::array<uint32_t, CONCAT_MAX_GRID_INPUTS> inputRowElements{};
        std::array<uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskInputStart{};
        std::array<uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskInputSplit{};
        std::array<uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskInputEnd{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskOutputStart{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskGroup0Elements{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskGroup1Elements{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskPairElements{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskSourceElements{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskBatchRows{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskSourceBufferBytes{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskOutputBufferBytes{};
        std::array<uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskOffsetBufferBytes{};
        std::array<uint64_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskRowStart{};
        std::array<uint64_t, CONCAT_MAX_PAIR_GATHER_TASKS> taskRowCount{};
        for (const InputInfo& input : inputs) {
            if (input.index >= CONCAT_MAX_GRID_INPUTS || input.rowBytes % GATHER_DTYPE_BYTES != 0 ||
                input.rowBytes / GATHER_DTYPE_BYTES > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            inputRowElements[input.index] = static_cast<uint32_t>(
                input.rowBytes / GATHER_DTYPE_BYTES);
        }
        for (size_t i = 0; i < pairGatherPlan.tasks.size(); ++i) {
            const PairGatherTask& task = pairGatherPlan.tasks[i];
            taskInputStart[i] = task.inputStart;
            taskInputSplit[i] = task.inputSplit;
            taskInputEnd[i] = task.inputEnd;
            taskOutputStart[i] = task.outputStart;
            taskGroup0Elements[i] = task.group0Elements;
            taskGroup1Elements[i] = task.group1Elements;
            taskPairElements[i] = task.pairElements;
            taskSourceElements[i] = task.sourceElements;
            taskBatchRows[i] = task.batchRows;
            taskSourceBufferBytes[i] = task.sourceBufferBytes;
            taskOutputBufferBytes[i] = task.outputBufferBytes;
            taskOffsetBufferBytes[i] = task.offsetBufferBytes;
            taskRowStart[i] = task.rowStart;
            taskRowCount[i] = task.rowCount;
        }
        ConcatPairGatherTilingData tiling;
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_taskCount(static_cast<uint32_t>(pairGatherPlan.tasks.size()));
        tiling.set_outer(outer);
        tiling.set_outputRowElements(outputRowBytes / GATHER_DTYPE_BYTES);
        tiling.set_inputRowElements(inputRowElements.data());
        tiling.set_taskInputStart(taskInputStart.data());
        tiling.set_taskInputSplit(taskInputSplit.data());
        tiling.set_taskInputEnd(taskInputEnd.data());
        tiling.set_taskOutputStart(taskOutputStart.data());
        tiling.set_taskGroup0Elements(taskGroup0Elements.data());
        tiling.set_taskGroup1Elements(taskGroup1Elements.data());
        tiling.set_taskPairElements(taskPairElements.data());
        tiling.set_taskSourceElements(taskSourceElements.data());
        tiling.set_taskBatchRows(taskBatchRows.data());
        tiling.set_taskSourceBufferBytes(taskSourceBufferBytes.data());
        tiling.set_taskOutputBufferBytes(taskOutputBufferBytes.data());
        tiling.set_taskOffsetBufferBytes(taskOffsetBufferBytes.data());
        tiling.set_taskRowStart(taskRowStart.data());
        tiling.set_taskRowCount(taskRowCount.data());
        context->SetBlockDim(static_cast<uint32_t>(pairGatherPlan.tasks.size()));
        context->SetTilingKey(CONCAT_KEY_ROW_GATHER_PAIR);
        return SaveTiling(context, tiling);
    }
    if (useRowGather) {
        if (useBandGatherRowGather) {
            if (bandGatherPlan.tasks.empty() ||
                bandGatherPlan.tasks.size() > CONCAT_MAX_BAND_GATHER_TASKS) {
                return ge::GRAPH_FAILED;
            }
            std::array<uint32_t, CONCAT_MAX_GRID_INPUTS> inputRowElements{};
            std::array<uint16_t, CONCAT_MAX_BAND_GATHER_TASKS> taskInputStart{};
            std::array<uint16_t, CONCAT_MAX_BAND_GATHER_TASKS> taskInputEnd{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskOutputStart{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskGroupElements{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskSourceElements{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskBatchRows{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskSourceBufferBytes{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskOutputBufferBytes{};
            std::array<uint32_t, CONCAT_MAX_BAND_GATHER_TASKS> taskOffsetBufferBytes{};
            std::array<uint64_t, CONCAT_MAX_BAND_GATHER_TASKS> taskRowStart{};
            std::array<uint64_t, CONCAT_MAX_BAND_GATHER_TASKS> taskRowCount{};
            for (const InputInfo& input : inputs) {
                if (input.index >= CONCAT_MAX_GRID_INPUTS ||
                    input.rowBytes % GATHER_DTYPE_BYTES != 0 ||
                    input.rowBytes / GATHER_DTYPE_BYTES > std::numeric_limits<uint32_t>::max()) {
                    return ge::GRAPH_FAILED;
                }
                inputRowElements[input.index] = static_cast<uint32_t>(
                    input.rowBytes / GATHER_DTYPE_BYTES);
            }
            for (size_t taskIndex = 0; taskIndex < bandGatherPlan.tasks.size(); ++taskIndex) {
                const BandGatherTask& task = bandGatherPlan.tasks[taskIndex];
                if (task.inputStart >= task.inputEnd || task.inputEnd > inputCount ||
                    task.outputStart >= outputRowBytes / GATHER_DTYPE_BYTES ||
                    task.groupElements == 0 || task.sourceElements == 0 || task.batchRows == 0 ||
                    task.sourceBufferBytes == 0 || task.outputBufferBytes == 0 ||
                    task.offsetBufferBytes == 0 || task.rowCount == 0 ||
                    task.rowStart > outer || task.rowCount > outer - task.rowStart) {
                    return ge::GRAPH_FAILED;
                }
                taskInputStart[taskIndex] = task.inputStart;
                taskInputEnd[taskIndex] = task.inputEnd;
                taskOutputStart[taskIndex] = task.outputStart;
                taskGroupElements[taskIndex] = task.groupElements;
                taskSourceElements[taskIndex] = task.sourceElements;
                taskBatchRows[taskIndex] = task.batchRows;
                taskSourceBufferBytes[taskIndex] = task.sourceBufferBytes;
                taskOutputBufferBytes[taskIndex] = task.outputBufferBytes;
                taskOffsetBufferBytes[taskIndex] = task.offsetBufferBytes;
                taskRowStart[taskIndex] = task.rowStart;
                taskRowCount[taskIndex] = task.rowCount;
            }
            ConcatBandGatherTilingData tiling;
            tiling.set_inputCount(inputCount);
            tiling.set_dtypeSize(dtypeSize);
            tiling.set_taskCount(static_cast<uint32_t>(bandGatherPlan.tasks.size()));
            tiling.set_outer(outer);
            tiling.set_outputRowElements(outputRowBytes / GATHER_DTYPE_BYTES);
            tiling.set_inputRowElements(inputRowElements.data());
            tiling.set_taskInputStart(taskInputStart.data());
            tiling.set_taskInputEnd(taskInputEnd.data());
            tiling.set_taskOutputStart(taskOutputStart.data());
            tiling.set_taskGroupElements(taskGroupElements.data());
            tiling.set_taskSourceElements(taskSourceElements.data());
            tiling.set_taskBatchRows(taskBatchRows.data());
            tiling.set_taskSourceBufferBytes(taskSourceBufferBytes.data());
            tiling.set_taskOutputBufferBytes(taskOutputBufferBytes.data());
            tiling.set_taskOffsetBufferBytes(taskOffsetBufferBytes.data());
            tiling.set_taskRowStart(taskRowStart.data());
            tiling.set_taskRowCount(taskRowCount.data());
            context->SetBlockDim(static_cast<uint32_t>(bandGatherPlan.tasks.size()));
            context->SetTilingKey(CONCAT_KEY_ROW_BAND_GATHER);
            return SaveTiling(context, tiling);
        }
        uint64_t rowGatherKey = CONCAT_KEY_ROW_GATHER;
        if (useSuffixPrefetchRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_SUFFIX_PREFETCH;
        } else if (useSourcePingPongOffsetOutputAliasRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_SOURCE_PINGPONG_OFFSET_OUTPUT_ALIAS;
        } else if (useOffsetOutputAliasRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_OFFSET_OUTPUT_ALIAS;
        } else if (useAdaptiveGroupBatchRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_GROUP_BATCH;
        } else if (useTwoPlaneRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_TWO_PLANE;
        } else if (useDirectWavefrontRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_DIRECT_WAVEFRONT;
        } else if (useWavefrontRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_WAVEFRONT;
        } else if (useGroupPhaseRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_GROUP_PHASE;
        } else if (useCoreSwizzleRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_CORE_SWIZZLE;
        } else if (useSharedSeedRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_SHARED_SEED;
        } else if (useMaskedCopyRow) {
            rowGatherKey = CONCAT_KEY_ROW_MASKED_COPY;
        } else if (useCompactStageOffsetRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_COMPACT_STAGE_OFFSETS;
        } else if (useDirectMteRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_DIRECT_MTE;
        } else if (useRollingStageOffsetRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_ROLLING_STAGE_OFFSETS;
        } else if (useDeltaStagedOffsetRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_DELTA_STAGE_OFFSETS;
        } else if (useMicroRegisterRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_MICRO;
        } else if (useFullStageOffsetRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_FULL_STAGE_OFFSETS;
        } else if (useCachedOffsetStagedRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_CACHED_OFFSETS;
        } else if (useSplitOutputRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_SPLIT_OUTPUT;
        } else if (useRowMajorStagedRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_ROW_MAJOR_STAGED;
        } else if (useBlockGatherRow) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_BLOCK;
        } else if (useStagedSourceRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_STAGED_SOURCE;
        } else if (useMergedBatchRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_MERGED_BATCH;
        } else if (useSourcePingPongRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_SOURCE_PINGPONG;
        } else if (useDirectRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_DIRECT;
        } else if (useResidentRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_RESIDENT;
        } else if (useOutputPingPongRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_OUTPUT_PINGPONG;
        } else if (usePartitionedRowGather) {
            rowGatherKey = CONCAT_KEY_ROW_GATHER_PARTITION_PIPELINE;
        }
        if (useMixedAivProbe && best.blockDim >= 2 && (best.blockDim & 1U) == 0) {
            // MIX_AIC_1_2 的一个 block 对应一个 AIC 与两个 AIV；RowGather 的逻辑核数
            // 仍为原 blockDim，实际 blockDim 因此折半。32 B workspace 是纯 Vector 代码
            // 使用 MIX 类型时的运行时要求，不承载算子数据。
            context->SetBlockDim(best.blockDim / 2);
            rowGatherKey = CONCAT_KEY_ROW_GATHER_MIX_AIV_PROBE;
        }
        context->SetTilingKey(rowGatherKey);
        ConcatRowGatherTilingData tiling;
        std::array<uint32_t, CONCAT_MAX_GRID_INPUTS> inputRowElements{};
        std::array<uint16_t, CONCAT_MAX_GATHER_GROUPS> groupInputEnd{};
        std::array<uint16_t, CONCAT_MAX_GATHER_GROUPS> groupInputSplit{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupElements{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupSourceElements{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupBatchRows{};
        std::array<uint16_t, CONCAT_MAX_GATHER_GROUPS> groupSuffixInputStart{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupSuffixOutputStart{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupSuffixElements{};
        std::array<uint32_t, CONCAT_MAX_GATHER_GROUPS> groupSuffixSourceElements{};
        for (const InputInfo& input : inputs) {
            if (input.index >= CONCAT_MAX_GRID_INPUTS || input.rowBytes % GATHER_DTYPE_BYTES != 0 ||
                input.rowBytes / GATHER_DTYPE_BYTES > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            inputRowElements[input.index] = static_cast<uint32_t>(input.rowBytes / GATHER_DTYPE_BYTES);
        }
        for (size_t planIndex = 0; planIndex < selectedRowGatherPlan.subPlans.size(); ++planIndex) {
            const GatherSubPlan& subPlan = selectedRowGatherPlan.subPlans[planIndex];
            const size_t base = planIndex * CONCAT_MAX_GRID_INPUTS;
            for (size_t group = 0; group < subPlan.groupInputEnd.size(); ++group) {
                groupInputEnd[base + group] = subPlan.groupInputEnd[group];
                if (group < subPlan.groupInputSplit.size()) {
                    groupInputSplit[base + group] = subPlan.groupInputSplit[group];
                }
                groupElements[base + group] = subPlan.groupElements[group];
                groupSourceElements[base + group] = subPlan.groupSourceElements[group];
                groupBatchRows[base + group] = group < subPlan.groupBatchRows.size() ?
                    subPlan.groupBatchRows[group] : subPlan.batchRows;
                if (group < subPlan.groupSuffixInputStart.size()) {
                    groupSuffixInputStart[base + group] = subPlan.groupSuffixInputStart[group];
                    groupSuffixOutputStart[base + group] = subPlan.groupSuffixOutputStart[group];
                    groupSuffixElements[base + group] = subPlan.groupSuffixElements[group];
                    groupSuffixSourceElements[base + group] =
                        group < subPlan.groupSuffixSourceElements.size() ?
                        subPlan.groupSuffixSourceElements[group] : 0;
                }
            }
        }
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_highCoreCount(selectedRowGatherPlan.highCoreCount);
        tiling.set_rowsPerCore0(selectedRowGatherPlan.subPlans[0].rowsPerCore);
        tiling.set_rowsPerCore1(selectedRowGatherPlan.subPlans[1].rowsPerCore);
        tiling.set_batchRows0(selectedRowGatherPlan.subPlans[0].batchRows);
        tiling.set_batchRows1(selectedRowGatherPlan.subPlans[1].batchRows);
        tiling.set_sourceBatchRows0(selectedRowGatherPlan.subPlans[0].sourceBatchRows);
        tiling.set_sourceBatchRows1(selectedRowGatherPlan.subPlans[1].sourceBatchRows);
        tiling.set_groupCount0(static_cast<uint32_t>(selectedRowGatherPlan.subPlans[0].groupInputEnd.size()));
        tiling.set_groupCount1(static_cast<uint32_t>(selectedRowGatherPlan.subPlans[1].groupInputEnd.size()));
        tiling.set_sourceBufferBytes0(selectedRowGatherPlan.subPlans[0].sourceBufferBytes);
        tiling.set_sourceBufferBytes1(selectedRowGatherPlan.subPlans[1].sourceBufferBytes);
        tiling.set_outputBufferBytes0(selectedRowGatherPlan.subPlans[0].outputBufferBytes);
        tiling.set_outputBufferBytes1(selectedRowGatherPlan.subPlans[1].outputBufferBytes);
        tiling.set_offsetBufferBytes0(selectedRowGatherPlan.subPlans[0].offsetBufferBytes);
        tiling.set_offsetBufferBytes1(selectedRowGatherPlan.subPlans[1].offsetBufferBytes);
        tiling.set_sourcePart0BufferBytes0(selectedRowGatherPlan.subPlans[0].sourcePart0BufferBytes);
        tiling.set_sourcePart0BufferBytes1(selectedRowGatherPlan.subPlans[1].sourcePart0BufferBytes);
        tiling.set_sourcePart1BufferBytes0(selectedRowGatherPlan.subPlans[0].sourcePart1BufferBytes);
        tiling.set_sourcePart1BufferBytes1(selectedRowGatherPlan.subPlans[1].sourcePart1BufferBytes);
        tiling.set_outputPart0BufferBytes0(selectedRowGatherPlan.subPlans[0].outputPart0BufferBytes);
        tiling.set_outputPart0BufferBytes1(selectedRowGatherPlan.subPlans[1].outputPart0BufferBytes);
        tiling.set_outputPart1BufferBytes0(selectedRowGatherPlan.subPlans[0].outputPart1BufferBytes);
        tiling.set_outputPart1BufferBytes1(selectedRowGatherPlan.subPlans[1].outputPart1BufferBytes);
        tiling.set_outer(outer);
        tiling.set_outputRowElements(outputRowBytes / GATHER_DTYPE_BYTES);
        tiling.set_inputRowElements(inputRowElements.data());
        tiling.set_groupInputEnd(groupInputEnd.data());
        tiling.set_groupInputSplit(groupInputSplit.data());
        tiling.set_groupElements(groupElements.data());
        tiling.set_groupSourceElements(groupSourceElements.data());
        tiling.set_groupBatchRows(groupBatchRows.data());
        tiling.set_groupSuffixInputStart(groupSuffixInputStart.data());
        tiling.set_groupSuffixOutputStart(groupSuffixOutputStart.data());
        tiling.set_groupSuffixElements(groupSuffixElements.data());
        tiling.set_groupSuffixSourceElements(groupSuffixSourceElements.data());
        if (useSharedSeedRowGather) {
            const size_t systemWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
            if (sharedSeedWorkspaceBytes >
                std::numeric_limits<size_t>::max() - systemWorkspaceBytes ||
                context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            return SaveTiling(context, tiling, systemWorkspaceBytes +
                static_cast<size_t>(sharedSeedWorkspaceBytes));
        }
        if (useMixedAivProbe && rowGatherKey == CONCAT_KEY_ROW_GATHER_MIX_AIV_PROBE) {
            return SaveTiling(context, tiling, 32);
        }
        return SaveTiling(context, tiling);
    }
    if (best.kind == PlanKind::FLAT) {
        ConcatPreparedFlatTwoCommandsTilingData twoCommandTiling;
        std::array<uint64_t, CONCAT_MAX_FLAT_CORES> twoCommandCoreOutputStart{};
        std::array<uint16_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS> commandInput{};
        std::array<uint64_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS> commandInputOffset{};
        std::array<uint64_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS> commandOutputOffset{};
        std::array<uint32_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS> commandBytes{};
        if (BuildPreparedFlatTwoCommands(inputs, best, totalBytes, tileBytes,
            twoCommandCoreOutputStart, commandInput, commandInputOffset,
            commandOutputOffset, commandBytes)) {
            context->SetTilingKey(CONCAT_KEY_PREPARED_FLAT_TWO_COMMANDS_STATIC);
            twoCommandTiling.set_inputCount(inputCount);
            twoCommandTiling.set_dtypeSize(dtypeSize);
            twoCommandTiling.set_tileBytes(tileBytes);
            twoCommandTiling.set_coreOutputStart(twoCommandCoreOutputStart.data());
            twoCommandTiling.set_commandInput(commandInput.data());
            twoCommandTiling.set_commandInputOffset(commandInputOffset.data());
            twoCommandTiling.set_commandOutputOffset(commandOutputOffset.data());
            twoCommandTiling.set_commandBytes(commandBytes.data());
            return SaveTiling(context, twoCommandTiling);
        }

        ConcatPreparedFlatTilingData preparedTiling;
        std::array<uint64_t, CONCAT_MAX_FLAT_CORES> coreOutputStart{};
        std::array<uint16_t, CONCAT_MAX_FLAT_CORES> coreTaskEnd{};
        std::array<uint16_t, CONCAT_MAX_PREPARED_FLAT_TASKS> taskInput{};
        std::array<uint64_t, CONCAT_MAX_PREPARED_FLAT_TASKS> taskInputOffset{};
        std::array<uint64_t, CONCAT_MAX_PREPARED_FLAT_TASKS> taskBytes{};
        if (BuildPreparedFlatTasks(inputs, best, totalBytes, tileBytes,
            coreOutputStart, coreTaskEnd,
            taskInput, taskInputOffset, taskBytes)) {
            context->SetTilingKey(CONCAT_KEY_PREPARED_FLAT);
            preparedTiling.set_inputCount(inputCount);
            preparedTiling.set_dtypeSize(dtypeSize);
            preparedTiling.set_tileBytes(tileBytes);
            preparedTiling.set_coreChunkBytes(best.coreChunkBytes);
            preparedTiling.set_coreOutputStart(coreOutputStart.data());
            preparedTiling.set_coreTaskEnd(coreTaskEnd.data());
            preparedTiling.set_taskInput(taskInput.data());
            preparedTiling.set_taskInputOffset(taskInputOffset.data());
            preparedTiling.set_taskBytes(taskBytes.data());
            return SaveTiling(context, preparedTiling);
        }

        // 任务数量超过固定表容量时保留描述符解析路径，保证大输入泛化。
        ConcatFlatTilingData tiling;
        std::array<uint32_t, CONCAT_MAX_FLAT_CORES> coreStartInput{};
        std::array<uint64_t, CONCAT_MAX_FLAT_CORES> coreStartInputOffset{};
        if (!BuildFlatCoreStarts(inputs, best, totalBytes, inputCount,
            coreStartInput, coreStartInputOffset)) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_tileBytes(tileBytes);
        tiling.set_totalBytes(totalBytes);
        tiling.set_coreChunkBytes(best.coreChunkBytes);
        tiling.set_coreStartInput(coreStartInput.data());
        tiling.set_coreStartInputOffset(coreStartInputOffset.data());
        return SaveTiling(context, tiling);
    }
    if (best.kind == PlanKind::GRID) {
        if (best.key == CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER) {
            ConcatPreparedRectTilingData preparedTiling;
            std::array<uint64_t, CONCAT_MAX_PREPARED_RECT_WORDS> commandData{};
            const bool bypassInputL2 = ShouldBypassPreparedRectInputL2(best, outer,
                firstDesc->GetDataType());
            if (BuildPreparedRectCommands(inputs, best, outer, outputRowBytes,
                bypassInputL2, commandData)) {
                context->SetTilingKey(bypassInputL2 ?
                    CONCAT_KEY_PREPARED_RECT_STATIC_STREAM_INPUT_L2_BYPASS :
                    CONCAT_KEY_PREPARED_RECT_STATIC);
                preparedTiling.set_inputCount(inputCount);
                preparedTiling.set_dtypeSize(dtypeSize);
                preparedTiling.set_tileBytes(best.tileBytes);
                preparedTiling.set_commandData(commandData.data());
                return SaveTiling(context, preparedTiling);
            }
        }
        std::array<uint64_t, CONCAT_MAX_GRID_COLS> coreSpan{};
        std::array<uint64_t, CONCAT_MAX_GRID_INPUTS> inputRowBytes{};
        bool gridSpanReady = false;
        uint64_t gridSpanKey = CONCAT_KEY_GRID_SPAN;
        if (best.key == CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER) {
            // key8 只会在 BuildGridSpans 预校验成功后入选；失败时保留旧 Plan。
            gridSpanReady = true;
            coreSpan = singleGridCoreSpan;
            inputRowBytes = singleGridInputRowBytes;
            gridSpanKey = CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER;
        } else {
            gridSpanReady = BuildGridSpans(inputs, best, outputRowBytes, inputCount,
                coreSpan, inputRowBytes);
        }
        if (gridSpanReady) {
            context->SetTilingKey(gridSpanKey);
            ConcatGridSpanTilingData tiling;
            tiling.set_inputCount(inputCount);
            tiling.set_dtypeSize(dtypeSize);
            tiling.set_tileBytes(best.tileBytes);
            tiling.set_rowGroups(best.rowGroups);
            tiling.set_colGroups(best.colGroups);
            tiling.set_reserved(0);
            tiling.set_outer(outer);
            tiling.set_outputRowBytes(outputRowBytes);
            tiling.set_colChunkBytes(best.colChunkBytes);
            tiling.set_coreSpan(coreSpan.data());
            tiling.set_inputRowBytes(inputRowBytes.data());
            return SaveTiling(context, tiling);
        }
        ConcatGridTilingData tiling;
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_tileBytes(best.tileBytes);
        tiling.set_rowGroups(best.rowGroups);
        tiling.set_colGroups(best.colGroups);
        tiling.set_outer(outer);
        tiling.set_outputRowBytes(outputRowBytes);
        tiling.set_colChunkBytes(best.colChunkBytes);
        return SaveTiling(context, tiling);
    }
    if (best.kind == PlanKind::INPUT_LPT) {
        ConcatInputLptTilingData tiling;
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_tileBytes(tileBytes);
        tiling.set_taskCount(static_cast<uint32_t>(best.tasks.size()));
        tiling.set_outer(outer);
        tiling.set_outputRowBytes(outputRowBytes);
        std::array<uint16_t, CONCAT_MAX_LPT_TASKS> taskInput{};
        std::array<uint16_t, CONCAT_MAX_LPT_TASKS> taskCore{};
        std::array<uint16_t, CONCAT_MAX_LPT_CORES> coreTaskStart{};
        std::array<uint16_t, CONCAT_MAX_LPT_CORES> coreTaskCount{};
        std::array<uint32_t, CONCAT_MAX_LPT_TASKS> taskRowStart{};
        std::array<uint32_t, CONCAT_MAX_LPT_TASKS> taskRowCount{};
        std::array<uint64_t, CONCAT_MAX_LPT_TASKS> taskRowBytes{};
        std::array<uint64_t, CONCAT_MAX_LPT_TASKS> taskOutputCol{};
        for (size_t i = 0; i < best.tasks.size(); ++i) {
            taskInput[i] = best.tasks[i].input;
            taskCore[i] = best.tasks[i].core;
            taskRowStart[i] = best.tasks[i].rowStart;
            taskRowCount[i] = best.tasks[i].rowCount;
            taskRowBytes[i] = inputs[best.tasks[i].input].rowBytes;
            taskOutputCol[i] = inputs[best.tasks[i].input].outputCol;
            const uint16_t core = best.tasks[i].core;
            if (core >= CONCAT_MAX_LPT_CORES) {
                return ge::GRAPH_FAILED;
            }
            if (coreTaskCount[core] == 0) {
                coreTaskStart[core] = static_cast<uint16_t>(i);
            }
            ++coreTaskCount[core];
        }
        tiling.set_coreTaskStart(coreTaskStart.data());
        tiling.set_coreTaskCount(coreTaskCount.data());
        tiling.set_taskInput(taskInput.data());
        tiling.set_taskCore(taskCore.data());
        tiling.set_taskRowStart(taskRowStart.data());
        tiling.set_taskRowCount(taskRowCount.data());
        tiling.set_taskRowBytes(taskRowBytes.data());
        tiling.set_taskOutputCol(taskOutputCol.data());
        return SaveTiling(context, tiling);
    }
    if (usePrecomputedRowBytes) {
        ConcatRowPrecomputedTilingData tiling;
        std::array<uint32_t, CONCAT_MAX_GRID_INPUTS> inputRowBytes{};
        for (const InputInfo& input : inputs) {
            if (input.index >= inputRowBytes.size() ||
                input.rowBytes > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            inputRowBytes[input.index] = static_cast<uint32_t>(input.rowBytes);
        }
        context->SetTilingKey(useSingleBufferRowBytes ?
            CONCAT_KEY_ROW_PRECOMPUTED_SINGLE_BUFFER : CONCAT_KEY_ROW_PRECOMPUTED_BYTES);
        tiling.set_inputCount(inputCount);
        tiling.set_dtypeSize(dtypeSize);
        tiling.set_tileBytes(useSingleBufferRowBytes ? singleTileBytes : tileBytes);
        tiling.set_outer(outer);
        tiling.set_outputRowBytes(outputRowBytes);
        tiling.set_inputRowBytes(inputRowBytes.data());
        return SaveTiling(context, tiling);
    }
    ConcatRowTilingData tiling;
    tiling.set_inputCount(inputCount);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_tileBytes(tileBytes);
    tiling.set_outer(outer);
    tiling.set_outputRowBytes(outputRowBytes);
    return SaveTiling(context, tiling);
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetAttrs() == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* computeNodeInfo = context->GetComputeNodeInfo();
    if (computeNodeInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* inputInstance = computeNodeInfo->GetInputInstanceInfo(0);
    if (inputInstance == nullptr || inputInstance->GetInstanceNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* firstShape = context->GetDynamicInputShape(0, 0);
    const int64_t* dimAttr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (firstShape == nullptr || dimAttr == nullptr || firstShape->GetDimNum() == 0 ||
        firstShape->GetDimNum() > MAX_SUPPORTED_RANK) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = firstShape->GetDimNum();
    int64_t dim = *dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *firstShape;
    int64_t concatSize = 0;
    for (uint32_t i = 0; i < inputInstance->GetInstanceNum(); ++i) {
        const gert::Shape* shape = context->GetDynamicInputShape(0, i);
        if (shape == nullptr || shape->GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != static_cast<size_t>(dim) && shape->GetDim(axis) != firstShape->GetDim(axis)) {
                return ge::GRAPH_FAILED;
            }
        }
        const int64_t axisSize = shape->GetDim(dim);
        if (axisSize < 0 || concatSize > std::numeric_limits<int64_t>::max() - axisSize) {
            return ge::GRAPH_FAILED;
        }
        concatSize += axisSize;
    }
    outputShape->SetDim(dim, concatSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}
