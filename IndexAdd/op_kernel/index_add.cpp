#include "kernel_operator.h"
#include "index_add_tiling_key.h"

namespace IndexAdd {
using namespace AscendC;

constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t VECTOR_BYTES = 256;
constexpr uint32_t MAX_COPY_ROWS = 4095;
constexpr uint64_t MAX_COPY_STRIDE = 0xFFFFFFFFULL;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
constexpr uint32_t COOP_RAW_SLOT_BYTES = 32;
constexpr uint32_t STRATEGY_ATOMIC = 0;
constexpr uint32_t STRATEGY_UB = 1;
constexpr uint32_t STRATEGY_NARROW_UB = 2;
constexpr uint32_t STRATEGY_GATHER_NARROW = 3;
constexpr uint32_t STRATEGY_BATCH_GATHER_NARROW = 4;
constexpr uint32_t STRATEGY_BF16_RESIDENT = 5;
constexpr uint32_t STRATEGY_INT8_RESIDENT = 6;
constexpr uint32_t STRATEGY_INT8_AXIS = 7;
constexpr uint32_t STRATEGY_BF16_SOURCE_DB = 8;
constexpr uint32_t STRATEGY_GATHER_NARROW_DB = 9;
constexpr uint32_t STRATEGY_UB_MIXED = 10;
constexpr uint32_t STRATEGY_GATHER_BATCH_WIDE = 11;
constexpr uint32_t STRATEGY_UB_MIXED_DB = 12;
// BF16 相邻更新仅在目标行两两独立时共用同一 Vector 窗口；四条工作行均为 32B 对齐。
constexpr uint32_t BF16_INDEPENDENT_WORK_ROWS = 4;
constexpr uint32_t GATHER_PIPELINE_ALIGNMENT_BYTES = 512;
constexpr uint32_t GATHER_PIPELINE_SOURCE_SKEW_BYTES = 64;
constexpr uint32_t GATHER_PIPELINE_DENSE_SKEW_BYTES = 128;
constexpr uint32_t GATHER_PIPELINE_FP32_DENSE_SKEW_BYTES = VECTOR_BYTES;
constexpr uint32_t GATHER_PIPELINE_TAIL_BYTES = 256;
constexpr uint32_t TRANSPOSE_BLOCK_GATHER_FP16 = 1;
constexpr uint32_t TRANSPOSE_BLOCK_GATHER_FP32 = 2;
constexpr uint32_t FP32_TRANSPOSE_ROWS = 16;
constexpr uint32_t FP32_TRANSPOSE_ACTIVE_ROWS = FP32_TRANSPOSE_ROWS;
constexpr uint32_t FP32_TRANSPOSE_COL_ALIGNMENT = UB_BLOCK_BYTES / sizeof(float);
// 低、高 half 分别落在独立的 8-float record 中，单条 Gatherb offset 的
// 物理步长为 32B；两半共用同一层 metadata，但不再合并成 64B record。
constexpr uint32_t FP32_TRANSPOSE_OFFSET_SCALE_BYTES =
    (FP32_TRANSPOSE_ROWS / 2) * sizeof(float);

template <typename T, bool WIDE_BATCH_GATHER = false>
class KernelIndexAdd {
public:
    template <bool BypassResidentL2Streams>
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
        GM_ADDR userWorkspace, const IndexAddTilingData* tilingData, TPipe* pipe, bool cooperative)
    {
        pipe_ = pipe;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if constexpr (BypassResidentL2Streams) {
            // 此 tiling key 仅由超 L2 的 resident source 单次顺序流选择。resident
            // 路径中的 self 与 output 也各只顺序访问一次，禁用三条数据流的 L2
            // 分配，保留跨 outer 复用的 index 表。
            selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (WIDE_BATCH_GATHER) {
            // 宽转置 Gather 的 source 每个 outer 行只顺序消费一次，且不会在后续
            // batch 重读；绕过 L2 分配避免这条只读流替换 index 与协同元数据。
            sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        outerSize_ = tilingData->outerSize;
        axisSize_ = tilingData->axisSize;
        innerSize_ = tilingData->innerSize;
        indexSize_ = tilingData->indexSize;
        innerTile_ = tilingData->innerTile;
        outerTile_ = tilingData->outerTile;
        outerBatch_ = tilingData->outerBatch;
        strategy_ = tilingData->strategy;
        innerTileNum_ = tilingData->innerTileNum;
        taskNum_ = tilingData->taskNum;
        sourceChunk_ = tilingData->sourceChunk;
        blockDim_ = tilingData->blockDim;
        coopLayerCap_ = tilingData->coopLayerCap;
        mixedInnerTile_ = tilingData->mixedInnerTile;
        mixedInnerTileNum_ = tilingData->mixedInnerTileNum;
        mixedOuterCount_ = tilingData->mixedOuterCount;
        residentBufferBytes_ = tilingData->residentBufferBytes;
        residentDoubleBuffer_ = tilingData->residentDoubleBuffer;
        gatherUbPaddedAxis_ = tilingData->gatherUbPaddedAxis;
        // 仅宽 Gather 模板实例消费转置字段；普通 dtype 的初始化路径不读取
        // FP32 专属元数据，避免给 int8/BF16/常规 FP16 kernel 增加无效标量访存。
        if constexpr (WIDE_BATCH_GATHER) {
            transposeBlockGatherMode_ = tilingData->transposeBlockGather;
            if constexpr (IsSameType<T, float>::value) {
                transposeSourceChunk_ = tilingData->transposeSourceChunk;
                transposeLayerCap_ = tilingData->transposeLayerCap;
                transposePayloadLayerCap_ = tilingData->transposePayloadLayerCap;
                transposeOffsetBytes_ = tilingData->transposeOffsetBytes;
            }
        }
        cooperative_ = cooperative;

        if (cooperative_) {
            workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
            workspaceOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(userWorkspace));
        }

        if (sourceChunk_ == 0) {
            sourceChunk_ = 1;
        }
        if (outerBatch_ == 0) {
            outerBatch_ = 1;
        }

        if constexpr (WIDE_BATCH_GATHER) {
            // true 模板只服务于 FP16 宽 Gather 与 FP32 分 chunk Gatherb 的转置路径；
            // 其他 dtype 仍保持原有 UB 布局和指令流。
            InitWideBuffers();
            return;
        }
        InitStandardBuffers();
    }

    template <bool COOPERATIVE = false>
    __aicore__ inline void Process()
    {
        if constexpr (WIDE_BATCH_GATHER) {
            ProcessWideBatchGather<COOPERATIVE>();
            return;
        }

        CacheIndex();
        const GatherExecutionState gatherState = PrepareGatherExecution<COOPERATIVE>();

        const uint64_t coreId = GetBlockIdx();
        const uint64_t taskStep = blockDim_ == 0 ? 1 : blockDim_;

        if (strategy_ == STRATEGY_BATCH_GATHER_NARROW) {
            ProcessBatchGatherTasks(coreId, taskStep, gatherState);
            return;
        }

        for (uint64_t taskId = coreId; taskId < taskNum_; taskId += taskStep) {
            ProcessTask(taskId, gatherState.actualLayers, gatherState.layerOverflow);
        }
    }

private:
    using GatherPipelineQueue = TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>;
    struct GatherExecutionState {
        uint32_t actualLayers{0};
        bool layerOverflow{false};
        bool useBatchGather{false};
    };

    template <bool COOPERATIVE>
    __aicore__ inline GatherExecutionState PrepareGatherExecution()
    {
        GatherExecutionState state;
        if (strategy_ == STRATEGY_GATHER_NARROW || strategy_ == STRATEGY_BATCH_GATHER_NARROW ||
            strategy_ == STRATEGY_GATHER_NARROW_DB) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                if constexpr (COOPERATIVE) {
                    state.layerOverflow = !BuildCooperativeLayerOffsets(state.actualLayers);
                } else {
                    state.layerOverflow = !BuildGatherLayerOffsets(state.actualLayers);
                }
            }
        }
        if (strategy_ == STRATEGY_BATCH_GATHER_NARROW) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                constexpr uint32_t gatherElements = VECTOR_BYTES / sizeof(T);
                constexpr uint32_t offsetElements = VECTOR_BYTES / sizeof(uint32_t);
                const uint64_t offsetCalls = (axisSize_ + offsetElements - 1) / offsetElements;
                const uint64_t dataChunks = (axisSize_ + gatherElements - 1) / gatherElements;
                const uint64_t batchedVectorCalls =
                    static_cast<uint64_t>(state.actualLayers) * (offsetCalls + 2ULL * dataChunks);
                state.useBatchGather = !state.layerOverflow && batchedVectorCalls < indexSize_;
                if (state.useBatchGather) {
                    PrepareBatchGatherConstants();
                }
            }
        }
        return state;
    }

    __aicore__ inline void ProcessBatchGatherTasks(uint64_t coreId, uint64_t taskStep,
        const GatherExecutionState& state)
    {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                      IsSameType<T, int32_t>::value) {
            const uint64_t coreOuter = outerSize_ / taskStep;
            const uint64_t extraOuter = outerSize_ - coreOuter * taskStep;
            const uint64_t outerStart = coreId * coreOuter + MinU64(coreId, extraOuter);
            const uint64_t outerCount = coreOuter + static_cast<uint64_t>(coreId < extraOuter);
            const uint64_t fullBatchCount = outerCount / outerBatch_;
            uint64_t batchStart = outerStart;
            for (uint64_t batch = 0; batch < fullBatchCount; ++batch) {
                ProcessBatchGatherTask(batchStart, static_cast<uint32_t>(outerBatch_), state);
                batchStart += outerBatch_;
            }
            const uint32_t tailCount =
                static_cast<uint32_t>(outerCount - fullBatchCount * outerBatch_);
            if (tailCount != 0) {
                ProcessBatchGatherTask(batchStart, tailCount, state);
            }
        }
    }

    __aicore__ inline void ProcessBatchGatherTask(uint64_t batchStart, uint32_t batchCount,
        const GatherExecutionState& state)
    {
        if (state.useBatchGather) {
            ProcessBatchGatherNarrowTask(batchStart, batchCount, state.actualLayers);
        } else {
            ProcessBatchGatherFallbackTask(batchStart, batchCount);
        }
    }

    __aicore__ inline void ProcessTask(uint64_t taskId, uint32_t actualLayers, bool layerOverflow)
    {
        if (strategy_ == STRATEGY_INT8_AXIS) {
            if constexpr (IsSameType<T, int8_t>::value) {
                const uint64_t innerGroup = taskId % innerTileNum_;
                const uint64_t axisTask = taskId / innerTileNum_;
                const uint64_t outputAxis = axisTask % axisSize_;
                const uint64_t outer = axisTask / axisSize_;
                const uint64_t innerStart = innerGroup * innerTile_;
                const uint32_t innerCount = static_cast<uint32_t>(
                    MinU64(innerTile_, innerSize_ - innerStart));
                ProcessInt8AxisTask(outer, outputAxis, innerStart, innerCount);
            }
            return;
        }

        if (strategy_ == STRATEGY_UB_MIXED || strategy_ == STRATEGY_UB_MIXED_DB) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                const uint64_t baseOuterCount = outerSize_ - mixedOuterCount_;
                const uint64_t baseTaskCount = baseOuterCount * innerTileNum_;
                uint64_t outer = 0;
                uint64_t innerStart = 0;
                uint32_t innerCount = 0;
                if (taskId < baseTaskCount) {
                    const uint64_t innerGroup = taskId % innerTileNum_;
                    outer = taskId / innerTileNum_;
                    innerStart = innerGroup * innerTile_;
                    innerCount = static_cast<uint32_t>(MinU64(innerTile_, innerSize_ - innerStart));
                } else {
                    const uint64_t mixedTaskId = taskId - baseTaskCount;
                    const uint64_t innerGroup = mixedTaskId % mixedInnerTileNum_;
                    outer = baseOuterCount + mixedTaskId / mixedInnerTileNum_;
                    innerStart = innerGroup * mixedInnerTile_;
                    innerCount = static_cast<uint32_t>(MinU64(mixedInnerTile_, innerSize_ - innerStart));
                }
                ProcessMixedResidentTask(outer, innerStart, innerCount);
            }
            return;
        }

        const uint64_t outerGroup = taskId / innerTileNum_;
        const uint64_t innerGroup = taskId - outerGroup * innerTileNum_;
        const uint64_t outerStart = outerGroup * outerTile_;
        const uint64_t innerStart = innerGroup * innerTile_;
        const uint32_t outerCount = static_cast<uint32_t>(MinU64(outerTile_, outerSize_ - outerStart));
        const uint32_t innerCount = static_cast<uint32_t>(MinU64(innerTile_, innerSize_ - innerStart));

        if (strategy_ == STRATEGY_ATOMIC) {
            ProcessAtomicTask(outerStart, outerCount, innerStart, innerCount, sourceChunk_, outerBatch_);
        } else if (strategy_ == STRATEGY_UB) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                ProcessResidentTask(outerStart, outerCount, innerStart, innerCount);
            }
        } else if (strategy_ == STRATEGY_NARROW_UB) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                ProcessNarrowMaskedTask<false>(outerStart, outerCount);
            }
        } else if (strategy_ == STRATEGY_GATHER_NARROW || strategy_ == STRATEGY_GATHER_NARROW_DB) {
            if constexpr (IsSameType<T, half>::value || IsSameType<T, float>::value ||
                          IsSameType<T, int32_t>::value) {
                const uint64_t gatherVectorCalls =
                    2ULL * static_cast<uint64_t>(actualLayers) * outerCount;
                if (layerOverflow || gatherVectorCalls >= indexSize_) {
                    ProcessNarrowFallbackTask(outerStart, outerCount);
                } else if (strategy_ == STRATEGY_GATHER_NARROW_DB) {
                    ProcessGatherNarrowDoubleBufferTask(outerStart, outerCount, actualLayers);
                } else {
                    ProcessGatherNarrowTask(outerStart, outerCount, actualLayers);
                }
            }
        } else if (strategy_ == STRATEGY_BF16_RESIDENT) {
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ProcessBf16ResidentTask(outerStart, outerCount, innerStart, innerCount);
            }
        } else if (strategy_ == STRATEGY_BF16_SOURCE_DB) {
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ProcessBf16SourceDoubleBufferTask(outerStart, outerCount, innerStart, innerCount);
            }
        } else if (strategy_ == STRATEGY_INT8_RESIDENT) {
            if constexpr (IsSameType<T, int8_t>::value) {
                ProcessInt8ResidentTask(outerStart, outerCount, innerStart, innerCount);
            }
        }
    }

    __aicore__ inline static uint64_t MinU64(uint64_t lhs, uint64_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline static uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
    {
        return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
    }

    __aicore__ inline static uint32_t AlignBytes(uint64_t bytes)
    {
        return static_cast<uint32_t>((bytes + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES * UB_BLOCK_BYTES);
    }

    __aicore__ inline static uint32_t AlignGatherPipelineBytes(uint64_t bytes)
    {
        return static_cast<uint32_t>((bytes + GATHER_PIPELINE_ALIGNMENT_BYTES - 1) /
            GATHER_PIPELINE_ALIGNMENT_BYTES * GATHER_PIPELINE_ALIGNMENT_BYTES);
    }

    __aicore__ inline static uint64_t AlignWorkspaceBytes(uint64_t bytes)
    {
        return (bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
    }

    template <HardEvent EVENT>
    __aicore__ inline void Sync()
    {
        const event_t eventId = static_cast<event_t>(pipe_->FetchEventID(EVENT));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }

    // 宽 Gather 与普通路径的 UB 生命周期完全不同，集中在这里选择对应布局。
    // 该函数只负责 buffer 规划，不参与运行时数据流，便于后续独立调整容量模型。
    __aicore__ inline void InitWideBuffers()
    {
        useTransposeBlockGather_ = CanUseTransposeBlockGather();
        if (useTransposeBlockGather_) {
            if constexpr (IsSameType<T, half>::value) {
                InitTransposeBlockGatherBuffers();
            } else if constexpr (IsSameType<T, float>::value) {
                InitFp32TransposeBlockGatherBuffers();
            }
        } else {
            InitWideBatchGatherBuffers();
        }
    }

    // 按 strategy 建立标准路径的 buffer。所有尺寸公式与原始 Init 保持一致，
    // 仅把策略分支从入口移出，避免初始化逻辑和运行时调度交叉增长。
    __aicore__ inline void InitStandardBuffers()
    {
        const uint32_t indexBytes = strategy_ == STRATEGY_INT8_AXIS ?
            static_cast<uint32_t>((indexSize_ * sizeof(int32_t) + VECTOR_BYTES - 1) /
                VECTOR_BYTES * VECTOR_BYTES) :
            AlignBytes(indexSize_ * sizeof(int32_t));
        pipe_->InitBuffer(indexBuf_, indexBytes);

        const uint32_t maxRowBytes = AlignBytes(innerTile_ * sizeof(T));
        switch (strategy_) {
            case STRATEGY_ATOMIC:
                pipe_->InitBuffer(dataBuf_, sourceChunk_ * maxRowBytes);
                break;
            case STRATEGY_UB:
                pipe_->InitBuffer(dataBuf_, axisSize_ * maxRowBytes);
                pipe_->InitBuffer(sourceBuf_, sourceChunk_ * maxRowBytes);
                break;
            case STRATEGY_UB_MIXED:
            case STRATEGY_UB_MIXED_DB:
                pipe_->InitBuffer(residentBuf_, residentBufferBytes_);
                break;
            case STRATEGY_GATHER_NARROW:
            case STRATEGY_GATHER_NARROW_DB:
                InitGatherNarrowBuffers();
                break;
            case STRATEGY_BATCH_GATHER_NARROW:
                InitBatchGatherBuffers();
                break;
            case STRATEGY_BF16_RESIDENT:
            case STRATEGY_BF16_SOURCE_DB:
                InitBf16Buffers(maxRowBytes);
                break;
            case STRATEGY_INT8_RESIDENT:
                InitInt8ResidentBuffers(maxRowBytes);
                break;
            case STRATEGY_INT8_AXIS:
                InitInt8AxisBuffers(maxRowBytes);
                break;
            default:
                InitDefaultBuffers();
                break;
        }
    }

    __aicore__ inline void InitGatherNarrowBuffers()
    {
        const uint32_t accumulatorRowBytes = gatherUbPaddedAxis_ != 0 ?
            AlignBytes(static_cast<uint64_t>(gatherUbPaddedAxis_) * sizeof(T)) :
            AlignBytes(axisSize_ * sizeof(T));
        const uint32_t counterRowBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t layerOffsetRowBytes = gatherUbPaddedAxis_ != 0 ?
            gatherUbPaddedAxis_ * sizeof(uint32_t) : counterRowBytes;
        const uint32_t sourceRowBytes = AlignBytes(indexSize_ * sizeof(T));
        const uint32_t sourceStorageBytes = sourceRowBytes + UB_BLOCK_BYTES;
        const uint32_t sourceBufferBytes =
            sourceStorageBytes > VECTOR_BYTES ? sourceStorageBytes : VECTOR_BYTES;
        pipe_->InitBuffer(dataBuf_, accumulatorRowBytes);
        pipe_->InitBuffer(counterBuf_, counterRowBytes);
        pipe_->InitBuffer(layerOffsetBuf_, sourceChunk_ * layerOffsetRowBytes);
        pipe_->InitBuffer(denseBuf_, accumulatorRowBytes);
        if (strategy_ != STRATEGY_GATHER_NARROW_DB) {
            pipe_->InitBuffer(sourceBuf_, sourceBufferBytes);
            return;
        }
        gatherSourceOffsetBytes_ = AlignGatherPipelineBytes(accumulatorRowBytes) +
            GATHER_PIPELINE_SOURCE_SKEW_BYTES;
        if constexpr (IsSameType<T, float>::value) {
            gatherDenseOffsetBytes_ = AlignGatherPipelineBytes(
                gatherSourceOffsetBytes_ + sourceStorageBytes) +
                GATHER_PIPELINE_FP32_DENSE_SKEW_BYTES;
        } else {
            gatherDenseOffsetBytes_ = AlignGatherPipelineBytes(
                gatherSourceOffsetBytes_ + sourceStorageBytes) + GATHER_PIPELINE_DENSE_SKEW_BYTES;
        }
        const uint32_t slotBytes = AlignGatherPipelineBytes(
            gatherDenseOffsetBytes_ + accumulatorRowBytes) + GATHER_PIPELINE_TAIL_BYTES;
        pipe_->InitBuffer(gatherPipelineFirstQueue_, 1, slotBytes);
        pipe_->InitBuffer(gatherPipelineSecondQueue_, 1, slotBytes);
    }

    __aicore__ inline void InitBatchGatherBuffers()
    {
        const uint32_t batchRows = static_cast<uint32_t>(outerBatch_);
        const uint32_t paddedAxisElements = AlignBytes(axisSize_ * sizeof(T)) / sizeof(T);
        const uint32_t sourcePitchBytes = AlignBytes(indexSize_ * sizeof(T)) + UB_BLOCK_BYTES;
        const uint32_t counterRowBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t layerOffsetRowBytes = paddedAxisElements * sizeof(uint32_t);
        const uint32_t batchOffsetRowBytes = VECTOR_BYTES / sizeof(T) * sizeof(uint32_t);
        const uint32_t dataSkewBytes = batchRows % 2 == 0 ? VECTOR_BYTES : 0;
        pipe_->InitBuffer(dataBuf_, batchRows * VECTOR_BYTES + dataSkewBytes);
        if constexpr (IsSameType<T, half>::value) {
            pipe_->InitBuffer(batchOffsetBuf_, batchRows * batchOffsetRowBytes);
        }
        pipe_->InitBuffer(denseBuf_, batchRows * VECTOR_BYTES);
        pipe_->InitBuffer(sourceBuf_, batchRows * sourcePitchBytes);
        pipe_->InitBuffer(counterBuf_, counterRowBytes);
        pipe_->InitBuffer(layerOffsetBuf_, sourceChunk_ * layerOffsetRowBytes);
        pipe_->InitBuffer(rowBaseBuf_, batchRows * UB_BLOCK_BYTES);
        if constexpr (!IsSameType<T, half>::value) {
            pipe_->InitBuffer(batchOffsetBuf_, batchRows * batchOffsetRowBytes);
        }
    }

    __aicore__ inline void InitBf16Buffers(uint32_t maxRowBytes)
    {
        const uint32_t fp32RowBytes = AlignBytes(innerTile_ * sizeof(float));
        pipe_->InitBuffer(dataBuf_, axisSize_ * maxRowBytes);
        if (strategy_ == STRATEGY_BF16_SOURCE_DB) {
            pipe_->InitBuffer(bf16SourceQueue_, 2, sourceChunk_ * maxRowBytes);
        } else {
            pipe_->InitBuffer(sourceBuf_, sourceChunk_ * maxRowBytes);
        }
        pipe_->InitBuffer(denseBuf_, sourceChunk_ * fp32RowBytes + VECTOR_BYTES);
        pipe_->InitBuffer(workBuf_, BF16_INDEPENDENT_WORK_ROWS * fp32RowBytes);
    }

    __aicore__ inline void InitInt8ResidentBuffers(uint32_t maxRowBytes)
    {
        const uint32_t halfRowBytes = AlignBytes(innerTile_ * sizeof(half));
        const uint32_t int32RowBytes = AlignBytes(innerTile_ * sizeof(int32_t));
        const uint64_t maxRows = axisSize_ > sourceChunk_ ? axisSize_ : sourceChunk_;
        pipe_->InitBuffer(dataBuf_, maxRows * maxRowBytes);
        pipe_->InitBuffer(sourceBuf_, maxRows * halfRowBytes);
        pipe_->InitBuffer(denseBuf_, axisSize_ * int32RowBytes + VECTOR_BYTES +
            sourceChunk_ * int32RowBytes);
        pipe_->InitBuffer(workBuf_, axisSize_ * int32RowBytes);
    }

    __aicore__ inline void InitInt8AxisBuffers(uint32_t maxRowBytes)
    {
        const uint32_t halfRowBytes = AlignBytes(innerTile_ * sizeof(half));
        const uint32_t int32RowBytes = AlignBytes(innerTile_ * sizeof(int32_t));
        const uint32_t compareElements = static_cast<uint32_t>(
            (indexSize_ + VECTOR_BYTES / sizeof(int32_t) - 1) /
            (VECTOR_BYTES / sizeof(int32_t)) * (VECTOR_BYTES / sizeof(int32_t)));
        const uint32_t compareMaskBytes = AlignBytes(compareElements / 8);
        const uint32_t maskGapBytes = compareMaskBytes > VECTOR_BYTES ?
            compareMaskBytes : VECTOR_BYTES;
        pipe_->InitBuffer(dataBuf_, maxRowBytes);
        pipe_->InitBuffer(sourceBuf_, halfRowBytes);
        pipe_->InitBuffer(denseBuf_, 2 * int32RowBytes + maskGapBytes);
        pipe_->InitBuffer(workBuf_, int32RowBytes);
    }

    __aicore__ inline void InitDefaultBuffers()
    {
        const uint32_t accumulatorRowBytes = AlignBytes(axisSize_ * sizeof(T));
        pipe_->InitBuffer(dataBuf_, outerTile_ * accumulatorRowBytes);
        pipe_->InitBuffer(sourceBuf_, outerTile_ * VECTOR_BYTES);
    }

    // 16 行转置路径的资格仅由已有 tiling、数据类型和指令/地址表边界推导，
    // 不依赖公开 shape 或 index 内容。
    __aicore__ inline bool CanUseTransposeBlockGather() const
    {
        if constexpr (!WIDE_BATCH_GATHER) {
            return false;
        }
        if constexpr (IsSameType<T, half>::value) {
            constexpr uint32_t transposeRows = 16;
            constexpr uint32_t gatherElements = VECTOR_BYTES / sizeof(T);
            constexpr uint32_t axisPadElements = 2 * gatherElements;
            return transposeBlockGatherMode_ == TRANSPOSE_BLOCK_GATHER_FP16 && cooperative_ &&
                strategy_ == STRATEGY_GATHER_BATCH_WIDE && indexSize_ >= transposeRows &&
                indexSize_ % transposeRows == 0 && indexSize_ / transposeRows <= 255 &&
                axisSize_ > gatherElements && axisSize_ <= axisPadElements && sourceChunk_ != 0;
        }
        if constexpr (IsSameType<T, float>::value) {
            const uint32_t axisPad = AlignBytes(axisSize_ * sizeof(float)) / sizeof(float);
            return transposeBlockGatherMode_ == TRANSPOSE_BLOCK_GATHER_FP32 && cooperative_ &&
                strategy_ == STRATEGY_GATHER_NARROW_DB && indexSize_ >= FP32_TRANSPOSE_ACTIVE_ROWS &&
                transposeSourceChunk_ >= FP32_TRANSPOSE_COL_ALIGNMENT && transposeLayerCap_ != 0 &&
                transposePayloadLayerCap_ != 0 && transposePayloadLayerCap_ <= transposeLayerCap_ &&
                (transposeOffsetBytes_ == sizeof(uint16_t) ||
                    transposeOffsetBytes_ == sizeof(uint32_t)) &&
                axisPad >= 4 * (VECTOR_BYTES / sizeof(float)) &&
                axisPad / FP32_TRANSPOSE_COL_ALIGNMENT <= 255 &&
                transposeSourceChunk_ / FP32_TRANSPOSE_COL_ALIGNMENT <= 255;
        }
        return false;
    }

    // FP16 宽 Gather 的所有 UB 分配只存在于 <half, true> 实例。常规模板不进入
    // 此函数，因而维持既有 dtype 的 buffer 顺序与指令流不变。
    __aicore__ inline void InitWideBatchGatherBuffers()
    {
        const uint32_t indexBytes = AlignBytes(indexSize_ * sizeof(int32_t));
        constexpr uint32_t wideElements = 2 * VECTOR_BYTES / sizeof(T);
        const uint32_t wideOffsetBytes = static_cast<uint32_t>(
            outerBatch_ * static_cast<uint64_t>(wideElements) * sizeof(uint32_t));
        pipe_->InitBuffer(indexBuf_, indexBytes < wideOffsetBytes ? wideOffsetBytes : indexBytes);

        const uint32_t batchRows = static_cast<uint32_t>(outerBatch_);
        const uint32_t sourcePitchBytes = AlignBytes(indexSize_ * sizeof(T)) + UB_BLOCK_BYTES;
        const uint32_t counterRowBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t layerOffsetRowBytes = wideElements * sizeof(uint32_t);
        constexpr uint32_t wideDataRowBytes = 2 * VECTOR_BYTES;

        // 宽行由 16 个 32B block 构成，固定错开一条 Vector 行以避免 accumulator/dense
        // 同 lane 落入同一 DAV_2201 bank group。
        pipe_->InitBuffer(dataBuf_, batchRows * wideDataRowBytes + VECTOR_BYTES);
        pipe_->InitBuffer(denseBuf_, batchRows * wideDataRowBytes);
        pipe_->InitBuffer(sourceBuf_, batchRows * sourcePitchBytes);
        pipe_->InitBuffer(counterBuf_, counterRowBytes);
        pipe_->InitBuffer(layerOffsetBuf_, sourceChunk_ * layerOffsetRowBytes);
        pipe_->InitBuffer(rowBaseBuf_, batchRows * UB_BLOCK_BYTES);
    }

    // FP32 source record 在转置后保留 16 个 float（64B），低、高两个 8 行半区
    // 分别对应同一批的前后 outer 行；raw source 因此缓存完整 16 行。
    // dataBuf_、denseBuf_ 分别承载物理布局下的输出与累加器。
    __aicore__ inline void InitFp32TransposeBlockGatherBuffers()
    {
        const uint32_t axisPadElements = AlignBytes(axisSize_ * sizeof(float)) / sizeof(float);
        const uint32_t indexBytes = AlignBytes(indexSize_ * sizeof(int32_t));
        const uint32_t chunkCount = static_cast<uint32_t>(
            CeilDivU64(indexSize_, transposeSourceChunk_));
        const uint32_t headerBytes = (chunkCount + 1) * UB_BLOCK_BYTES;
        const uint32_t counterBytes = axisPadElements * sizeof(uint32_t);
        // metadata counter 与 payload offset 表不重叠，payload 仅保留当前分组的
        // 直接 byte-offset 行。
        const uint32_t offsetBytes = transposeOffsetBytes_ == sizeof(uint16_t) ?
            sizeof(uint16_t) : sizeof(uint32_t);
        const uint32_t layerBytes = transposePayloadLayerCap_ * axisPadElements * offsetBytes;
        const uint32_t accumulatorBytes = FP32_TRANSPOSE_ROWS * axisPadElements * sizeof(float);
        const uint32_t sourceStageElements = FP32_TRANSPOSE_ACTIVE_ROWS * transposeSourceChunk_;
        const uint32_t sourceBlockElements = FP32_TRANSPOSE_ROWS * transposeSourceChunk_;
        const uint32_t sourceStageBytes = sourceStageElements * sizeof(float);
        const uint32_t sourceTransposeBytes =
            sourceBlockElements * sizeof(float) + FP32_TRANSPOSE_ROWS * sizeof(float);
        const uint32_t sourcePayloadBytes = sourceStageBytes + sourceTransposeBytes;
        const uint32_t accumulatorPlaneBytes =
            FP32_TRANSPOSE_COL_ALIGNMENT * axisPadElements * sizeof(float);
        const uint32_t expandedOffsetBytes =
            transposePayloadLayerCap_ * axisPadElements * sizeof(uint32_t);
        const uint32_t dataBufferBytes = accumulatorPlaneBytes + expandedOffsetBytes;

        // metadata 阶段 counterBuf_ 是完整 axis counter，发布后才复用为 header。
        // 二者生命周期相邻而非重叠，容量取较大值，禁止 header 小缓冲被 counter
        // 初始化越界覆盖。
        pipe_->InitBuffer(counterBuf_, counterBytes > headerBytes ? counterBytes : headerBytes);
        pipe_->InitBuffer(layerOffsetBuf_, layerBytes);
        // dataBuf_ 的前半区承载单个 Gather 输出平面，尾部仅保留当前
        // payload 分组所需的 expanded offset scratch；不再固定占满 16 行。
        pipe_->InitBuffer(dataBuf_, dataBufferBytes);
        // 完整 index 只在 metadata 阶段存活；发布屏障之后 payload 覆盖 sourceBuf。
        // 两阶段严格串行，故这里按最大值而不是叠加值分配 UB。
        pipe_->InitBuffer(sourceBuf_, indexBytes > sourcePayloadBytes ? indexBytes : sourcePayloadBytes);
        pipe_->InitBuffer(denseBuf_, accumulatorBytes);
    }

    // 转置块路径复用现有 cooperative metadata：dataBuf_/denseBuf_ 在建表阶段分别
    // 作为 raw/rank scratch，之后 dataBuf_/sourceBuf_ 分区为双缓冲 self/source、转置结果和累加器。
    __aicore__ inline void InitTransposeBlockGatherBuffers()
    {
        constexpr uint32_t transposeRows = 16;
        constexpr uint32_t axisPadElements = 2 * VECTOR_BYTES / sizeof(T);
        const uint32_t indexBytes = AlignBytes(indexSize_ * sizeof(int32_t));
        const uint32_t counterBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t rawScratchBytes = static_cast<uint32_t>(axisSize_ * COOP_RAW_SLOT_BYTES);
        const uint32_t accumulatorBytes = axisPadElements * transposeRows * sizeof(T);
        // dataBuf_ 在 payload 阶段承载双缓冲 self (2 * accumulatorBytes)。
        const uint32_t prefetchSelfBytes = 2 * accumulatorBytes;
        const uint32_t sourceStageBytes = static_cast<uint32_t>(transposeRows * indexSize_ * sizeof(T));
        const uint32_t sourceTransposeBytes = sourceStageBytes + UB_BLOCK_BYTES;
        // sourceBuf_ 承载双缓冲 sourceRows (2 * sourceStageBytes) + 转置 sourceBlocks + accumulatorBlocks。
        const uint32_t sourceWorkBytes = 2 * sourceStageBytes + sourceTransposeBytes + accumulatorBytes;
        const uint32_t layerRowBytes = axisPadElements * sizeof(uint32_t);

        pipe_->InitBuffer(indexBuf_, indexBytes);
        pipe_->InitBuffer(counterBuf_, counterBytes);
        pipe_->InitBuffer(layerOffsetBuf_, sourceChunk_ * layerRowBytes);
        pipe_->InitBuffer(dataBuf_, rawScratchBytes < prefetchSelfBytes ?
            prefetchSelfBytes : rawScratchBytes);
        pipe_->InitBuffer(sourceBuf_, sourceWorkBytes);
        pipe_->InitBuffer(denseBuf_, accumulatorBytes);
    }

    template <bool COOPERATIVE>
    __aicore__ inline void ProcessWideBatchGather()
    {
        if constexpr (IsSameType<T, float>::value) {
            // 正常路径由 source-chunk owner 直接搬入自己的 index 分片；只有
            // occurrence 溢出、需要原子回退时才完整缓存 index，避免每个 core
            // 重复搬运同一份 metadata。
            ProcessFp32TransposeBlockGather();
            return;
        }

        CacheIndex();
        uint32_t actualLayers = 0;
        bool layerOverflow = false;
        if constexpr (COOPERATIVE) {
            layerOverflow = !BuildCooperativeLayerOffsets(actualLayers);
        } else {
            layerOverflow = !BuildGatherLayerOffsets(actualLayers);
        }

        if (useTransposeBlockGather_) {
            ProcessTransposeBlockGather(actualLayers, layerOverflow);
            return;
        }

        constexpr uint32_t gatherElements = VECTOR_BYTES / sizeof(T);
        constexpr uint32_t wideElements = 2 * gatherElements;
        constexpr uint32_t offsetElements = VECTOR_BYTES / sizeof(uint32_t);
        const uint64_t offsetCalls = (wideElements + offsetElements - 1) / offsetElements;
        const uint64_t batchedVectorCalls =
            static_cast<uint64_t>(actualLayers) * (offsetCalls + 2ULL);
        const bool useBatchGather = !layerOverflow && batchedVectorCalls < indexSize_;
        if (useBatchGather) {
            PrepareBatchGatherConstants();
        }

        const uint64_t coreId = GetBlockIdx();
        const uint64_t taskStep = blockDim_ == 0 ? 1 : blockDim_;
        const uint64_t coreOuter = outerSize_ / taskStep;
        const uint64_t extraOuter = outerSize_ - coreOuter * taskStep;
        const uint64_t outerStart = coreId * coreOuter + MinU64(coreId, extraOuter);
        const uint64_t outerCount = coreOuter + static_cast<uint64_t>(coreId < extraOuter);
        const uint64_t fullBatchCount = outerCount / outerBatch_;

        uint64_t batchStart = outerStart;
        for (uint64_t batch = 0; batch < fullBatchCount; ++batch) {
            if (useBatchGather) {
                ProcessBatchGatherWideTask(
                    batchStart, static_cast<uint32_t>(outerBatch_), actualLayers);
            } else {
                ProcessBatchGatherFallbackTask(
                    batchStart, static_cast<uint32_t>(outerBatch_));
            }
            batchStart += outerBatch_;
        }

        const uint32_t tailCount = static_cast<uint32_t>(outerCount - fullBatchCount * outerBatch_);
        if (tailCount != 0) {
            if (useBatchGather) {
                ProcessBatchGatherWideTask(batchStart, tailCount, actualLayers);
            } else {
                ProcessBatchGatherFallbackTask(batchStart, tailCount);
            }
        }
    }

    __aicore__ inline void CacheIndex()
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const uint32_t indexValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(int32_t));
        DataCopyExtParams copyParams{1, indexValidBytes, 0, 0, 0};
        if (strategy_ == STRATEGY_INT8_AXIS) {
            const uint32_t compareElements = static_cast<uint32_t>(
                (indexSize_ + VECTOR_BYTES / sizeof(int32_t) - 1) /
                (VECTOR_BYTES / sizeof(int32_t)) * (VECTOR_BYTES / sizeof(int32_t)));
            const uint32_t copyElements = AlignBytes(indexSize_ * sizeof(int32_t)) / sizeof(int32_t);
            Duplicate<int32_t>(indexLocal, -1, static_cast<int32_t>(compareElements));
            Sync<HardEvent::V_MTE2>();
            DataCopyPadExtParams<int32_t> padParams{
                true, 0, static_cast<uint8_t>(copyElements - indexSize_), -1};
            DataCopyPad(indexLocal, indexGm_, copyParams, padParams);
            Sync<HardEvent::MTE2_V>();
        } else {
            // index 的合法长度不要求是 32B 整数倍。显式填充尾部，确保后续
            // Scalar/Gather 只访问定义好的 UB 数据，而不是依赖 DMA 尾块行为。
            const uint32_t alignedIndexElements = AlignBytes(indexValidBytes) / sizeof(int32_t);
            DataCopyPadExtParams<int32_t> padParams{
                true, 0, static_cast<uint8_t>(alignedIndexElements - indexSize_), 0};
            DataCopyPad(indexLocal, indexGm_, copyParams, padParams);
            Sync<HardEvent::MTE2_S>();
        }
    }

    // FP32 转置路径仅在 metadata 溢出回退时完整缓存 index。正常路径由各 metadata
    // owner 直接搬入自己的连续 chunk，避免 blockDim 个 core 重复访问同一源数据。
    __aicore__ inline void CacheFp32TransposeIndex()
    {
        LocalTensor<int32_t> indexLocal = sourceBuf_.Get<int32_t>();
        const uint32_t indexValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(int32_t));
        const uint32_t alignedIndexElements = AlignBytes(indexValidBytes) / sizeof(int32_t);
        DataCopyExtParams copyParams{1, indexValidBytes, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{
            true, 0, static_cast<uint8_t>(alignedIndexElements - indexSize_), 0};
        DataCopyPad(indexLocal, indexGm_, copyParams, padParams);
        Sync<HardEvent::MTE2_S>();
    }

    __aicore__ inline static bool AppendGatherOffset(__ubuf__ int32_t* indexAddr,
        __ubuf__ uint32_t* counterAddr, __ubuf__ uint32_t* layerOffsetAddr, uint32_t m,
        uint32_t sourceByteOffset, uint32_t layerOffsetRowElements, uint32_t layerCap)
    {
        const uint32_t outputIndex = static_cast<uint32_t>(indexAddr[m]);
        const uint32_t layer = counterAddr[outputIndex];
        if (layer >= layerCap) {
            return false;
        }
        layerOffsetAddr[layer * layerOffsetRowElements + outputIndex] = sourceByteOffset;
        counterAddr[outputIndex] = layer + 1;
        return true;
    }

    __aicore__ inline bool BuildGatherLayerOffsets(uint32_t& actualLayers)
    {
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        LocalTensor<uint32_t> counterLocal = counterBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> layerOffsetLocal = layerOffsetBuf_.Get<uint32_t>();
        const uint32_t counterRowBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t counterRowElements = counterRowBytes / sizeof(uint32_t);
        const uint32_t layerOffsetRowElements = WIDE_BATCH_GATHER ?
            2 * VECTOR_BYTES / sizeof(T) :
            (strategy_ == STRATEGY_BATCH_GATHER_NARROW ?
                AlignBytes(axisSize_ * sizeof(T)) / sizeof(T) : counterRowElements);
        const uint32_t sourceSentinelByteOffset = useTransposeBlockGather_ ?
            static_cast<uint32_t>(indexSize_ * UB_BLOCK_BYTES) : AlignBytes(indexSize_ * sizeof(T));
        const uint32_t layerOffsetElements = sourceChunk_ * layerOffsetRowElements;

        Duplicate<uint32_t>(counterLocal, 0, static_cast<int32_t>(counterRowElements));
        Duplicate<uint32_t>(layerOffsetLocal, sourceSentinelByteOffset,
            static_cast<int32_t>(layerOffsetElements));
        Sync<HardEvent::V_S>();

        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        __ubuf__ uint32_t* counterAddr = (__ubuf__ uint32_t*)counterLocal.GetPhyAddr();
        __ubuf__ uint32_t* layerOffsetAddr = (__ubuf__ uint32_t*)layerOffsetLocal.GetPhyAddr();
        const uint32_t indexSize = static_cast<uint32_t>(indexSize_);
        const uint32_t elementBytes = sizeof(T);
        bool overflow = false;
        uint32_t m = 0;
        const uint32_t mainEnd = indexSize / 8 * 8;
        for (; m < mainEnd; m += 8) {
            const uint32_t sourceByteOffset = m * elementBytes;
            if (!AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m, sourceByteOffset,
                    layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 1,
                    sourceByteOffset + elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 2,
                    sourceByteOffset + 2 * elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 3,
                    sourceByteOffset + 3 * elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 4,
                    sourceByteOffset + 4 * elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 5,
                    sourceByteOffset + 5 * elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 6,
                    sourceByteOffset + 6 * elementBytes, layerOffsetRowElements, sourceChunk_) ||
                !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m + 7,
                    sourceByteOffset + 7 * elementBytes, layerOffsetRowElements, sourceChunk_)) {
                overflow = true;
                break;
            }
        }
        for (; !overflow && m < indexSize; ++m) {
            overflow = !AppendGatherOffset(indexAddr, counterAddr, layerOffsetAddr, m, m * elementBytes,
                layerOffsetRowElements, sourceChunk_);
        }

        actualLayers = 0;
        if (!overflow) {
            const uint32_t axisSize = static_cast<uint32_t>(axisSize_);
            for (uint32_t axis = 0; axis < axisSize; ++axis) {
                actualLayers = actualLayers < counterAddr[axis] ? counterAddr[axis] : actualLayers;
            }
        }
        Sync<HardEvent::S_V>();
        return !overflow;
    }

    __aicore__ inline void CompactCooperativeLayerOffsets(
        uint64_t rawOffsetBytes, uint32_t actualLayers)
    {
        const uint32_t counterRowBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t counterRowElements = counterRowBytes / sizeof(uint32_t);
        const uint32_t layerOffsetRowElements = WIDE_BATCH_GATHER ?
            2 * VECTOR_BYTES / sizeof(T) :
            (strategy_ == STRATEGY_BATCH_GATHER_NARROW ?
                AlignBytes(axisSize_ * sizeof(T)) / sizeof(T) : counterRowElements);
        const bool useUbPaddedRepeat = gatherUbPaddedAxis_ > layerOffsetRowElements;
        const uint32_t paddedRowElements = useUbPaddedRepeat ? gatherUbPaddedAxis_ : layerOffsetRowElements;
        const uint32_t sourceSentinelByteOffset = useTransposeBlockGather_ ?
            static_cast<uint32_t>(indexSize_ * UB_BLOCK_BYTES) : AlignBytes(indexSize_ * sizeof(T));
        const uint32_t rawRowElements = static_cast<uint32_t>(axisSize_) *
            COOP_RAW_SLOT_BYTES / sizeof(int32_t);
        const uint64_t rawBytes = static_cast<uint64_t>(coopLayerCap_) * axisSize_ *
            COOP_RAW_SLOT_BYTES;
        const uint64_t compactOffsetBytes = AlignWorkspaceBytes(rawOffsetBytes + rawBytes);

        LocalTensor<int32_t> layerTableLocal = layerOffsetBuf_.Get<int32_t>();
        LocalTensor<int32_t> compactStageLocal;
        LocalTensor<int32_t> rawLocal;
        LocalTensor<uint32_t> gatherOffsetLocal;
        if (strategy_ == STRATEGY_GATHER_NARROW || strategy_ == STRATEGY_GATHER_NARROW_DB) {
            rawLocal = layerTableLocal;
            gatherOffsetLocal = layerTableLocal[rawRowElements].ReinterpretCast<uint32_t>();
            compactStageLocal = layerTableLocal[rawRowElements + counterRowElements];
        } else {
            rawLocal = dataBuf_.Get<int32_t>();
            gatherOffsetLocal = counterBuf_.Get<uint32_t>();
            compactStageLocal = layerTableLocal;
        }

        ArithProgression<int32_t>(gatherOffsetLocal.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(COOP_RAW_SLOT_BYTES), static_cast<int32_t>(axisSize_));
        const uint32_t coreId = GetBlockIdx();
        for (uint32_t layer = coreId; layer < actualLayers; layer += blockDim_) {
            const uint64_t rawLayerOffset = rawOffsetBytes / sizeof(int32_t) +
                static_cast<uint64_t>(layer) * rawRowElements;
            DataCopy(rawLocal, workspaceGm_[rawLayerOffset], rawRowElements);
            Sync<HardEvent::MTE2_V>();
            if (useTransposeBlockGather_ || useUbPaddedRepeat) {
                // Gather 只写逻辑 axis，先显式定义原行对齐尾与随后扩展的尾 lane。
                Duplicate<int32_t>(compactStageLocal, static_cast<int32_t>(sourceSentinelByteOffset),
                    static_cast<int32_t>(layerOffsetRowElements));
                PipeBarrier<PIPE_V>();
            }
            Gather<int32_t>(compactStageLocal, rawLocal,
                gatherOffsetLocal, 0, static_cast<uint32_t>(axisSize_));
            Sync<HardEvent::V_MTE3>();
            DataCopy(workspaceGm_[compactOffsetBytes / sizeof(int32_t) +
                static_cast<uint64_t>(layer) * layerOffsetRowElements],
                compactStageLocal, layerOffsetRowElements);
            if (layer + blockDim_ < actualLayers) {
                Sync<HardEvent::MTE3_V>();
            }
        }
        // 最后一层没有下一轮 Vector 写入触发等待；先完成本核写回再跨核汇合。
        Sync<HardEvent::MTE3_MTE2>();
        SyncAll();

        DataCopy(layerTableLocal, workspaceGm_[compactOffsetBytes / sizeof(int32_t)],
            actualLayers * layerOffsetRowElements);
        Sync<HardEvent::MTE2_V>();
        if (useUbPaddedRepeat) {
            // 从后向前展开，避免覆盖尚未搬走的紧凑行。counterBuf 在 metadata 完成后
            // 不再承载计数，复用为一行 32B 对齐的 UB→UB staging，未新增 GM 传输。
            LocalTensor<int32_t> rowStageLocal = counterBuf_.Get<int32_t>();
            for (uint32_t layer = actualLayers; layer > 0; --layer) {
                const uint32_t row = layer - 1;
                DataCopy(rowStageLocal, layerTableLocal[row * layerOffsetRowElements],
                    layerOffsetRowElements);
                PipeBarrier<PIPE_V>();
                LocalTensor<int32_t> paddedRowLocal = layerTableLocal[row * paddedRowElements];
                DataCopy(paddedRowLocal, rowStageLocal, layerOffsetRowElements);
                Duplicate<int32_t>(paddedRowLocal[layerOffsetRowElements],
                    static_cast<int32_t>(sourceSentinelByteOffset),
                    static_cast<int32_t>(paddedRowElements - layerOffsetRowElements));
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline bool BuildCooperativeLayerOffsets(uint32_t& actualLayers)
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = blockDim_;
        const uint32_t axisSize = static_cast<uint32_t>(axisSize_);
        const uint32_t indexSize = static_cast<uint32_t>(indexSize_);
        const uint32_t countPitchBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t countRowElements = countPitchBytes / sizeof(uint32_t);
        // prefix 按消费者 core 连续存放；后续每个核可一次读回整条 axis 前缀。
        const uint32_t prefixCorePitchBytes = AlignBytes(axisSize_ * sizeof(uint32_t));
        const uint32_t prefixCorePitchElements = prefixCorePitchBytes / sizeof(uint32_t);
        const uint32_t sourceSentinelByteOffset = useTransposeBlockGather_ ?
            static_cast<uint32_t>(indexSize_ * UB_BLOCK_BYTES) : AlignBytes(indexSize_ * sizeof(T));

        const uint64_t countBytes = static_cast<uint64_t>(coreNum) * countPitchBytes;
        const uint64_t countOffsetBytes = 0;
        const uint64_t prefixOffsetBytes = AlignWorkspaceBytes(countOffsetBytes + countBytes);
        const uint64_t prefixBytes = static_cast<uint64_t>(coreNum) * prefixCorePitchBytes;
        const uint64_t coreMaxOffsetBytes = AlignWorkspaceBytes(prefixOffsetBytes + prefixBytes);
        const uint64_t coreMaxBytes = static_cast<uint64_t>(coreNum) * COOP_RAW_SLOT_BYTES;
        const uint64_t actualOffsetBytes = AlignWorkspaceBytes(coreMaxOffsetBytes + coreMaxBytes);
        const uint64_t rawOffsetBytes = AlignWorkspaceBytes(actualOffsetBytes + COOP_RAW_SLOT_BYTES);

        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        LocalTensor<uint32_t> countLocal = counterBuf_.Get<uint32_t>();
        LocalTensor<int32_t> scratchLocal = layerOffsetBuf_.Get<int32_t>();
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        __ubuf__ uint32_t* countAddr = (__ubuf__ uint32_t*)countLocal.GetPhyAddr();
        __ubuf__ int32_t* scratchAddr = (__ubuf__ int32_t*)scratchLocal.GetPhyAddr();

        const uint64_t indexPerCore = (indexSize_ + coreNum - 1) / coreNum;
        const uint64_t indexStart = MinU64(indexSize_, static_cast<uint64_t>(coreId) * indexPerCore);
        const uint64_t indexEnd = MinU64(indexSize_, indexStart + indexPerCore);
        const uint32_t localIndexCount = static_cast<uint32_t>(indexEnd - indexStart);

        const uint64_t totalRawSlots = static_cast<uint64_t>(coopLayerCap_) * axisSize_;
        const uint64_t rawSlotsPerCore = (totalRawSlots + coreNum - 1) / coreNum;
        const uint64_t rawSlotStart = MinU64(totalRawSlots,
            static_cast<uint64_t>(coreId) * rawSlotsPerCore);
        const uint64_t rawSlotEnd = MinU64(totalRawSlots, rawSlotStart + rawSlotsPerCore);
        const uint32_t localRawSlots = static_cast<uint32_t>(rawSlotEnd - rawSlotStart);
        const uint32_t localRawElements = localRawSlots * COOP_RAW_SLOT_BYTES / sizeof(int32_t);

        Duplicate<uint32_t>(countLocal, 0, static_cast<int32_t>(countRowElements));
        if (localRawElements != 0) {
            Duplicate<int32_t>(scratchLocal, static_cast<int32_t>(sourceSentinelByteOffset),
                static_cast<int32_t>(localRawElements));
        }
        Sync<HardEvent::V_S>();
        for (uint64_t m = indexStart; m < indexEnd; ++m) {
            ++countAddr[static_cast<uint32_t>(indexAddr[m])];
        }
        Sync<HardEvent::S_MTE3>();
        Sync<HardEvent::V_MTE3>();
        DataCopy(workspaceGm_[(countOffsetBytes + static_cast<uint64_t>(coreId) * countPitchBytes) /
            sizeof(int32_t)], countLocal.ReinterpretCast<int32_t>(), countRowElements);
        if (localRawElements != 0) {
            DataCopy(workspaceGm_[rawOffsetBytes / sizeof(int32_t) +
                rawSlotStart * COOP_RAW_SLOT_BYTES / sizeof(int32_t)], scratchLocal, localRawElements);
        }
        SyncAll();

        // 连续轴块归属让每个核一次读回自己的 count 子矩阵，避免按交错轴重复发射窄列 DMA。
        const uint32_t axisPerCore = (axisSize + coreNum - 1) / coreNum;
        const uint32_t axisStart = static_cast<uint32_t>(MinU64(axisSize,
            static_cast<uint64_t>(coreId) * axisPerCore));
        const uint32_t ownedAxisCount = axisStart < axisSize ?
            static_cast<uint32_t>(MinU64(axisPerCore, axisSize - axisStart)) : 0;
        const uint32_t countMatrixValidBytes = ownedAxisCount * static_cast<uint32_t>(sizeof(uint32_t));
        const uint32_t countMatrixPitchBytes = AlignBytes(countMatrixValidBytes);
        const uint32_t countMatrixPitchElements = countMatrixPitchBytes / sizeof(uint32_t);
        LocalTensor<int32_t> countMatrixLocal = scratchLocal;
        __ubuf__ int32_t* countMatrixAddr = (__ubuf__ int32_t*)countMatrixLocal.GetPhyAddr();
        DataCopyPadExtParams<int32_t> noPadParams{false, 0, 0, 0};
        uint32_t localMax = 0;
        if (ownedAxisCount != 0) {
            DataCopyExtParams countMatrixCopyParams{
                static_cast<uint16_t>(coreNum), countMatrixValidBytes,
                countPitchBytes - countMatrixValidBytes, 0, 0};
            DataCopyPad(countMatrixLocal,
                workspaceGm_[(countOffsetBytes + static_cast<uint64_t>(axisStart) * sizeof(int32_t)) /
                    sizeof(int32_t)], countMatrixCopyParams, noPadParams);
            Sync<HardEvent::MTE2_S>();
            for (uint32_t owned = 0; owned < ownedAxisCount; ++owned) {
                uint32_t running = 0;
                for (uint32_t core = 0; core < coreNum; ++core) {
                    const uint32_t count = static_cast<uint32_t>(
                        countMatrixAddr[core * countMatrixPitchElements + owned]);
                    // 覆盖已消费的 count，将矩阵原地转为 [consumer core][owned axis] prefix。
                    countMatrixAddr[core * countMatrixPitchElements + owned] =
                        static_cast<int32_t>(running);
                    running += count;
                }
                localMax = localMax < running ? running : localMax;
            }
        }

        for (uint32_t value = 0; value < COOP_RAW_SLOT_BYTES / sizeof(int32_t); ++value) {
            countAddr[value] = localMax;
        }
        Sync<HardEvent::S_MTE3>();
        if (ownedAxisCount != 0) {
            const uint32_t prefixValidBytes = ownedAxisCount * static_cast<uint32_t>(sizeof(int32_t));
            // 一条二维 MTE3 指令把 [consumer core][owned axis] 写到各 core 的连续 prefix 段。
            DataCopyExtParams prefixWriteParams{static_cast<uint16_t>(coreNum), prefixValidBytes,
                0, prefixCorePitchBytes - prefixValidBytes, 0};
            DataCopyPad(workspaceGm_[(prefixOffsetBytes +
                static_cast<uint64_t>(axisStart) * sizeof(int32_t)) / sizeof(int32_t)],
                countMatrixLocal, prefixWriteParams);
        }
        DataCopy(workspaceGm_[(coreMaxOffsetBytes +
            static_cast<uint64_t>(coreId) * COOP_RAW_SLOT_BYTES) / sizeof(int32_t)],
            countLocal.ReinterpretCast<int32_t>(), COOP_RAW_SLOT_BYTES / sizeof(int32_t));
        SyncAll();

        const uint32_t coreMaxElements = coreNum * COOP_RAW_SLOT_BYTES / sizeof(int32_t);
        DataCopy(scratchLocal, workspaceGm_[coreMaxOffsetBytes / sizeof(int32_t)], coreMaxElements);
        Sync<HardEvent::MTE2_S>();
        uint32_t globalMax = 0;
        for (uint32_t core = 0; core < coreNum; ++core) {
            const uint32_t coreMax = static_cast<uint32_t>(
                scratchAddr[core * COOP_RAW_SLOT_BYTES / sizeof(int32_t)]);
            globalMax = globalMax < coreMax ? coreMax : globalMax;
        }
        actualLayers = globalMax;
        if (actualLayers > coopLayerCap_) {
            SyncAll();
            return false;
        }

        DataCopy(scratchLocal, workspaceGm_[(prefixOffsetBytes +
            static_cast<uint64_t>(coreId) * prefixCorePitchBytes) / sizeof(int32_t)],
            prefixCorePitchElements);
        LocalTensor<int32_t> rawStageLocal = scratchLocal[prefixCorePitchElements];
        LocalTensor<uint32_t> rankLocal = denseBuf_.Get<uint32_t>();
        __ubuf__ int32_t* rawStageAddr = (__ubuf__ int32_t*)rawStageLocal.GetPhyAddr();
        __ubuf__ uint32_t* rankAddr = (__ubuf__ uint32_t*)rankLocal.GetPhyAddr();

        Duplicate<uint32_t>(countLocal, 0, static_cast<int32_t>(countRowElements));
        if (localIndexCount != 0) {
            Duplicate<int32_t>(rawStageLocal, static_cast<int32_t>(sourceSentinelByteOffset),
                static_cast<int32_t>(localIndexCount * COOP_RAW_SLOT_BYTES / sizeof(int32_t)));
        }
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
        for (uint32_t local = 0; local < localIndexCount; ++local) {
            const uint32_t m = static_cast<uint32_t>(indexStart) + local;
            const uint32_t axis = static_cast<uint32_t>(indexAddr[m]);
            const uint32_t prefix = static_cast<uint32_t>(scratchAddr[axis]);
            const uint32_t rank = prefix + countAddr[axis]++;
            rawStageAddr[local * COOP_RAW_SLOT_BYTES / sizeof(int32_t)] = static_cast<int32_t>(
                m * (useTransposeBlockGather_ ? UB_BLOCK_BYTES : sizeof(T)));
            rankAddr[local] = rank;
        }
        Sync<HardEvent::S_MTE3>();
        for (uint32_t local = 0; local < localIndexCount; ++local) {
            const uint32_t m = static_cast<uint32_t>(indexStart) + local;
            const uint32_t axis = static_cast<uint32_t>(indexAddr[m]);
            const uint32_t rank = rankAddr[local];
            DataCopy(workspaceGm_[rawOffsetBytes / sizeof(int32_t) +
                (static_cast<uint64_t>(rank) * axisSize_ + axis) *
                    COOP_RAW_SLOT_BYTES / sizeof(int32_t)],
                rawStageLocal[local * COOP_RAW_SLOT_BYTES / sizeof(int32_t)],
                COOP_RAW_SLOT_BYTES / sizeof(int32_t));
        }
        // SyncAll 只负责核心间次序；compact 前必须保证所有本核 raw slot 已写入 GM。
        Sync<HardEvent::MTE3_MTE2>();
        SyncAll();

        CompactCooperativeLayerOffsets(rawOffsetBytes, actualLayers);
        return true;
    }

    // 将 GM 中等距的多行搬入 UB。UB 行距由 DataCopyPad 自动补齐到 32B。
    __aicore__ inline void CopyRowsIn(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t gmOffset,
        uint32_t rowCount, uint32_t validBytes, uint64_t gmRowStride, uint32_t localRowElements)
    {
        const uint64_t gmGapBytes = (gmRowStride - validBytes / sizeof(T)) * sizeof(T);
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        if (gmGapBytes <= MAX_COPY_STRIDE) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount), validBytes, static_cast<uint32_t>(gmGapBytes), 0, 0};
            DataCopyPad(dst, src[gmOffset], copyParams, padParams);
            return;
        }

        // 极大张量的 GM stride 超过 Ext 参数范围时退化为逐行 DMA，保证泛化正确性。
        DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        for (uint32_t row = 0; row < rowCount; ++row) {
            DataCopyPad(dst[row * localRowElements], src[gmOffset + row * gmRowStride], copyParams, padParams);
        }
    }

    // 当 GM 行距不是 32B 整数倍而有效行已对齐时，按偶/奇行拆成两条二维 DMA。
    // 每条 DMA 的跨行 GM stride 恰为两个物理行，固定了该 DMA 内的行首相位；UB
    // 仍按原逻辑行号落位，后续 Add 的数据与顺序完全不变。
    __aicore__ inline void CopyResidentSourceRowsIn(LocalTensor<T> dst, GlobalTensor<T> src,
        uint64_t gmOffset, uint32_t rowCount, uint32_t validBytes, uint64_t gmRowStride,
        uint32_t localRowElements)
    {
        const uint64_t elementBytes = sizeof(T);
        const bool validPitchProduct = gmRowStride <= UINT64_MAX / elementBytes;
        const bool isAlignedPayload = validBytes % UB_BLOCK_BYTES == 0;
        const bool hasUnalignedGmPitch = validPitchProduct &&
            (gmRowStride * elementBytes) % UB_BLOCK_BYTES != 0;
        if (residentDoubleBuffer_ == 0 || rowCount < 2 || !isAlignedPayload || !hasUnalignedGmPitch ||
            gmRowStride > UINT64_MAX / 2) {
            CopyRowsIn(dst, src, gmOffset, rowCount, validBytes, gmRowStride, localRowElements);
            return;
        }

        const uint64_t validElements = validBytes / elementBytes;
        const uint64_t pairedRowStride = gmRowStride * 2;
        if (pairedRowStride < validElements ||
            pairedRowStride - validElements > UINT64_MAX / elementBytes) {
            CopyRowsIn(dst, src, gmOffset, rowCount, validBytes, gmRowStride, localRowElements);
            return;
        }
        const uint64_t pairedGapBytes = (pairedRowStride - validElements) * elementBytes;
        if (pairedGapBytes > MAX_COPY_STRIDE) {
            CopyRowsIn(dst, src, gmOffset, rowCount, validBytes, gmRowStride, localRowElements);
            return;
        }

        const uint32_t evenRows = (rowCount + 1) / 2;
        const uint32_t oddRows = rowCount / 2;
        const uint32_t localGapBlocks = validBytes / UB_BLOCK_BYTES;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams evenParams{
            static_cast<uint16_t>(evenRows), validBytes, static_cast<uint32_t>(pairedGapBytes),
            localGapBlocks, 0};
        DataCopyPad(dst, src[gmOffset], evenParams, padParams);
        if (oddRows != 0) {
            DataCopyExtParams oddParams{
                static_cast<uint16_t>(oddRows), validBytes, static_cast<uint32_t>(pairedGapBytes),
                localGapBlocks, 0};
            DataCopyPad(dst[localRowElements], src[gmOffset + gmRowStride], oddParams, padParams);
        }
    }

    // 将 UB 中 32B 对齐存放的多行搬回 GM，可由当前 Atomic 配置决定是否原子累加。
    __aicore__ inline void CopyRowsOut(GlobalTensor<T> dst, LocalTensor<T> src, uint64_t gmOffset,
        uint32_t rowCount, uint32_t validBytes, uint64_t gmRowStride, uint32_t localRowElements)
    {
        const uint64_t gmGapBytes = (gmRowStride - validBytes / sizeof(T)) * sizeof(T);
        if (gmGapBytes <= MAX_COPY_STRIDE) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount), validBytes, 0, static_cast<uint32_t>(gmGapBytes), 0};
            DataCopyPad(dst[gmOffset], src, copyParams);
            return;
        }

        DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        for (uint32_t row = 0; row < rowCount; ++row) {
            DataCopyPad(dst[gmOffset + row * gmRowStride], src[row * localRowElements], copyParams);
        }
    }

    __aicore__ inline void CopyRowsInFixedPitch(LocalTensor<T> dst, GlobalTensor<T> src,
        uint64_t gmOffset, uint32_t rowCount, uint32_t validBytes, uint64_t gmRowStride,
        uint32_t localRowBytes)
    {
        const uint32_t alignedValidBytes = AlignBytes(validBytes);
        const uint32_t localRowElements = localRowBytes / sizeof(T);
        const uint32_t dstStride = (localRowBytes - alignedValidBytes) / UB_BLOCK_BYTES;
        const uint64_t gmGapBytes = (gmRowStride - validBytes / sizeof(T)) * sizeof(T);
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        if (gmGapBytes <= MAX_COPY_STRIDE) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount), validBytes, static_cast<uint32_t>(gmGapBytes), dstStride, 0};
            DataCopyPad(dst, src[gmOffset], copyParams, padParams);
            return;
        }
        DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        for (uint32_t row = 0; row < rowCount; ++row) {
            DataCopyPad(dst[row * localRowElements], src[gmOffset + row * gmRowStride], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyRowsOutFixedPitch(GlobalTensor<T> dst, LocalTensor<T> src,
        uint64_t gmOffset, uint32_t rowCount, uint32_t validBytes, uint64_t gmRowStride,
        uint32_t localRowBytes)
    {
        const uint32_t alignedValidBytes = AlignBytes(validBytes);
        const uint32_t localRowElements = localRowBytes / sizeof(T);
        const uint32_t srcStride = (localRowBytes - alignedValidBytes) / UB_BLOCK_BYTES;
        const uint64_t gmGapBytes = (gmRowStride - validBytes / sizeof(T)) * sizeof(T);
        if (gmGapBytes <= MAX_COPY_STRIDE) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount), validBytes, srcStride, static_cast<uint32_t>(gmGapBytes), 0};
            DataCopyPad(dst[gmOffset], src, copyParams);
            return;
        }
        DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        for (uint32_t row = 0; row < rowCount; ++row) {
            DataCopyPad(dst[gmOffset + row * gmRowStride], src[row * localRowElements], copyParams);
        }
    }

    __aicore__ inline void ProcessAtomicTask(uint64_t outerStart, uint32_t outerCount,
        uint64_t innerStart, uint32_t innerCount, uint32_t sourceChunk, uint64_t outerBatch)
    {
        LocalTensor<T> dataLocal = dataBuf_.Get<T>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t rowBytes = AlignBytes(validBytes);
        const uint32_t localRowElements = rowBytes / sizeof(T);

        // 每个 task 独占 outer x inner 区域，先完整初始化 output，再开启原子累加。
        const uint64_t totalSelfRows = static_cast<uint64_t>(outerCount) * axisSize_;
        for (uint64_t rowBase = 0; rowBase < totalSelfRows; rowBase += sourceChunk) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(sourceChunk, totalSelfRows - rowBase));
            const uint64_t gmOffset = ((outerStart * axisSize_ + rowBase) * innerSize_) + innerStart;
            CopyRowsIn(dataLocal, selfGm_, gmOffset, rows, validBytes, innerSize_, localRowElements);
            Sync<HardEvent::MTE2_MTE3>();
            CopyRowsOut(outputGm_, dataLocal, gmOffset, rows, validBytes, innerSize_, localRowElements);
            // 同一 staging buffer 下一批会被 MTE2 覆盖，必须等 MTE3 完成。
            Sync<HardEvent::MTE3_MTE2>();
        }

        PipeBarrier<PIPE_MTE3>();
        SetAtomicAdd<T>();
        for (uint64_t m = 0; m < indexSize_; ++m) {
            const int32_t outputIndex = indexLocal.GetValue(m);
            for (uint64_t outerBase = 0; outerBase < outerCount; outerBase += outerBatch) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(outerBatch, outerCount - outerBase));
                const uint64_t sourceOffset =
                    ((outerStart + outerBase) * indexSize_ + m) * innerSize_ + innerStart;
                CopyRowsIn(dataLocal, sourceGm_, sourceOffset, rows, validBytes,
                    indexSize_ * innerSize_, localRowElements);
                Sync<HardEvent::MTE2_MTE3>();

                const uint64_t outputOffset =
                    ((outerStart + outerBase) * axisSize_ + static_cast<uint64_t>(outputIndex)) * innerSize_ +
                    innerStart;
                CopyRowsOut(outputGm_, dataLocal, outputOffset, rows, validBytes,
                    axisSize_ * innerSize_, localRowElements);
                // 保证 staging 可复用，也严格保持重复 index 的 m 顺序。
                Sync<HardEvent::MTE3_MTE2>();
            }
        }
        PipeBarrier<PIPE_MTE3>();
        SetAtomicNone();
    }

    __aicore__ inline void ProcessResidentTask(uint64_t outerStart, uint32_t outerCount,
        uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> accLocal = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t rowBytes = AlignBytes(validBytes);
        const uint32_t localRowElements = rowBytes / sizeof(T);
        const uint32_t maxRowBytes = AlignBytes(innerTile_ * sizeof(T));
        const uint64_t sourceCapacityBytes = static_cast<uint64_t>(sourceChunk_) * maxRowBytes;
        const uint32_t taskSourceChunk = static_cast<uint32_t>(MinU64(
            MinU64(indexSize_, MAX_COPY_ROWS), sourceCapacityBytes / rowBytes));

        // Host 保证驻留策略每个 task 只处理一个 outer；保留循环以防尾部参数变化。
        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsIn(accLocal[rowBase * localRowElements], selfGm_, selfOffset, rows, validBytes,
                    innerSize_, localRowElements);
            }

            for (uint64_t mBase = 0; mBase < indexSize_; mBase += taskSourceChunk) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(taskSourceChunk, indexSize_ - mBase));
                const uint64_t sourceOffset = (outer * indexSize_ + mBase) * innerSize_ + innerStart;
                CopyResidentSourceRowsIn(
                    sourceLocal, sourceGm_, sourceOffset, rows, validBytes, innerSize_, localRowElements);
                Sync<HardEvent::MTE2_V>();

                for (uint32_t row = 0; row < rows; ++row) {
                    const int32_t outputIndex = indexLocal.GetValue(mBase + row);
                    const uint64_t accOffset = static_cast<uint64_t>(outputIndex) * localRowElements;
                    Add<T>(accLocal[accOffset], accLocal[accOffset], sourceLocal[row * localRowElements], innerCount);
                }
                // 下一批 source 会覆盖同一 UB，等待本批 Vector 读取完成。
                Sync<HardEvent::V_MTE2>();
            }

            Sync<HardEvent::V_MTE3>();
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t outputOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsOut(outputGm_, accLocal[rowBase * localRowElements], outputOffset, rows, validBytes,
                    innerSize_, localRowElements);
            }
            // 下一个 outer/task 会复用 accumulator，等待写回完成。
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    // 混合切分将同一块 UB 按当前 task 的累加器大小动态划分，剩余区域全部用于 source panel。
    // 这样窄 tile 不会被宽 tile 的静态 accumulator 占用 UB，仍保持每个元素的 m 顺序。
    __aicore__ inline void ProcessMixedResidentTask(uint64_t outer, uint64_t innerStart,
        uint32_t innerCount)
    {
        LocalTensor<T> residentLocal = residentBuf_.Get<T>();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t rowBytes = AlignBytes(validBytes);
        const uint32_t localRowElements = rowBytes / sizeof(T);
        const uint64_t accumulatorBytes = axisSize_ * rowBytes;
        const uint64_t sourceCapacityBytes = residentBufferBytes_ - accumulatorBytes;
        const uint64_t slotCapacityBytes = residentDoubleBuffer_ != 0 ? sourceCapacityBytes / 2 :
            sourceCapacityBytes;
        const uint32_t taskSourceChunk = static_cast<uint32_t>(MinU64(
            MinU64(indexSize_, MAX_COPY_ROWS), slotCapacityBytes / rowBytes));
        if (taskSourceChunk == 0) {
            return;
        }
        LocalTensor<T> accLocal = residentLocal;
        const uint32_t sourceStartElements = static_cast<uint32_t>(accumulatorBytes / sizeof(T));
        const uint32_t slotElements = taskSourceChunk * localRowElements;
        LocalTensor<T> sourceLocal = residentLocal[sourceStartElements];
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();

        for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
            const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
            CopyRowsIn(accLocal[rowBase * localRowElements], selfGm_, selfOffset, rows, validBytes,
                innerSize_, localRowElements);
        }

        if (residentDoubleBuffer_ == 0) {
            for (uint64_t mBase = 0; mBase < indexSize_; mBase += taskSourceChunk) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(taskSourceChunk, indexSize_ - mBase));
                const uint64_t sourceOffset = (outer * indexSize_ + mBase) * innerSize_ + innerStart;
                CopyRowsIn(sourceLocal, sourceGm_, sourceOffset, rows, validBytes, innerSize_, localRowElements);
                Sync<HardEvent::MTE2_V>();

                for (uint32_t row = 0; row < rows; ++row) {
                    const int32_t outputIndex = indexLocal.GetValue(mBase + row);
                    const uint64_t accOffset = static_cast<uint64_t>(outputIndex) * localRowElements;
                    Add<T>(accLocal[accOffset], accLocal[accOffset], sourceLocal[row * localRowElements], innerCount);
                }
                Sync<HardEvent::V_MTE2>();
            }
        } else {
            // 两个 source 槽轮换：当前 panel 做 Vector 累加时，MTE2 同时搬入下一 panel。
            // 每次复用槽位前仍等待上一轮 Vector，保持 source 数据生命周期安全。
            const uint64_t panelCount = CeilDivU64(indexSize_, taskSourceChunk);
            uint64_t mBase = 0;
            uint32_t rows = static_cast<uint32_t>(MinU64(taskSourceChunk, indexSize_));
            uint32_t slot = 0;
            LocalTensor<T> currentSource = sourceLocal;
            const uint64_t slotOffsetElements = static_cast<uint64_t>(slotElements);
            const uint64_t firstSourceOffset = (outer * indexSize_) * innerSize_ + innerStart;
            CopyResidentSourceRowsIn(currentSource, sourceGm_, firstSourceOffset, rows, validBytes,
                innerSize_, localRowElements);
            Sync<HardEvent::MTE2_V>();

            for (uint64_t panel = 0; panel < panelCount; ++panel) {
                const uint64_t nextBase = mBase + rows;
                const uint32_t nextSlot = slot ^ 1U;
                if (nextBase < indexSize_) {
                    const uint32_t nextRows = static_cast<uint32_t>(MinU64(taskSourceChunk, indexSize_ - nextBase));
                    LocalTensor<T> nextSource = residentLocal[sourceStartElements +
                        static_cast<uint64_t>(nextSlot) * slotOffsetElements];
                    const uint64_t nextSourceOffset =
                        (outer * indexSize_ + nextBase) * innerSize_ + innerStart;
                    CopyResidentSourceRowsIn(nextSource, sourceGm_, nextSourceOffset, nextRows, validBytes,
                        innerSize_, localRowElements);
                }

                for (uint32_t row = 0; row < rows; ++row) {
                    const int32_t outputIndex = indexLocal.GetValue(mBase + row);
                    const uint64_t accOffset = static_cast<uint64_t>(outputIndex) * localRowElements;
                    Add<T>(accLocal[accOffset], accLocal[accOffset], currentSource[row * localRowElements], innerCount);
                }
                Sync<HardEvent::V_MTE2>();
                if (nextBase >= indexSize_) {
                    break;
                }
                Sync<HardEvent::MTE2_V>();
                mBase = nextBase;
                rows = static_cast<uint32_t>(MinU64(taskSourceChunk, indexSize_ - mBase));
                slot = nextSlot;
                currentSource = residentLocal[sourceStartElements +
                    static_cast<uint64_t>(slot) * slotOffsetElements];
            }
        }

        Sync<HardEvent::V_MTE3>();
        for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
            const uint64_t outputOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
            CopyRowsOut(outputGm_, accLocal[rowBase * localRowElements], outputOffset, rows, validBytes,
                innerSize_, localRowElements);
        }
        Sync<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void AddNarrowMasked(LocalTensor<T> accumulator, LocalTensor<T> source,
        uint32_t lane, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
    {
        if constexpr (sizeof(T) == sizeof(half)) {
            uint64_t mask[2] = {0, 0};
            if (lane < 64) {
                mask[0] = 1ULL << lane;
            } else {
                mask[1] = 1ULL << (lane - 64);
            }
            Add<T>(accumulator, accumulator, source, mask, repeatTime, repeatParams);
        } else {
            uint64_t mask[1] = {1ULL << lane};
            Add<T>(accumulator, accumulator, source, mask, repeatTime, repeatParams);
        }
    }

    __aicore__ inline void ProcessGatherNarrowTask(
        uint64_t outerStart, uint32_t outerCount, uint32_t actualLayers)
    {
        LocalTensor<T> accLocal = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<T> denseLocal = denseBuf_.Get<T>();
        LocalTensor<uint32_t> layerOffsetLocal = layerOffsetBuf_.Get<uint32_t>();
        const uint32_t accumulatorValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
        const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));
        const uint32_t sourceRowBytes = AlignBytes(sourceValidBytes);
        const uint32_t sourceSentinelElements = sourceRowBytes / sizeof(T);
        const uint32_t counterRowElements = AlignBytes(axisSize_ * sizeof(uint32_t)) / sizeof(uint32_t);
        constexpr uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);

        Duplicate<T>(sourceLocal[sourceSentinelElements], static_cast<T>(0), blockElements);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams selfCopyParams{1, accumulatorValidBytes, 0, 0, 0};
        DataCopyExtParams sourceCopyParams{1, sourceValidBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams outputCopyParams{1, accumulatorValidBytes, 0, 0, 0};

        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            DataCopyPad(accLocal, selfGm_[outer * axisSize_], selfCopyParams, padParams);
            DataCopyPad(sourceLocal, sourceGm_[outer * indexSize_], sourceCopyParams, padParams);
            Sync<HardEvent::MTE2_V>();

            for (uint32_t layer = 0; layer < actualLayers; ++layer) {
                Gather(denseLocal, sourceLocal, layerOffsetLocal[layer * counterRowElements], 0,
                    static_cast<uint32_t>(axisSize_));
                PipeBarrier<PIPE_V>();
                Add<T>(accLocal, accLocal, denseLocal, static_cast<int32_t>(axisSize_));
                if (layer + 1 < actualLayers) {
                    PipeBarrier<PIPE_V>();
                }
            }

            Sync<HardEvent::V_MTE3>();
            DataCopyPad(outputGm_[outer * axisSize_], accLocal, outputCopyParams);
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void CopyGatherNarrowPipelineIn(GatherPipelineQueue& queue, uint64_t outer)
    {
        LocalTensor<T> payloadLocal = queue.AllocTensor<T>();
        LocalTensor<T> sourceLocal = payloadLocal[gatherSourceOffsetBytes_ / sizeof(T)];
        const uint32_t accumulatorValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
        const uint32_t accumulatorAlignedElements = AlignBytes(accumulatorValidBytes) / sizeof(T);
        const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));
        DataCopyExtParams selfCopyParams{1, accumulatorValidBytes, 0, 0, 0};
        DataCopyExtParams sourceCopyParams{1, sourceValidBytes, 0, 0, 0};
        const bool useUbPaddedRepeat = gatherUbPaddedAxis_ > accumulatorAlignedElements;
        DataCopyPadExtParams<T> selfPadParams{useUbPaddedRepeat, 0,
            static_cast<uint8_t>(useUbPaddedRepeat ? accumulatorAlignedElements - axisSize_ : 0),
            static_cast<T>(0)};
        DataCopyPadExtParams<T> sourcePadParams{
            true, 0, static_cast<uint8_t>(UB_BLOCK_BYTES / sizeof(T)), static_cast<T>(0)};
        DataCopyPad(payloadLocal, selfGm_[outer * axisSize_], selfCopyParams, selfPadParams);
        DataCopyPad(sourceLocal, sourceGm_[outer * indexSize_], sourceCopyParams, sourcePadParams);
        queue.template EnQue<TPosition::GM, TPosition::VECIN>(payloadLocal);
    }

    __aicore__ inline void FinishGatherNarrowPipelinePayload(GatherPipelineQueue& queue,
        LocalTensor<T> payloadLocal, uint64_t outer, uint32_t actualLayers)
    {
        LocalTensor<uint32_t> layerOffsetLocal = layerOffsetBuf_.Get<uint32_t>();
        const uint32_t counterRowElements = AlignBytes(axisSize_ * sizeof(uint32_t)) / sizeof(uint32_t);
        const uint32_t accumulatorAlignedElements = AlignBytes(axisSize_ * sizeof(T)) / sizeof(T);
        const bool useUbPaddedRepeat = gatherUbPaddedAxis_ > accumulatorAlignedElements;
        const uint32_t tableRowElements = useUbPaddedRepeat ? gatherUbPaddedAxis_ : counterRowElements;
        const uint32_t gatherCount = useUbPaddedRepeat ? gatherUbPaddedAxis_ :
            static_cast<uint32_t>(axisSize_);
        const uint32_t accumulatorValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
        DataCopyExtParams outputCopyParams{1, accumulatorValidBytes, 0, 0, 0};

        LocalTensor<T> accLocal = payloadLocal;
        LocalTensor<T> sourceLocal = payloadLocal[gatherSourceOffsetBytes_ / sizeof(T)];
        LocalTensor<T> denseLocal = payloadLocal[gatherDenseOffsetBytes_ / sizeof(T)];
        if (useUbPaddedRepeat) {
            // 对齐首块中的 lane 已由 DMA pad 清零；剩余完整 32B 块在 V 中补零。
            Duplicate<T>(accLocal[accumulatorAlignedElements], static_cast<T>(0),
                static_cast<int32_t>(gatherUbPaddedAxis_ - accumulatorAlignedElements));
            PipeBarrier<PIPE_V>();
        }
        if constexpr (IsSameType<T, float>::value) {
            // 当 axis 已补齐到完整 256B repeat 时，offset 表的每一行也是完整
            // repeat。利用 layerOffsetBuf_ 中 compact 表后的空闲区域暂存多行
            // dense，令一次 Gather 生成多个 occurrence layer；Add 仍逐层执行，
            // 保持累加的 RAW 依赖、层序和 m 顺序不变，不增加 GM 传输。
            constexpr uint32_t elementsPerRepeat = VECTOR_BYTES / sizeof(float);
            const uint32_t repeatsPerLayer =
                (gatherCount + elementsPerRepeat - 1) / elementsPerRepeat;
            const uint32_t tableCapacityRows = sourceChunk_;
            const uint32_t freeRows = tableCapacityRows > actualLayers ?
                tableCapacityRows - actualLayers : 0;
            const uint32_t fusedRows = freeRows > 1 ?
                (freeRows < actualLayers ? freeRows : actualLayers) : 1;
            LocalTensor<float> fusedDense = layerOffsetBuf_.Get<float>()[
                static_cast<uint64_t>(actualLayers) * tableRowElements];
            // 只有 host 已为每层 offset 补齐完整 repeat 时，下一层才会紧接在
            // repeat 边界；逻辑轴宽不足一个 repeat 的泛化尾部必须逐层回退。
            if (useUbPaddedRepeat && fusedRows > 1 && fusedRows * repeatsPerLayer <= 255) {
                for (uint32_t layer = 0; layer < actualLayers; layer += fusedRows) {
                    const uint32_t groupRows =
                        (actualLayers - layer) < fusedRows ? actualLayers - layer : fusedRows;
                    const uint32_t repeatCount = groupRows * repeatsPerLayer;
                    Gather(fusedDense, sourceLocal,
                        layerOffsetLocal[layer * tableRowElements], 0,
                        static_cast<uint64_t>(elementsPerRepeat),
                        static_cast<uint8_t>(repeatCount),
                        static_cast<uint16_t>(UB_BLOCK_BYTES / sizeof(float)));
                    for (uint32_t groupLayer = 0; groupLayer < groupRows; ++groupLayer) {
                        Add<float>(accLocal, accLocal,
                            fusedDense[static_cast<uint64_t>(groupLayer) * tableRowElements],
                            gatherCount);
                    }
                }
            } else {
                // 非完整 repeat 或 UB 无余量时保留原逐层路径，覆盖所有泛化尾部。
                for (uint32_t layer = 0; layer < actualLayers; ++layer) {
                    Gather(denseLocal, sourceLocal, layerOffsetLocal[layer * tableRowElements], 0,
                        gatherCount);
                    Add<T>(accLocal, accLocal, denseLocal, gatherCount);
                }
            }
        } else {
            for (uint32_t layer = 0; layer < actualLayers; ++layer) {
                Gather(denseLocal, sourceLocal, layerOffsetLocal[layer * counterRowElements], 0,
                    static_cast<uint32_t>(axisSize_));
                PipeBarrier<PIPE_V>();
                Add<T>(accLocal, accLocal, denseLocal, static_cast<int32_t>(axisSize_));
                if (layer + 1 < actualLayers) {
                    PipeBarrier<PIPE_V>();
                }
            }
        }

        Sync<HardEvent::V_MTE3>();
        DataCopyPad(outputGm_[outer * axisSize_], accLocal, outputCopyParams);
        queue.FreeTensor(payloadLocal);
    }

    __aicore__ inline void ProcessGatherNarrowDoubleBufferTask(
        uint64_t outerStart, uint32_t outerCount, uint32_t actualLayers)
    {
        CopyGatherNarrowPipelineIn(gatherPipelineFirstQueue_, outerStart);
        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            if ((outerOffset & 1U) == 0) {
                LocalTensor<T> payloadLocal = gatherPipelineFirstQueue_.template DeQue<
                    TPosition::GM, TPosition::VECIN, T>();
                if (outerOffset + 1 < outerCount) {
                    CopyGatherNarrowPipelineIn(gatherPipelineSecondQueue_, outer + 1);
                }
                FinishGatherNarrowPipelinePayload(
                    gatherPipelineFirstQueue_, payloadLocal, outer, actualLayers);
            } else {
                LocalTensor<T> payloadLocal = gatherPipelineSecondQueue_.template DeQue<
                    TPosition::GM, TPosition::VECIN, T>();
                if (outerOffset + 1 < outerCount) {
                    CopyGatherNarrowPipelineIn(gatherPipelineFirstQueue_, outer + 1);
                }
                FinishGatherNarrowPipelinePayload(
                    gatherPipelineSecondQueue_, payloadLocal, outer, actualLayers);
            }
        }
    }

    __aicore__ inline void ProcessBatchGatherNarrowTask(
        uint64_t outerStart, uint32_t outerCount, uint32_t actualLayers)
    {
        LocalTensor<T> accLocal = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<T> denseLocal = denseBuf_.Get<T>();
        LocalTensor<int32_t> layerOffsetLocal = layerOffsetBuf_.Get<int32_t>();
        LocalTensor<int32_t> rowBaseLocal = rowBaseBuf_.Get<int32_t>();
        LocalTensor<int32_t> batchOffsetLocal = batchOffsetBuf_.Get<int32_t>();
        LocalTensor<uint32_t> batchOffsetUintLocal = batchOffsetBuf_.Get<uint32_t>();

        constexpr uint32_t gatherElements = VECTOR_BYTES / sizeof(T);
        constexpr uint32_t dataRowBlocks = VECTOR_BYTES / UB_BLOCK_BYTES;
        constexpr uint32_t offsetElements = VECTOR_BYTES / sizeof(uint32_t);
        constexpr uint32_t scratchRowBlocks = gatherElements * sizeof(uint32_t) / UB_BLOCK_BYTES;

        const uint32_t paddedAxisElements = AlignBytes(axisSize_ * sizeof(T)) / sizeof(T);
        const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));

        DataCopyExtParams sourceCopyParams{
            static_cast<uint16_t>(outerCount), sourceValidBytes, 0, 1, 0};
        DataCopyPadExtParams<T> sourcePadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(sourceLocal, sourceGm_[outerStart * indexSize_], sourceCopyParams, sourcePadParams);

        const BinaryRepeatParams offsetRepeatParams{1, 1, 0,
            static_cast<uint8_t>(scratchRowBlocks), 0, 1};
        const BinaryRepeatParams dataRepeatParams{1, 1, 1,
            static_cast<uint8_t>(dataRowBlocks), static_cast<uint8_t>(dataRowBlocks),
            static_cast<uint8_t>(dataRowBlocks)};

        for (uint64_t axisBase = 0; axisBase < axisSize_; axisBase += gatherElements) {
            const uint32_t chunkCount =
                static_cast<uint32_t>(MinU64(gatherElements, axisSize_ - axisBase));
            const uint32_t chunkValidBytes = chunkCount * sizeof(T);
            const uint32_t chunkAlignedBytes = AlignBytes(chunkValidBytes);
            const uint32_t selfSrcGap = static_cast<uint32_t>((axisSize_ - chunkCount) * sizeof(T));
            const uint32_t selfDstGap = (VECTOR_BYTES - chunkAlignedBytes) / UB_BLOCK_BYTES;
            DataCopyExtParams selfCopyParams{
                static_cast<uint16_t>(outerCount), chunkValidBytes, selfSrcGap, selfDstGap, 0};
            DataCopyPadExtParams<T> selfPadParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(accLocal, selfGm_[outerStart * axisSize_ + axisBase], selfCopyParams, selfPadParams);
            Sync<HardEvent::MTE2_V>();

            for (uint32_t layer = 0; layer < actualLayers; ++layer) {
                for (uint32_t part = 0; part < chunkCount; part += offsetElements) {
                    const uint32_t partCount =
                        static_cast<uint32_t>(MinU64(offsetElements, chunkCount - part));
                    const uint64_t layerOffset =
                        static_cast<uint64_t>(layer) * paddedAxisElements + axisBase + part;
                    Add<int32_t>(batchOffsetLocal[part], layerOffsetLocal[layerOffset], rowBaseLocal,
                        static_cast<uint64_t>(partCount), static_cast<uint8_t>(outerCount), offsetRepeatParams);
                }
                PipeBarrier<PIPE_V>();

                Gather<T>(denseLocal, sourceLocal, batchOffsetUintLocal, 0,
                    static_cast<uint64_t>(chunkCount), static_cast<uint8_t>(outerCount),
                    static_cast<uint16_t>(dataRowBlocks));
                PipeBarrier<PIPE_V>();
                Add<T>(accLocal, accLocal, denseLocal, static_cast<uint64_t>(chunkCount),
                    static_cast<uint8_t>(outerCount), dataRepeatParams);
                PipeBarrier<PIPE_V>();
            }

            Sync<HardEvent::V_MTE3>();
            const uint32_t outputSrcGap = (VECTOR_BYTES - chunkAlignedBytes) / UB_BLOCK_BYTES;
            const uint32_t outputDstGap = static_cast<uint32_t>((axisSize_ - chunkCount) * sizeof(T));
            DataCopyExtParams outputCopyParams{
                static_cast<uint16_t>(outerCount), chunkValidBytes, outputSrcGap, outputDstGap, 0};
            DataCopyPad(outputGm_[outerStart * axisSize_ + axisBase], accLocal, outputCopyParams);
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    // 两个 128-lane Gather repeat 连续组织为一行。offset 表为 [outer][256]，
    // 其尾部预置指向每行 source sentinel；因此 axis 未覆盖的 lane 对输出无影响，
    // 同时每层只需要一组 Gather/Add 依赖链。
    __aicore__ inline void ProcessBatchGatherWideTask(
        uint64_t outerStart, uint32_t outerCount, uint32_t actualLayers)
    {
        LocalTensor<T> accLocal = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<T> denseLocal = denseBuf_.Get<T>();
        LocalTensor<int32_t> layerOffsetLocal = layerOffsetBuf_.Get<int32_t>();
        LocalTensor<int32_t> rowBaseLocal = rowBaseBuf_.Get<int32_t>();
        // BuildGatherLayerOffsets 完成后 indexBuf 不再被读取，复用其更大的物理区间。
        LocalTensor<int32_t> batchOffsetLocal = indexBuf_.Get<int32_t>();
        LocalTensor<uint32_t> batchOffsetUintLocal = indexBuf_.Get<uint32_t>();

        constexpr uint32_t gatherElements = VECTOR_BYTES / sizeof(T);
        constexpr uint32_t wideElements = 2 * gatherElements;
        constexpr uint32_t dataRowBlocks = VECTOR_BYTES / UB_BLOCK_BYTES;
        constexpr uint32_t wideDataRowBytes = 2 * VECTOR_BYTES;
        constexpr uint32_t offsetElements = VECTOR_BYTES / sizeof(uint32_t);
        constexpr uint32_t wideOffsetRowBlocks = wideElements * sizeof(uint32_t) / UB_BLOCK_BYTES;

        const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));
        DataCopyExtParams sourceCopyParams{
            static_cast<uint16_t>(outerCount), sourceValidBytes, 0, 1, 0};
        DataCopyPadExtParams<T> sourcePadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(sourceLocal, sourceGm_[outerStart * indexSize_], sourceCopyParams, sourcePadParams);

        const BinaryRepeatParams offsetRepeatParams{1, 1, 0,
            static_cast<uint8_t>(wideOffsetRowBlocks), 0, 1};
        const BinaryRepeatParams dataRepeatParams{1, 1, 1,
            static_cast<uint8_t>(dataRowBlocks), static_cast<uint8_t>(dataRowBlocks),
            static_cast<uint8_t>(dataRowBlocks)};

        const uint32_t axisValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
        const uint32_t axisAlignedBytes = AlignBytes(axisValidBytes);
        const uint32_t rowPaddingBlocks = (wideDataRowBytes - axisAlignedBytes) / UB_BLOCK_BYTES;
        DataCopyExtParams selfCopyParams{
            static_cast<uint16_t>(outerCount), axisValidBytes, 0, rowPaddingBlocks, 0};
        DataCopyPadExtParams<T> selfPadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(accLocal, selfGm_[outerStart * axisSize_], selfCopyParams, selfPadParams);
        Sync<HardEvent::MTE2_V>();

        for (uint32_t layer = 0; layer < actualLayers; ++layer) {
            for (uint32_t part = 0; part < wideElements; part += offsetElements) {
                const uint64_t layerOffset = static_cast<uint64_t>(layer) * wideElements + part;
                Add<int32_t>(batchOffsetLocal[part], layerOffsetLocal[layerOffset], rowBaseLocal,
                    static_cast<uint64_t>(offsetElements), static_cast<uint8_t>(outerCount), offsetRepeatParams);
            }
            PipeBarrier<PIPE_V>();

            const uint8_t repeatTime = static_cast<uint8_t>(outerCount * 2);
            Gather<T>(denseLocal, sourceLocal, batchOffsetUintLocal, 0,
                static_cast<uint64_t>(gatherElements), repeatTime, static_cast<uint16_t>(dataRowBlocks));
            PipeBarrier<PIPE_V>();
            Add<T>(accLocal, accLocal, denseLocal, static_cast<uint64_t>(gatherElements),
                repeatTime, dataRepeatParams);
            PipeBarrier<PIPE_V>();
        }

        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams outputCopyParams{
            static_cast<uint16_t>(outerCount), axisValidBytes, rowPaddingBlocks, 0, 0};
        DataCopyPad(outputGm_[outerStart * axisSize_], accLocal, outputCopyParams);
        Sync<HardEvent::MTE3_MTE2>();
    }

    // 用显式 VA 地址表避开跨 batch 复用 LocalTensor 地址描述时的状态歧义。每次
    // TransDataTo5HD 完成后等待全部流水，确保下一次改写同一组 VA 寄存器前已退休。
    __aicore__ inline void Transpose16RowsToBlocks(LocalTensor<T> dst, LocalTensor<T> src,
        uint32_t rowElements)
    {
        uint64_t srcList[16];
        uint64_t dstList[16];
        for (uint32_t row = 0; row < 16; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(src[row * rowElements].GetPhyAddr());
            dstList[row] = reinterpret_cast<uint64_t>(dst[row * 16].GetPhyAddr());
        }
        const uint8_t repeats = static_cast<uint8_t>(rowElements / 16);
        const TransDataTo5HDParams params{false, false, repeats,
            static_cast<uint16_t>(repeats == 1 ? 0 : 16), static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
        TransDataTo5HD<T>(dstList, srcList, params);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
    }

    __aicore__ inline void TransposeBlocksTo16Rows(LocalTensor<T> dst, LocalTensor<T> src,
        uint32_t rowElements)
    {
        uint64_t srcList[16];
        uint64_t dstList[16];
        for (uint32_t row = 0; row < 16; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(src[row * 16].GetPhyAddr());
            dstList[row] = reinterpret_cast<uint64_t>(dst[row * rowElements].GetPhyAddr());
        }
        const uint8_t repeats = static_cast<uint8_t>(rowElements / 16);
        const TransDataTo5HDParams params{false, false, repeats,
            static_cast<uint16_t>(repeats == 1 ? 0 : 1), static_cast<uint16_t>(repeats == 1 ? 0 : 16)};
        TransDataTo5HD<T>(dstList, srcList, params);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
    }

    // FP32 的一个 16x8 基本块输出为低、高两个独立的 32B plane。每个
    // source column 在两个 plane 各占一个 datablock，随后 repeat 以 8 个
    // datablock 推进下一个 8 列块。
    __aicore__ inline void Transpose16Fp32RowsToBlocks(LocalTensor<float> dst,
        LocalTensor<float> src, uint32_t rowElements)
    {
        uint64_t srcList[FP32_TRANSPOSE_ROWS];
        uint64_t dstList[FP32_TRANSPOSE_ROWS];
        for (uint32_t row = 0; row < FP32_TRANSPOSE_ROWS; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(src[row * rowElements].GetPhyAddr());
        }
        const uint32_t planeElements = rowElements * FP32_TRANSPOSE_COL_ALIGNMENT;
        for (uint32_t column = 0; column < FP32_TRANSPOSE_COL_ALIGNMENT; ++column) {
            dstList[2 * column] = reinterpret_cast<uint64_t>(
                dst[column * FP32_TRANSPOSE_COL_ALIGNMENT].GetPhyAddr());
            dstList[2 * column + 1] = reinterpret_cast<uint64_t>(
                dst[planeElements + column * FP32_TRANSPOSE_COL_ALIGNMENT].GetPhyAddr());
        }
        const uint8_t repeats = static_cast<uint8_t>(rowElements / FP32_TRANSPOSE_COL_ALIGNMENT);
        const TransDataTo5HDParams params{false, false, repeats,
            static_cast<uint16_t>(repeats == 1 ? 0 : FP32_TRANSPOSE_COL_ALIGNMENT),
            static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
        TransDataTo5HD<float>(dstList, srcList, params);
        PipeBarrier<PIPE_ALL>();
    }

    // 16 行物理矩阵要求总是提供 16 个源地址；尾部不足时补齐行复用首行，
    // 输出 DMA 只写真实 outer 行，因此不会影响有效累加顺序。FP32 source
    // 的低/高 half 需要按独立 32B plane 排布，供 32B Gatherb offset 直接寻址。
    __aicore__ inline void TransposeFp32ActiveRowsToBlocks(LocalTensor<float> dst,
        LocalTensor<float> src, uint32_t rowElements, uint32_t activeRows)
    {
        uint64_t srcList[FP32_TRANSPOSE_ROWS];
        uint64_t dstList[FP32_TRANSPOSE_ROWS];
        for (uint32_t row = 0; row < FP32_TRANSPOSE_ROWS; ++row) {
            const uint32_t sourceRow = row < activeRows ? row : 0;
            srcList[row] = reinterpret_cast<uint64_t>(src[sourceRow * rowElements].GetPhyAddr());
        }
        // 每个 source column 对应一个 32B half-record；因此一个 plane 的
        // datablock 数等于列数 rowElements，而不是一个 repeat 中的列组数。
        const uint32_t planeBlocks = rowElements;
        // 每个 repeat 消费一个 8-float source block，并写入低/高 plane 各一个
        // 32B block。低 plane 之后留出一个 sentinel block，再开始高 plane，
        // 这样 local * 32B 的 offset 在两个 Gatherb 中都严格落在有效 plane 内。
        const uint32_t highPlaneBlockBase = planeBlocks + 1;
        for (uint32_t column = 0; column < FP32_TRANSPOSE_COL_ALIGNMENT; ++column) {
            dstList[2 * column] = reinterpret_cast<uint64_t>(
                dst[column * FP32_TRANSPOSE_COL_ALIGNMENT].GetPhyAddr());
            dstList[2 * column + 1] = reinterpret_cast<uint64_t>(
                dst[(highPlaneBlockBase + column) * FP32_TRANSPOSE_COL_ALIGNMENT].GetPhyAddr());
        }
        const uint8_t repeats = static_cast<uint8_t>(rowElements / FP32_TRANSPOSE_COL_ALIGNMENT);
        const TransDataTo5HDParams params{false, false, repeats,
            static_cast<uint16_t>(repeats == 1 ? 0 : FP32_TRANSPOSE_COL_ALIGNMENT),
            static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
        TransDataTo5HD<float>(dstList, srcList, params);
        PipeBarrier<PIPE_ALL>();
    }

    // FP32 转置 Gather 的 source 面板统一走同一套二维补齐搬运；调用方只需提供
    // 有效行数，避免首批、当前 chunk 和下一 chunk 三处重复拼装参数。
    __aicore__ inline void CopyFp32SourceRows(LocalTensor<float> dst, uint64_t gmOffset,
        uint32_t rows, uint32_t validCount)
    {
        const uint32_t validBytes = validCount * sizeof(float);
        const uint32_t alignedElements = AlignBytes(validBytes) / sizeof(float);
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), validBytes,
            static_cast<uint32_t>((indexSize_ - validCount) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0,
            static_cast<uint8_t>(alignedElements - validCount), 0.0f};
        DataCopyPad(dst, sourceGm_[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void TransposeFp32BlocksTo16Rows(LocalTensor<float> dst,
        LocalTensor<float> src, uint32_t rowElements)
    {
        uint64_t srcList[FP32_TRANSPOSE_ROWS];
        uint64_t dstList[FP32_TRANSPOSE_ROWS];
        // 低、高 plane 分别处理；每个 plane 的一个 datablock 对应一个 source
        // column，16 列为一次 TransDataTo5HD repeat 组。
        const uint8_t repeats = static_cast<uint8_t>(rowElements / FP32_TRANSPOSE_ROWS);
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t lane = 0; lane < FP32_TRANSPOSE_ROWS; ++lane) {
                srcList[lane] = reinterpret_cast<uint64_t>(
                src[half * rowElements * FP32_TRANSPOSE_COL_ALIGNMENT +
                    lane * FP32_TRANSPOSE_COL_ALIGNMENT].GetPhyAddr());
                const uint32_t outputRow = lane / 2;
                const uint32_t outputOffset =
                    (half * FP32_TRANSPOSE_ACTIVE_ROWS / 2 + outputRow) * rowElements +
                    (lane % 2) * FP32_TRANSPOSE_COL_ALIGNMENT;
                dstList[lane] = reinterpret_cast<uint64_t>(dst[outputOffset].GetPhyAddr());
            }
            const TransDataTo5HDParams params{false, false, repeats,
                static_cast<uint16_t>(repeats == 1 ? 0 : 2),
                static_cast<uint16_t>(repeats == 1 ? 0 : FP32_TRANSPOSE_ROWS)};
            TransDataTo5HD<float>(dstList, srcList, params);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }

    // FP32 转置 Gather 的 metadata 由 source chunk 的 owner core 建立。每个 owner
    // 只扫描自己连续的 chunk、写入独占表槽和 32B header，之后统一发布；因此既不
    // 重物化全局 raw/compact 表，也不让 core 0 串行扫描全部 index。
    __aicore__ inline bool BuildFp32TransposeBlockGatherTables(bool& initialSelfPrefetched,
        bool& initialSourcePrefetched)
    {
        initialSelfPrefetched = false;
        initialSourcePrefetched = false;
        const uint32_t axisPadElements = AlignBytes(axisSize_ * sizeof(float)) / sizeof(float);
        const uint32_t headerEntryElements = UB_BLOCK_BYTES / sizeof(uint32_t);
        const uint32_t sourceChunk = transposeSourceChunk_;
        const uint32_t layerCap = transposeLayerCap_;
        const bool compactOffsets = transposeOffsetBytes_ == sizeof(uint16_t);
        const uint32_t chunkCount = static_cast<uint32_t>(CeilDivU64(indexSize_, sourceChunk));
        const uint32_t headerElements = (chunkCount + 1) * headerEntryElements;
        const uint32_t tableElements = layerCap * axisPadElements;
        const uint64_t headerBytes = static_cast<uint64_t>(chunkCount + 1) * UB_BLOCK_BYTES;
        const uint64_t tableOffsetBytes = AlignWorkspaceBytes(headerBytes);
        const uint64_t tableSlotBytes =
            AlignWorkspaceBytes(static_cast<uint64_t>(tableElements) *
                (compactOffsets ? sizeof(uint16_t) : sizeof(uint32_t)));

        // metadata 阶段使用已驻留的完整 index；payload 在发布屏障后才会覆盖
        // sourceBuf。每个 owner 仍只写自己的 table slot。
        LocalTensor<int32_t> indexLocal = sourceBuf_.Get<int32_t>();
        // counterBuf 在 metadata 后改作跨批次 header 缓存，两个阶段严格串行。
        LocalTensor<uint32_t> counterLocal = counterBuf_.Get<uint32_t>();
        // denseBuf 在 metadata 阶段尚未承载 accumulator，可完整容纳 offset 表。
        LocalTensor<uint32_t> tableLocal = denseBuf_.Get<uint32_t>();
        // dataBuf_ 在建表阶段仅借出一个 32B header staging 槽，payload 开始前会
        // 完整覆盖它，因此不增加常驻 UB buffer。
        LocalTensor<int32_t> headerStageLocal = dataBuf_.Get<int32_t>();
        // int16 -> half 的编码不能原地覆盖 tableLocal：A2 的 Cast 对重叠地址
        // 没有可依赖的语义。借用 dataBuf 的第二个 32B 对齐起点作为独立编码
        // 缓冲，首个 block 仍保留给 header 发布。
        LocalTensor<half> compactTableEncodedLocal = dataBuf_.Get<half>()[
            UB_BLOCK_BYTES / sizeof(half)];
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        __ubuf__ uint32_t* counterAddr = (__ubuf__ uint32_t*)counterLocal.GetPhyAddr();
        __ubuf__ uint32_t* tableAddr = (__ubuf__ uint32_t*)tableLocal.GetPhyAddr();
        __ubuf__ uint16_t* compactTableAddr = (__ubuf__ uint16_t*)tableLocal.GetPhyAddr();
        __ubuf__ int32_t* headerStageAddr = (__ubuf__ int32_t*)headerStageLocal.GetPhyAddr();

        const uint64_t coreId = GetBlockIdx();
        const uint64_t taskStep = blockDim_ == 0 ? 1 : blockDim_;
        bool localOverflow = false;
        uint64_t nextUnpublishedChunk = chunkCount;
        for (uint64_t chunk = coreId; chunk < chunkCount; chunk += taskStep) {
            const uint32_t chunkId = static_cast<uint32_t>(chunk);
            const uint64_t chunkStart = static_cast<uint64_t>(chunkId) * sourceChunk;
            const uint32_t validCount = static_cast<uint32_t>(
                MinU64(sourceChunk, indexSize_ - chunkStart));
            const uint32_t indexValidBytes = validCount * sizeof(int32_t);
            const uint32_t alignedIndexElements =
                AlignBytes(indexValidBytes) / sizeof(int32_t);
            DataCopyExtParams indexCopyParams{1, indexValidBytes, 0, 0, 0};
            DataCopyPadExtParams<int32_t> indexPadParams{
                true, 0, static_cast<uint8_t>(alignedIndexElements - validCount), 0};
            // 每个 owner 只搬当前 chunk 的 index；sourceBuf 在 metadata 阶段未被
            // 其他数据流使用，发布屏障后再复用为 source staging。
            DataCopyPad(indexLocal, indexGm_[chunkStart], indexCopyParams, indexPadParams);
            Sync<HardEvent::MTE2_S>();
            const uint32_t sourceRowElements = AlignBytes(validCount * sizeof(float)) / sizeof(float);
            const uint32_t sentinelOffset = sourceRowElements * FP32_TRANSPOSE_OFFSET_SCALE_BYTES;

            Duplicate<uint32_t>(counterLocal, 0, static_cast<int32_t>(axisPadElements));
            if (compactOffsets) {
                Duplicate<uint16_t>(tableLocal.ReinterpretCast<uint16_t>(),
                    static_cast<uint16_t>(sentinelOffset), static_cast<int32_t>(tableElements));
            } else {
                Duplicate<uint32_t>(tableLocal,
                    sentinelOffset,
                    static_cast<int32_t>(tableElements));
            }
            Sync<HardEvent::V_S>();

            uint32_t actualLayers = 0;
            for (uint32_t local = 0; local < validCount; ++local) {
                const uint32_t axis = static_cast<uint32_t>(indexAddr[local]);
                const uint32_t layer = counterAddr[axis];
                if (layer >= layerCap) {
                    localOverflow = true;
                    break;
                }
                const uint32_t offset = local * FP32_TRANSPOSE_OFFSET_SCALE_BYTES;
                if (compactOffsets) {
                    compactTableAddr[layer * axisPadElements + axis] = static_cast<uint16_t>(offset);
                } else {
                    tableAddr[layer * axisPadElements + axis] = offset;
                }
                counterAddr[axis] = layer + 1;
                actualLayers = actualLayers < layer + 1 ? layer + 1 : actualLayers;
            }

            if (compactOffsets && !localOverflow && actualLayers != 0) {
                // 只有 header 宣布的 occurrence 层会被消费者读取；其余哨兵层不
                // 参与本 chunk 的 Gather，跳过无效的 half 转换以缩短 metadata 关键链。
                Sync<HardEvent::S_V>();
                Cast<half, int16_t>(compactTableEncodedLocal,
                    tableLocal.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE,
                    actualLayers * axisPadElements);
                PipeBarrier<PIPE_V>();
            }

            // 值大于 layerCap 是发布的溢出哨兵。即使不写 table，header 也必须
            // 写回，消费者在同一屏障后会一致回退而不会读取未初始化的表槽。
            headerStageAddr[0] = static_cast<int32_t>(
                localOverflow ? layerCap + 1 : actualLayers);
            Sync<HardEvent::S_MTE3>();
            if (!localOverflow) {
                const uint64_t tableAddress = tableOffsetBytes +
                    static_cast<uint64_t>(chunkId) * tableSlotBytes;
                if (compactOffsets) {
                    DataCopy(workspaceOffsetGm_[tableAddress / sizeof(uint16_t)],
                        compactTableEncodedLocal.ReinterpretCast<uint16_t>(),
                        actualLayers * axisPadElements);
                } else {
                    DataCopy(workspaceGm_[tableAddress / sizeof(int32_t)],
                        tableLocal.ReinterpretCast<int32_t>(), actualLayers * axisPadElements);
                }
            }
            DataCopy(workspaceGm_[(static_cast<uint64_t>(chunkId + 1) * UB_BLOCK_BYTES) /
                sizeof(int32_t)], headerStageLocal, headerEntryElements);
            // 下一次改写 table/header staging 前，必须确认本 chunk 的 MTE3 已取完。
            Sync<HardEvent::MTE3_S>();
            if (localOverflow) {
                // 当前 chunk 的溢出 header 已发布；同一 owner 后续未建表的 chunk
                // 也必须显式发布哨兵，不能让消费者读取未初始化 workspace。
                nextUnpublishedChunk = chunk + taskStep;
                break;
            }
        }

        // 仅发生局部溢出时执行。正常路径保持原有每个 chunk 的 metadata 流水；
        // 回退路径把未处理 slot 统一标记为无效，随后所有 core 经 SyncAll 一致回退。
        for (uint64_t chunk = nextUnpublishedChunk; chunk < chunkCount; chunk += taskStep) {
            headerStageAddr[0] = static_cast<int32_t>(layerCap + 1);
            Sync<HardEvent::S_MTE3>();
            DataCopy(workspaceGm_[(static_cast<uint64_t>(chunk + 1) * UB_BLOCK_BYTES) /
                sizeof(int32_t)], headerStageLocal, headerEntryElements);
            Sync<HardEvent::MTE3_S>();
        }

        // metadata owner 完成自己的表后，sourceBuf 中的 index 生命周期已经结束。
        // 先为每个 consumer core 排入首批 self/source，随后再发布全局 metadata；
        // 这样首批搬运可以与其他 core 的标量建表和表写回重叠。
        const uint64_t batchCount = CeilDivU64(outerSize_, FP32_TRANSPOSE_ACTIVE_ROWS);
        const uint64_t coreBatchCount = batchCount / taskStep;
        const uint64_t extraBatches = batchCount - coreBatchCount * taskStep;
        const uint64_t firstBatch = coreId * coreBatchCount + MinU64(coreId, extraBatches);
        const uint32_t localBatchCount = static_cast<uint32_t>(
            coreBatchCount + static_cast<uint64_t>(coreId < extraBatches));
        if (!localOverflow && localBatchCount != 0) {
            const uint64_t outerStart = firstBatch * FP32_TRANSPOSE_ACTIVE_ROWS;
            const uint32_t outerCount = static_cast<uint32_t>(MinU64(
                FP32_TRANSPOSE_ACTIVE_ROWS, outerSize_ - outerStart));
            const uint32_t axisPadElements = AlignBytes(axisSize_ * sizeof(float)) / sizeof(float);
            const uint32_t axisValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(float));
            LocalTensor<float> sourceRows = sourceBuf_.Get<float>();
            LocalTensor<float> sourceBlocks = sourceRows[
                FP32_TRANSPOSE_ACTIVE_ROWS * transposeSourceChunk_];
            DataCopyExtParams selfCopyParams{static_cast<uint16_t>(outerCount), axisValidBytes, 0, 0, 0};
            DataCopyPadExtParams<float> selfPadParams{true, 0,
                static_cast<uint8_t>(axisPadElements - axisSize_), 0.0f};
            if (outerCount < FP32_TRANSPOSE_ACTIVE_ROWS) {
                Sync<HardEvent::MTE3_V>();
                Duplicate<float>(sourceBlocks, 0.0f,
                    static_cast<int32_t>(FP32_TRANSPOSE_ACTIVE_ROWS * axisPadElements));
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyPad(sourceBlocks, selfGm_[outerStart * axisSize_],
                selfCopyParams, selfPadParams);

            const uint32_t validCount = static_cast<uint32_t>(MinU64(
                transposeSourceChunk_, indexSize_));
            CopyFp32SourceRows(sourceRows, outerStart * indexSize_, outerCount, validCount);
            initialSelfPrefetched = true;
            initialSourcePrefetched = true;
        }

        // 每个 header/table 槽只有一个 owner；所有 owner 的 MTE3 写回完成后再
        // 进入屏障，保证任一消费者均不会看到半发布 metadata。
        SyncAll();

        // counterBuf 仅常驻压缩前的 header slot；payload 不再访问 header workspace。
        // slot 0 未使用，chunk k 的层数在 slot k+1。
        LocalTensor<int32_t> headerLocal = counterBuf_.Get<int32_t>();
        DataCopy(headerLocal, workspaceGm_, headerElements);
        Sync<HardEvent::MTE2_S>();
        __ubuf__ int32_t* headerAddr = (__ubuf__ int32_t*)headerLocal.GetPhyAddr();
        bool ready = true;
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const int32_t actualLayers = headerAddr[(chunk + 1) * headerEntryElements];
            if (actualLayers < 0 || static_cast<uint32_t>(actualLayers) > layerCap) {
                ready = false;
                break;
            }
        }
        Sync<HardEvent::S_MTE2>();
        return ready;
    }

    __aicore__ inline void ProcessFp32TransposeBlockGather()
    {
        bool initialSelfPrefetched = false;
        bool initialSourcePrefetched = false;
        const bool tableReady = BuildFp32TransposeBlockGatherTables(
            initialSelfPrefetched, initialSourcePrefetched);
        const uint64_t coreId = GetBlockIdx();
        const uint64_t taskStep = blockDim_ == 0 ? 1 : blockDim_;
        if (!tableReady) {
            // 局部 chunk 的冲突超出资源上限时，所有 core 已经越过同一发布屏障，
            // 再走既有逐 index 保序窄路径，不使用会改变 FP32 加法顺序的 AtomicAdd。
            // 回退路径需要完整 index；正常 owner-only 路径不会重复这次搬运。
            CacheFp32TransposeIndex();
            const uint64_t coreOuter = outerSize_ / taskStep;
            const uint64_t extraOuter = outerSize_ - coreOuter * taskStep;
            const uint64_t outerStart = coreId * coreOuter + MinU64(coreId, extraOuter);
            const uint32_t outerCount = static_cast<uint32_t>(
                coreOuter + static_cast<uint64_t>(coreId < extraOuter));
            if (outerCount == 0) {
                return;
            }
            uint64_t fallbackStart = outerStart;
            uint32_t fallbackRemaining = outerCount;
            while (fallbackRemaining != 0) {
                const uint32_t fallbackRows = static_cast<uint32_t>(
                    MinU64(FP32_TRANSPOSE_ACTIVE_ROWS, fallbackRemaining));
                ProcessNarrowMaskedTask<true>(fallbackStart, fallbackRows);
                fallbackStart += fallbackRows;
                fallbackRemaining -= fallbackRows;
            }
            return;
        }

        constexpr uint32_t transposeRows = FP32_TRANSPOSE_ACTIVE_ROWS;
        // 以完整转置 batch 为核间工作单元，而不是先按 outer 行切分。这样尾部
        // 只形成一个全局短 batch，绝大多数 core 处理 3/4 个完整 16 行 batch，
        // 减少每个 core 都排空一次短尾的固定转置/写回开销。batch 的数量和分配
        // 仅由运行时 outer 拓扑与物理矩阵宽度推导，不绑定公开 shape。
        const uint64_t batchCount = CeilDivU64(outerSize_, transposeRows);
        const uint64_t coreBatchCount = batchCount / taskStep;
        const uint64_t extraBatches = batchCount - coreBatchCount * taskStep;
        const uint64_t firstBatch = coreId * coreBatchCount + MinU64(coreId, extraBatches);
        const uint32_t localBatchCount = static_cast<uint32_t>(
            coreBatchCount + static_cast<uint64_t>(coreId < extraBatches));
        if (localBatchCount == 0) {
            return;
        }
        uint64_t batchId = firstBatch;
        bool sourcePrefetched = initialSourcePrefetched;
        bool selfPrefetched = initialSelfPrefetched;
        for (uint32_t localBatch = 0; localBatch < localBatchCount; ++localBatch, ++batchId) {
            const uint64_t batchStart = batchId * transposeRows;
            const uint32_t currentRows = static_cast<uint32_t>(MinU64(
                transposeRows, outerSize_ - batchStart));
            const bool hasNextBatch = localBatch + 1 < localBatchCount;
            const uint64_t nextBatchId = batchId + 1;
            const uint64_t nextBatchStart = hasNextBatch ? nextBatchId * transposeRows : 0;
            const uint32_t nextRows = hasNextBatch ? static_cast<uint32_t>(MinU64(
                transposeRows, outerSize_ - nextBatchStart)) : 0;
            bool nextSourcePrefetched = false;
            bool nextSelfPrefetched = false;
            ProcessFp32TransposeBlockGatherTask(batchStart, currentRows,
                nextBatchStart, nextRows, sourcePrefetched, selfPrefetched,
                nextSourcePrefetched, nextSelfPrefetched);
            sourcePrefetched = nextSourcePrefetched;
            selfPrefetched = nextSelfPrefetched;
        }
    }

    __aicore__ inline void ProcessFp32TransposeBlockGatherTask(uint64_t outerStart, uint32_t outerCount,
        uint64_t nextOuterStart, uint32_t nextOuterCount, bool usePrefetchedSource,
        bool usePrefetchedSelf, bool& nextSourcePrefetched, bool& nextSelfPrefetched)
    {
        const uint32_t axisPadElements = AlignBytes(axisSize_ * sizeof(float)) / sizeof(float);
        const uint32_t axisValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(float));
        const uint32_t axisAlignedElements = AlignBytes(axisValidBytes) / sizeof(float);
        const uint32_t chunkCount = static_cast<uint32_t>(CeilDivU64(indexSize_, transposeSourceChunk_));
        const uint32_t headerEntryElements = UB_BLOCK_BYTES / sizeof(uint32_t);
        const uint64_t headerBytes = static_cast<uint64_t>(chunkCount + 1) * UB_BLOCK_BYTES;
        const uint32_t offsetBytes = transposeOffsetBytes_ == sizeof(uint16_t) ?
            sizeof(uint16_t) : sizeof(uint32_t);
        const uint64_t tableOffsetBytes = AlignWorkspaceBytes(headerBytes);
        const uint64_t tableSlotBytes = AlignWorkspaceBytes(static_cast<uint64_t>(transposeLayerCap_) *
            axisPadElements * offsetBytes);
        const uint32_t payloadLayerRows = transposePayloadLayerCap_;
        const uint32_t maxSourceStageElements = FP32_TRANSPOSE_ACTIVE_ROWS * transposeSourceChunk_;
        const bool compactOffsets = transposeOffsetBytes_ == sizeof(uint16_t);
        const uint32_t offsetTableRowBytes = axisPadElements *
            (compactOffsets ? sizeof(uint16_t) : sizeof(uint32_t));
        // dataBuf 保持原有 dense 基址以维持已验证的 bank 映射；紧凑表展开的
        // uint32 offset 放在单个 Gather 输出平面之后，容量由 payload 分组决定。
        const uint32_t offsetBaseElements = FP32_TRANSPOSE_COL_ALIGNMENT * axisPadElements;

        LocalTensor<uint32_t> expandedOffsetBase = dataBuf_.Get<uint32_t>();
        LocalTensor<float> denseBlocks = dataBuf_.Get<float>();
        if (compactOffsets) {
            expandedOffsetBase = expandedOffsetBase[offsetBaseElements];
        }
        LocalTensor<float> sourceRows = sourceBuf_.Get<float>();
        LocalTensor<float> sourceBlocks = sourceRows[maxSourceStageElements];
        LocalTensor<float> accumulatorBlocks = denseBuf_.Get<float>();
        LocalTensor<uint32_t> layerOffsetLocal = layerOffsetBuf_.Get<uint32_t>();
        LocalTensor<uint16_t> compactLayerOffsetLocal = layerOffsetBuf_.Get<uint16_t>();
        LocalTensor<int32_t> headerLocal = counterBuf_.Get<int32_t>();
        __ubuf__ int32_t* headerAddr = (__ubuf__ int32_t*)headerLocal.GetPhyAddr();

        DataCopyExtParams selfCopyParams{static_cast<uint16_t>(outerCount), axisValidBytes, 0, 0, 0};
        DataCopyPadExtParams<float> selfPadParams{true, 0,
            static_cast<uint8_t>(axisPadElements - axisSize_), 0.0f};
        if (usePrefetchedSelf) {
            // 上一批已将 self 原行预取到 sourceBlocks；这里只等待真实 DMA 完成，
            // 不再重复读取 self，也不占用当前 batch 的 dense staging。
            Sync<HardEvent::MTE2_V>();
            Transpose16Fp32RowsToBlocks(accumulatorBlocks, sourceBlocks, axisPadElements);
        } else {
            // dataBuf_ 现在只保留一个 Gather 输出平面，self 不能再写入 denseBlocks。
            // 所有 batch 都使用空闲的 sourceBlocks 暂存 self；sourceRows 始终保留给
            // source 原行预取。这样首批也能在 self DMA 后立即排入首个 source DMA，
            // 再统一等待并转置，恢复启动阶段的 MTE2/V 重叠。sourceChunk >= axisPad
            // 的 host 准入保证该暂存不会越过 sourceBlocks 容量。
            if (outerCount < FP32_TRANSPOSE_ROWS) {
                Sync<HardEvent::MTE3_V>();
                Duplicate<float>(sourceBlocks, 0.0f,
                    static_cast<int32_t>(FP32_TRANSPOSE_ROWS * axisPadElements));
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyPad(sourceBlocks, selfGm_[outerStart * axisSize_], selfCopyParams, selfPadParams);
        }

        // 每个 source column 拆成低、高两个独立的 32B record。两次 Gatherb
        // 仍共用同一份 offset 表，分别写入同一 accumulator 的偶、奇 block，
        // 但 offset 只需表示一个 half 的 32B 物理步长。
        const GatherRepeatParams gatherParams{1, 8};
        // 累加器与 Gather 输出均按连续 32B plane 排布；每个 repeat 覆盖
        // 8 个连续 datablock，不能沿用旧交错 record 的 2/16 步长。
        const BinaryRepeatParams activeRowAddParams{1, 1, 1, 8, 8, 8};
        const uint8_t gatherRepeats = static_cast<uint8_t>(axisPadElements / FP32_TRANSPOSE_COL_ALIGNMENT);
        // sourceRows 在转置完成后不再被当前 Gather 读取：Gatherb 只访问
        // sourceBlocks。因此完整 16 行 batch 可在当前 Gather/Add 的 Vector 链期间
        // 预取下一 chunk 的 raw source；下一轮只需等待表 DMA 后转置即可。
        bool sourcePrefetched = usePrefetchedSource;
        bool metadataPrefetched = false;
        uint32_t metadataPrefetchedChunk = 0;
        uint32_t metadataPrefetchedLayerStart = 0;
        uint32_t sentinelRowElements = 0;
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t actualLayers = static_cast<uint32_t>(
                headerAddr[(chunk + 1) * headerEntryElements]);
            const uint32_t denseLayerCount = actualLayers;
            const uint64_t chunkStart = static_cast<uint64_t>(chunk) * transposeSourceChunk_;
            const uint32_t validCount = static_cast<uint32_t>(
                MinU64(transposeSourceChunk_, indexSize_ - chunkStart));
            const uint32_t sourceValidBytes = validCount * sizeof(float);
            const uint32_t sourceRowElements = AlignBytes(sourceValidBytes) / sizeof(float);
            const uint32_t sourcePlaneElements = FP32_TRANSPOSE_COL_ALIGNMENT * sourceRowElements;
            const uint32_t accumulatorPlaneElements =
                FP32_TRANSPOSE_COL_ALIGNMENT * axisPadElements;

            const uint32_t firstLayerCount = denseLayerCount < payloadLayerRows ?
                denseLayerCount : payloadLayerRows;
            if (firstLayerCount != 0) {
                const bool usePrefetchedMetadata = compactOffsets && metadataPrefetched &&
                    metadataPrefetchedChunk == chunk && metadataPrefetchedLayerStart == 0;
                if (usePrefetchedMetadata) {
                    // 下一 chunk 的表 DMA 已在上一组 Vector 链期间排队；首组 Cast
                    // 前必须等待 MTE2 完成，避免读取尚未写完的紧凑 offset。
                    Sync<HardEvent::MTE2_V>();
                } else {
                    const uint64_t tableAddress = tableOffsetBytes +
                        static_cast<uint64_t>(chunk) * tableSlotBytes;
                    if (compactOffsets) {
                        DataCopy(compactLayerOffsetLocal,
                            workspaceOffsetGm_[tableAddress / sizeof(uint16_t)],
                            firstLayerCount * axisPadElements);
                    } else {
                        DataCopy(layerOffsetLocal.ReinterpretCast<int32_t>(),
                            workspaceGm_[tableAddress / sizeof(int32_t)],
                            firstLayerCount * axisPadElements);
                    }
                }
            }
            if (!sourcePrefetched) {
                CopyFp32SourceRows(sourceRows, outerStart * indexSize_ + chunkStart,
                    outerCount, validCount);
            }
            Sync<HardEvent::MTE2_V>();
            if (chunk == 0 && !usePrefetchedSelf) {
                // 首批 self 与首个 source/metadata DMA 已同时排队；此处等待后仅
                // 转置一次 self，随后 sourceRows 立即进入 source 转置。
                Transpose16Fp32RowsToBlocks(accumulatorBlocks, sourceBlocks, axisPadElements);
            }
            TransposeFp32ActiveRowsToBlocks(sourceBlocks, sourceRows, sourceRowElements, outerCount);
            // 低、高平面各自有一个 32B sentinel，offset 不会越过对应 plane 的
            // 有效 source record。sentinel 需要位于每个平面之后，因此高平面
            // 起点要越过低平面的 sentinel；否则会覆盖首个高平面 record，且
            // 高平面 sentinel 会落到 sourceBuf 尾界之外。
            // 同一 batch 中完整 source chunk 的转置输出行宽相同，sentinel 只需在
            // 每种实际行宽第一次出现时清零；最后一个短 chunk 行宽变化时再补一次，
            // 不必每个 chunk 重复执行两条 Duplicate。
            if (sentinelRowElements != sourceRowElements) {
                Duplicate<float>(sourceBlocks[sourcePlaneElements], 0.0f,
                    FP32_TRANSPOSE_COL_ALIGNMENT);
                LocalTensor<float> sourceBlocksHigh = sourceBlocks[
                    sourcePlaneElements + FP32_TRANSPOSE_COL_ALIGNMENT];
                Duplicate<float>(sourceBlocksHigh[sourcePlaneElements], 0.0f,
                    FP32_TRANSPOSE_COL_ALIGNMENT);
                PipeBarrier<PIPE_V>();
                sentinelRowElements = sourceRowElements;
            }
            LocalTensor<float> sourceBlocksHigh = sourceBlocks[
                sourcePlaneElements + FP32_TRANSPOSE_COL_ALIGNMENT];

            // sourceRows 在转置完成后不再被当前 Gather 读取：Gatherb 只访问
            // sourceBlocks。因此完整 16 行 batch 可在当前 Gather/Add 的 Vector 链期间
            // 预取下一 chunk 的 raw source；下一轮只需等待 MTE2→V 依赖后转置即可。
            const bool prefetchNextChunk = chunk + 1 < chunkCount;
            const bool prefetchNextOuter = !prefetchNextChunk && nextOuterCount != 0;
            if (prefetchNextChunk || prefetchNextOuter) {
                const uint64_t nextChunkStart = prefetchNextChunk ?
                    static_cast<uint64_t>(chunk + 1) * transposeSourceChunk_ : 0;
                const uint64_t nextSourceBase = prefetchNextChunk ?
                    outerStart * indexSize_ + nextChunkStart : nextOuterStart * indexSize_;
                const uint32_t nextRows = prefetchNextChunk ? outerCount : nextOuterCount;
                const uint32_t nextValidCount = static_cast<uint32_t>(
                    MinU64(transposeSourceChunk_, indexSize_ - nextChunkStart));
                CopyFp32SourceRows(sourceRows, nextSourceBase, nextRows, nextValidCount);
            }

            for (uint32_t layerStart = 0; layerStart < denseLayerCount; layerStart += payloadLayerRows) {
                const uint32_t currentLayerCount =
                    (denseLayerCount - layerStart) < payloadLayerRows ?
                    (denseLayerCount - layerStart) : payloadLayerRows;
                const bool usePrefetchedMetadata = compactOffsets && metadataPrefetched &&
                    metadataPrefetchedChunk == chunk &&
                    metadataPrefetchedLayerStart == layerStart;
                if (layerStart != 0) {
                    if (usePrefetchedMetadata) {
                        // Cast 前只等待前一组计算期间排队的 metadata DMA。
                        Sync<HardEvent::MTE2_V>();
                    } else {
                        // 上一组已完成对 offset 行的 Vector 读取，MTE2 才能覆盖同一
                        // 小表缓冲。metadata workspace 仍保存完整层表，不改变层序。
                        Sync<HardEvent::V_MTE2>();
                        const uint64_t tableAddress = tableOffsetBytes +
                            static_cast<uint64_t>(chunk) * tableSlotBytes +
                            static_cast<uint64_t>(layerStart) * offsetTableRowBytes;
                        if (compactOffsets) {
                            DataCopy(compactLayerOffsetLocal,
                                workspaceOffsetGm_[tableAddress / sizeof(uint16_t)],
                                currentLayerCount * axisPadElements);
                        } else {
                            DataCopy(layerOffsetLocal.ReinterpretCast<int32_t>(),
                                workspaceGm_[tableAddress / sizeof(int32_t)],
                                currentLayerCount * axisPadElements);
                        }
                        Sync<HardEvent::MTE2_V>();
                    }
                }
                if (compactOffsets) {
                    // 连续展开整组 half offset，消除逐 layer 的两次 Cast 和同步。
                    Cast<int32_t, half>(expandedOffsetBase.ReinterpretCast<int32_t>(),
                        compactLayerOffsetLocal.ReinterpretCast<half>(), RoundMode::CAST_RINT,
                        currentLayerCount * axisPadElements);
                    PipeBarrier<PIPE_V>();
                }

                // Cast 完成后紧凑表缓冲不再被当前组读取，可在 Gather/Add 链期间
                // 预取下一组或下一 chunk 的 metadata。下一次 Cast 前只等待该 DMA，
                // 不增加表的 GM 流量，也不改变 layer-major 累加顺序。
                metadataPrefetched = false;
                if (compactOffsets) {
                    const bool hasNextGroup = layerStart + currentLayerCount < denseLayerCount;
                    const uint32_t nextChunk = hasNextGroup ? chunk : chunk + 1;
                    const uint32_t nextLayerStart = hasNextGroup ?
                        layerStart + currentLayerCount : 0;
                    uint32_t nextLayerCount = 0;
                    if (hasNextGroup) {
                        nextLayerCount = (denseLayerCount - nextLayerStart) < payloadLayerRows ?
                            (denseLayerCount - nextLayerStart) : payloadLayerRows;
                    } else if (nextChunk < chunkCount) {
                        nextLayerCount = static_cast<uint32_t>(
                            headerAddr[(nextChunk + 1) * headerEntryElements]);
                        nextLayerCount = nextLayerCount < payloadLayerRows ?
                            nextLayerCount : payloadLayerRows;
                    }
                    if (nextLayerCount != 0) {
                        Sync<HardEvent::V_MTE2>();
                        const uint64_t nextTableAddress = tableOffsetBytes +
                            static_cast<uint64_t>(nextChunk) * tableSlotBytes +
                            static_cast<uint64_t>(nextLayerStart) * offsetTableRowBytes;
                        DataCopy(compactLayerOffsetLocal,
                            workspaceOffsetGm_[nextTableAddress / sizeof(uint16_t)],
                            nextLayerCount * axisPadElements);
                        metadataPrefetched = true;
                        metadataPrefetchedChunk = nextChunk;
                        metadataPrefetchedLayerStart = nextLayerStart;
                    }
                }
                for (uint32_t layer = 0; layer < currentLayerCount; ++layer) {
                    LocalTensor<uint32_t> offsets = compactOffsets ?
                        expandedOffsetBase[layer * axisPadElements] :
                        layerOffsetLocal[layer * axisPadElements];
                    Gatherb<uint32_t>(denseBlocks.ReinterpretCast<uint32_t>(),
                        sourceBlocks.ReinterpretCast<uint32_t>(), offsets, gatherRepeats, gatherParams);
                    PipeBarrier<PIPE_V>();
                    Add<float>(accumulatorBlocks, accumulatorBlocks, denseBlocks,
                        static_cast<uint64_t>(VECTOR_BYTES / sizeof(float)), gatherRepeats,
                        activeRowAddParams);
                    PipeBarrier<PIPE_V>();
                    if (outerCount > FP32_TRANSPOSE_ACTIVE_ROWS / 2) {
                        Gatherb<uint32_t>(denseBlocks.ReinterpretCast<uint32_t>(),
                            sourceBlocksHigh.ReinterpretCast<uint32_t>(),
                            offsets, gatherRepeats, gatherParams);
                        PipeBarrier<PIPE_V>();
                        Add<float>(accumulatorBlocks[accumulatorPlaneElements],
                            accumulatorBlocks[accumulatorPlaneElements], denseBlocks,
                            static_cast<uint64_t>(VECTOR_BYTES / sizeof(float)), gatherRepeats,
                            activeRowAddParams);
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            // 下一轮将覆盖 layerOffsetLocal 与 sourceBlocks；sourceRows 的下一
            // chunk DMA 已在本轮 Vector 链中发起，因而这里只约束后续表 DMA。
            Sync<HardEvent::V_MTE2>();
            sourcePrefetched = prefetchNextChunk;
        }

        if (nextOuterCount != 0) {
            // sourceRows 已在最后一个 chunk 的 Vector 链期间预取下一批 source，
            // 因而 sourceBlocks 改作当前结果的逆转置/写回 staging。下一批 self
            // 在其开始处再搬入 sourceBlocks，避免与预取 source 争用同一 UB 区。
            nextSelfPrefetched = false;
        }
        nextSourcePrefetched = nextOuterCount != 0;

        // dataBuf_ 只容纳一个 Gather 输出平面，完整 16 行结果改写入已释放的
        // sourceBlocks；sourceRows 中的下一批 source 预取保持不动。
        TransposeFp32BlocksTo16Rows(sourceBlocks, accumulatorBlocks, axisPadElements);
        Sync<HardEvent::V_MTE3>();
        const uint32_t outputSrcGapBlocks =
            (axisPadElements - axisAlignedElements) * sizeof(float) / UB_BLOCK_BYTES;
        DataCopyExtParams outputCopyParams{static_cast<uint16_t>(outerCount), axisValidBytes,
            outputSrcGapBlocks, 0, 0};
        DataCopyPad(outputGm_[outerStart * axisSize_], sourceBlocks, outputCopyParams);
        Sync<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void ProcessTransposeBlockGatherFallbackTask(uint64_t outerStart,
        uint32_t outerCount)
    {
        // dataBuf_ 的转置布局保留 axisSize 个 32B staging 行。极端重复 index 使
        // occurrence 表溢出时，仍以原子路径保序回退，而不假设普通 batch 的 UB 布局。
        const uint32_t stagingRows = static_cast<uint32_t>(axisSize_);
        ProcessAtomicTask(outerStart, outerCount, 0, 1, stagingRows,
            MinU64(outerCount, static_cast<uint64_t>(stagingRows)));
    }

    __aicore__ inline void ProcessTransposeBlockGather(uint32_t actualLayers, bool layerOverflow)
    {
        constexpr uint32_t transposeRows = 16;
        constexpr uint32_t axisPadElements = 2 * VECTOR_BYTES / sizeof(T);
        const uint64_t blockGatherRepeats = axisPadElements / 8;
        const uint64_t transposeRepeats = indexSize_ / transposeRows +
            2 * (axisPadElements / transposeRows);
        const uint64_t blockGatherWork =
            static_cast<uint64_t>(actualLayers) * 2 * blockGatherRepeats + transposeRepeats;
        const bool useBlockGather = !layerOverflow && blockGatherWork < indexSize_;

        const uint64_t coreId = GetBlockIdx();
        const uint64_t taskStep = blockDim_ == 0 ? 1 : blockDim_;
        const uint64_t coreOuter = outerSize_ / taskStep;
        const uint64_t extraOuter = outerSize_ - coreOuter * taskStep;
        const uint64_t outerStart = coreId * coreOuter + MinU64(coreId, extraOuter);
        const uint64_t outerCount = coreOuter + static_cast<uint64_t>(coreId < extraOuter);

        uint64_t batchStart = outerStart;
        uint64_t remaining = outerCount;
        uint32_t batchIdx = 0;

        if (useBlockGather && remaining != 0) {
            const uint32_t currentRows = static_cast<uint32_t>(MinU64(transposeRows, remaining));
            const uint32_t axisValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
            const uint32_t axisAlignedBytes = AlignBytes(axisValidBytes);
            const uint32_t outputSrcGapBlocks =
                (axisPadElements * sizeof(T) - axisAlignedBytes) / UB_BLOCK_BYTES;
            const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));
            const uint32_t sourceStageElements = static_cast<uint32_t>(transposeRows * indexSize_);
            const uint32_t accumulatorElements = transposeRows * axisPadElements;

            LocalTensor<T> allSourceRows = sourceBuf_.Get<T>();
            LocalTensor<T> allPrefetchedSelf = dataBuf_.Get<T>();
            LocalTensor<T> sourceRows0 = allSourceRows;
            LocalTensor<T> prefetchedSelf0 = allPrefetchedSelf;

            if (currentRows < transposeRows) {
                Duplicate<T>(sourceRows0, static_cast<T>(0),
                    static_cast<int32_t>(sourceStageElements));
                Duplicate<T>(prefetchedSelf0, static_cast<T>(0),
                    static_cast<int32_t>(accumulatorElements));
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyExtParams selfCopyParams{
                static_cast<uint16_t>(currentRows), axisValidBytes, 0, outputSrcGapBlocks, 0};
            DataCopyPadExtParams<T> selfPadParams{false, 0, 0, static_cast<T>(0)};
            if (currentRows == transposeRows) {
                DataCopy(sourceRows0, sourceGm_[batchStart * indexSize_], transposeRows * indexSize_);
            } else {
                DataCopyExtParams sourceCopyParams{
                    static_cast<uint16_t>(currentRows), sourceValidBytes, 0, 0, 0};
                DataCopyPadExtParams<T> sourcePadParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(sourceRows0, sourceGm_[batchStart * indexSize_], sourceCopyParams, sourcePadParams);
            }
            DataCopyPad(prefetchedSelf0, selfGm_[batchStart * axisSize_], selfCopyParams, selfPadParams);
        }

        while (remaining != 0) {
            const uint32_t currentRows = static_cast<uint32_t>(MinU64(transposeRows, remaining));
            const uint64_t nextBatchStart = batchStart + currentRows;
            const uint32_t nextRows = static_cast<uint32_t>(MinU64(
                transposeRows, remaining - currentRows));
            if (useBlockGather) {
                ProcessTransposeBlockGatherTask(batchStart, currentRows, actualLayers, nextBatchStart,
                    nextRows, batchIdx);
            } else {
                ProcessTransposeBlockGatherFallbackTask(batchStart, currentRows);
            }
            batchStart += currentRows;
            remaining -= currentRows;
            ++batchIdx;
        }
    }

    __aicore__ inline void ProcessTransposeBlockGatherTask(uint64_t outerStart, uint32_t outerCount,
        uint32_t actualLayers, uint64_t nextOuterStart, uint32_t nextOuterCount, uint32_t batchIdx)
    {
        constexpr uint32_t transposeRows = 16;
        constexpr uint32_t axisPadElements = 2 * VECTOR_BYTES / sizeof(T);
        constexpr uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t axisValidBytes = static_cast<uint32_t>(axisSize_ * sizeof(T));
        const uint32_t axisAlignedBytes = AlignBytes(axisValidBytes);
        const uint32_t sourceValidBytes = static_cast<uint32_t>(indexSize_ * sizeof(T));
        const uint32_t sourceStageElements = static_cast<uint32_t>(transposeRows * indexSize_);
        const uint32_t accumulatorElements = transposeRows * axisPadElements;
        const uint32_t sourceTransposeElements = sourceStageElements + blockElements;
        const uint32_t accumulatorOffsetElements = 2 * sourceStageElements + sourceTransposeElements;
        const uint32_t outputSrcGapBlocks =
            (axisPadElements * sizeof(T) - axisAlignedBytes) / UB_BLOCK_BYTES;

        const uint32_t currSlot = batchIdx & 1U;
        const uint32_t nextSlot = 1U - currSlot;

        LocalTensor<T> allSourceRows = sourceBuf_.Get<T>();
        LocalTensor<T> sourceRowsCurr = allSourceRows[currSlot * sourceStageElements];
        LocalTensor<T> sourceRowsNext = allSourceRows[nextSlot * sourceStageElements];
        LocalTensor<T> sourceBlocks = allSourceRows[2 * sourceStageElements];
        LocalTensor<T> accumulatorBlocks = allSourceRows[accumulatorOffsetElements];

        LocalTensor<T> allPrefetchedSelf = dataBuf_.Get<T>();
        LocalTensor<T> prefetchedSelfCurr = allPrefetchedSelf[currSlot * accumulatorElements];
        LocalTensor<T> prefetchedSelfNext = allPrefetchedSelf[nextSlot * accumulatorElements];

        LocalTensor<T> denseBlocks = denseBuf_.Get<T>();
        LocalTensor<uint32_t> layerOffsetLocal = layerOffsetBuf_.Get<uint32_t>();

        Sync<HardEvent::MTE2_V>();

        if (nextOuterCount != 0) {
            if (batchIdx >= 1) {
                Sync<HardEvent::V_MTE2>();
            }
            if (nextOuterCount < transposeRows) {
                Duplicate<T>(sourceRowsNext, static_cast<T>(0),
                    static_cast<int32_t>(sourceStageElements));
                Duplicate<T>(prefetchedSelfNext, static_cast<T>(0),
                    static_cast<int32_t>(accumulatorElements));
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyExtParams nextSelfCopyParams{
                static_cast<uint16_t>(nextOuterCount), axisValidBytes, 0, outputSrcGapBlocks, 0};
            DataCopyPadExtParams<T> selfPadParams{false, 0, 0, static_cast<T>(0)};

            if (nextOuterCount == transposeRows) {
                DataCopy(sourceRowsNext, sourceGm_[nextOuterStart * indexSize_], transposeRows * indexSize_);
            } else {
                DataCopyExtParams nextSourceCopyParams{
                    static_cast<uint16_t>(nextOuterCount), sourceValidBytes, 0, 0, 0};
                DataCopyPadExtParams<T> sourcePadParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(sourceRowsNext, sourceGm_[nextOuterStart * indexSize_], nextSourceCopyParams,
                    sourcePadParams);
            }
            DataCopyPad(prefetchedSelfNext, selfGm_[nextOuterStart * axisSize_], nextSelfCopyParams,
                selfPadParams);
        }

        Transpose16RowsToBlocks(accumulatorBlocks, prefetchedSelfCurr, axisPadElements);
        Transpose16RowsToBlocks(sourceBlocks, sourceRowsCurr, static_cast<uint32_t>(indexSize_));
        Duplicate<T>(sourceBlocks[sourceStageElements], static_cast<T>(0), blockElements);
        PipeBarrier<PIPE_V>();

        if (batchIdx != 0) {
            Sync<HardEvent::MTE3_V>();
        }

        const GatherRepeatParams gatherParams{1, 8};
        const uint8_t gatherRepeats = static_cast<uint8_t>(axisPadElements / 8);
        for (uint32_t layer = 0; layer < actualLayers; ++layer) {
            Gatherb<uint16_t>(denseBlocks.template ReinterpretCast<uint16_t>(),
                sourceBlocks.template ReinterpretCast<uint16_t>(), layerOffsetLocal[layer * axisPadElements],
                gatherRepeats, gatherParams);
            PipeBarrier<PIPE_V>();
            Add<T>(accumulatorBlocks, accumulatorBlocks, denseBlocks,
                static_cast<int32_t>(transposeRows * axisPadElements));
            PipeBarrier<PIPE_V>();
        }

        TransposeBlocksTo16Rows(denseBlocks, accumulatorBlocks, axisPadElements);
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams outputCopyParams{
            static_cast<uint16_t>(outerCount), axisValidBytes, outputSrcGapBlocks, 0, 0};
        DataCopyPad(outputGm_[outerStart * axisSize_], denseBlocks, outputCopyParams);
    }

    __aicore__ inline void PrepareBatchGatherConstants()
    {
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<int32_t> rowBaseLocal = rowBaseBuf_.Get<int32_t>();
        constexpr uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        constexpr uint32_t offsetBlockElements = UB_BLOCK_BYTES / sizeof(uint32_t);
        const uint32_t sourceRowBytes = AlignBytes(indexSize_ * sizeof(T));
        const uint32_t sourcePitchBytes = sourceRowBytes + UB_BLOCK_BYTES;
        const uint32_t sourcePitchElements = sourcePitchBytes / sizeof(T);
        const uint32_t sourceSentinelElements = sourceRowBytes / sizeof(T);

        // source 搬入不会覆盖独立 sentinel，rowBase 也与 task 无关，每核只初始化一次。
        for (uint32_t row = 0; row < outerBatch_; ++row) {
            Duplicate<T>(sourceLocal[row * sourcePitchElements + sourceSentinelElements],
                static_cast<T>(0), blockElements);
            Duplicate<int32_t>(rowBaseLocal[row * offsetBlockElements],
                static_cast<int32_t>(row * sourcePitchBytes), offsetBlockElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessNarrowFallbackTask(uint64_t outerStart, uint32_t outerCount)
    {
        // Gather 缓冲区的一行 accumulator 可复用为若干 32B Atomic staging 行。
        const uint32_t stagingRows = AlignBytes(axisSize_ * sizeof(T)) / UB_BLOCK_BYTES;
        ProcessAtomicTask(outerStart, outerCount, 0, 1, stagingRows, MinU64(outerCount, stagingRows));
    }

    __aicore__ inline void ProcessBatchGatherFallbackTask(uint64_t outerStart, uint32_t outerCount)
    {
        // strategy 4 的 dataBuf 含 B 个 256B 行，可作为 B*8 个 32B Atomic staging 行。
        const uint32_t stagingRows = static_cast<uint32_t>(outerBatch_) * VECTOR_BYTES / UB_BLOCK_BYTES;
        ProcessAtomicTask(outerStart, outerCount, 0, 1, stagingRows, MinU64(outerCount, stagingRows));
    }

    __aicore__ inline void ProcessBf16ResidentTask(uint64_t outerStart, uint32_t outerCount,
        uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> accumulator = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<float> sourceFp32 = denseBuf_.Get<float>();
        LocalTensor<float> workFp32 = workBuf_.Get<float>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t bf16RowBytes = AlignBytes(innerTile_ * sizeof(T));
        const uint32_t fp32RowBytes = AlignBytes(innerTile_ * sizeof(float));
        const uint32_t bf16RowElements = bf16RowBytes / sizeof(T);
        const uint32_t fp32RowElements = fp32RowBytes / sizeof(float);

        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(accumulator[rowBase * bf16RowElements], selfGm_, selfOffset, rows,
                    validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE2_V>();

            for (uint32_t mBase = 0; mBase < indexSize_; mBase += sourceChunk_) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - mBase));
                const uint64_t sourceOffset = (outer * indexSize_ + mBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(sourceLocal, sourceGm_, sourceOffset, rows, validBytes,
                    innerSize_, bf16RowBytes);
                Sync<HardEvent::MTE2_V>();

                Cast<float, T>(sourceFp32, sourceLocal, RoundMode::CAST_NONE,
                    static_cast<uint32_t>(rows * bf16RowElements));
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t outputIndex = static_cast<uint32_t>(indexAddr[mBase + row]);
                    const uint64_t accumulatorOffset = static_cast<uint64_t>(outputIndex) * bf16RowElements;
                    Cast<float, T>(workFp32, accumulator[accumulatorOffset], RoundMode::CAST_NONE, innerCount);
                    PipeBarrier<PIPE_V>();
                    Add<float>(workFp32, workFp32, sourceFp32[row * fp32RowElements], innerCount);
                    PipeBarrier<PIPE_V>();
                    Cast<T, float>(accumulator[accumulatorOffset], workFp32, RoundMode::CAST_RINT, innerCount);
                    PipeBarrier<PIPE_V>();
                }
                Sync<HardEvent::V_MTE2>();
            }

            Sync<HardEvent::V_MTE3>();
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t outputOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsOutFixedPitch(outputGm_, accumulator[rowBase * bf16RowElements], outputOffset, rows,
                    validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void CopyBf16SourcePanel(uint64_t outer, uint32_t mBase,
        uint32_t rows, uint64_t innerStart, uint32_t validBytes, uint32_t bf16RowBytes)
    {
        LocalTensor<T> sourceLocal = bf16SourceQueue_.AllocTensor<T>();
        const uint64_t sourceOffset = (outer * indexSize_ + mBase) * innerSize_ + innerStart;
        CopyRowsInFixedPitch(sourceLocal, sourceGm_, sourceOffset, rows, validBytes,
            innerSize_, bf16RowBytes);
        bf16SourceQueue_.EnQue(sourceLocal);
    }

    __aicore__ inline void ProcessBf16SourceDoubleBufferTask(uint64_t outerStart,
        uint32_t outerCount, uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> accumulator = dataBuf_.Get<T>();
        LocalTensor<float> sourceFp32 = denseBuf_.Get<float>();
        LocalTensor<float> workFp32 = workBuf_.Get<float>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        // 高重复序列直接保留原流水；首个四元组两两不同时才进入独立工作行实现。
        // 后续窗口若存在依赖会在该函数内退回既有 pair/单行顺序。
        if (indexSize_ >= BF16_INDEPENDENT_WORK_ROWS && indexAddr[0] != indexAddr[1] &&
            indexAddr[0] != indexAddr[2] && indexAddr[0] != indexAddr[3] &&
            indexAddr[1] != indexAddr[2] && indexAddr[1] != indexAddr[3] &&
            indexAddr[2] != indexAddr[3]) {
            ProcessBf16SourceDoubleBufferIndependentTask(
                outerStart, outerCount, innerStart, innerCount);
            return;
        }
        // 小 index 或首四元组存在依赖时保留已验收的相邻 pair 入口，避免扩大回退范围。
        if (indexSize_ > 1 && indexAddr[0] != indexAddr[1]) {
            ProcessBf16SourceDoubleBufferIndependentTask(
                outerStart, outerCount, innerStart, innerCount);
            return;
        }
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t bf16RowBytes = AlignBytes(innerTile_ * sizeof(T));
        const uint32_t fp32RowBytes = AlignBytes(innerTile_ * sizeof(float));
        const uint32_t bf16RowElements = bf16RowBytes / sizeof(T);
        const uint32_t fp32RowElements = fp32RowBytes / sizeof(float);

        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(accumulator[rowBase * bf16RowElements], selfGm_, selfOffset, rows,
                    validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE2_V>();

            uint32_t mBase = 0;
            uint32_t rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_));
            CopyBf16SourcePanel(outer, mBase, rows, innerStart, validBytes, bf16RowBytes);
            while (mBase < indexSize_) {
                LocalTensor<T> sourceLocal = bf16SourceQueue_.DeQue<T>();
                const uint32_t nextBase = mBase + rows;
                if (nextBase < indexSize_) {
                    const uint32_t nextRows =
                        static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - nextBase));
                    CopyBf16SourcePanel(
                        outer, nextBase, nextRows, innerStart, validBytes, bf16RowBytes);
                }

                Cast<float, T>(sourceFp32, sourceLocal, RoundMode::CAST_NONE,
                    static_cast<uint32_t>(rows * bf16RowElements));
                PipeBarrier<PIPE_V>();
                bf16SourceQueue_.FreeTensor(sourceLocal);

                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t outputIndex = static_cast<uint32_t>(indexAddr[mBase + row]);
                    const uint64_t accumulatorOffset =
                        static_cast<uint64_t>(outputIndex) * bf16RowElements;
                    Cast<float, T>(workFp32, accumulator[accumulatorOffset],
                        RoundMode::CAST_NONE, innerCount);
                    PipeBarrier<PIPE_V>();
                    Add<float>(workFp32, workFp32, sourceFp32[row * fp32RowElements], innerCount);
                    PipeBarrier<PIPE_V>();
                    Cast<T, float>(accumulator[accumulatorOffset], workFp32,
                        RoundMode::CAST_RINT, innerCount);
                    PipeBarrier<PIPE_V>();
                }

                mBase = nextBase;
                rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - mBase));
            }

            Sync<HardEvent::V_MTE3>();
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t outputRows =
                    static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t outputOffset =
                    (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsOutFixedPitch(outputGm_, accumulator[rowBase * bf16RowElements],
                    outputOffset, outputRows, validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __attribute__((noinline)) __aicore__ void ProcessBf16SourceDoubleBufferIndependentTask(
        uint64_t outerStart, uint32_t outerCount, uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> accumulator = dataBuf_.Get<T>();
        LocalTensor<float> sourceFp32 = denseBuf_.Get<float>();
        LocalTensor<float> workFp32 = workBuf_.Get<float>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t bf16RowBytes = AlignBytes(innerTile_ * sizeof(T));
        const uint32_t fp32RowBytes = AlignBytes(innerTile_ * sizeof(float));
        const uint32_t bf16RowElements = bf16RowBytes / sizeof(T);
        const uint32_t fp32RowElements = fp32RowBytes / sizeof(float);

        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(accumulator[rowBase * bf16RowElements], selfGm_, selfOffset, rows,
                    validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE2_V>();

            uint32_t mBase = 0;
            uint32_t rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_));
            CopyBf16SourcePanel(outer, mBase, rows, innerStart, validBytes, bf16RowBytes);
            while (mBase < indexSize_) {
                LocalTensor<T> sourceLocal = bf16SourceQueue_.DeQue<T>();
                const uint32_t nextBase = mBase + rows;
                if (nextBase < indexSize_) {
                    const uint32_t nextRows =
                        static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - nextBase));
                    CopyBf16SourcePanel(
                        outer, nextBase, nextRows, innerStart, validBytes, bf16RowBytes);
                }

                Cast<float, T>(sourceFp32, sourceLocal, RoundMode::CAST_NONE,
                    static_cast<uint32_t>(rows * bf16RowElements));
                PipeBarrier<PIPE_V>();
                bf16SourceQueue_.FreeTensor(sourceLocal);

                LocalTensor<float> nextWorkFp32 = workFp32[fp32RowElements];
                LocalTensor<float> thirdWorkFp32 = workFp32[2 * fp32RowElements];
                LocalTensor<float> fourthWorkFp32 = workFp32[3 * fp32RowElements];
                uint32_t row = 0;
                while (row < rows) {
                    const uint32_t outputIndex = static_cast<uint32_t>(indexAddr[mBase + row]);
                    const uint64_t accumulatorOffset =
                        static_cast<uint64_t>(outputIndex) * bf16RowElements;
                    // 四次更新的目标行两两独立时可并发发射；每条输出行本身仍按 m 顺序
                    // 做 Cast/Add/Cast，因而不改变 BF16 的逐次舍入语义。
                    if (row + BF16_INDEPENDENT_WORK_ROWS <= rows) {
                        const uint32_t index1 = static_cast<uint32_t>(indexAddr[mBase + row + 1]);
                        const uint32_t index2 = static_cast<uint32_t>(indexAddr[mBase + row + 2]);
                        const uint32_t index3 = static_cast<uint32_t>(indexAddr[mBase + row + 3]);
                        // 用无短路位与消除标量跳转分支预测停顿
                        const bool allDistinct = ((outputIndex != index1) & (outputIndex != index2) &
                                                  (outputIndex != index3) & (index1 != index2) &
                                                  (index1 != index3) & (index2 != index3));
                        if (allDistinct) {
                            const uint64_t accumulatorOffset1 =
                                static_cast<uint64_t>(index1) * bf16RowElements;
                            const uint64_t accumulatorOffset2 =
                                static_cast<uint64_t>(index2) * bf16RowElements;
                            const uint64_t accumulatorOffset3 =
                                static_cast<uint64_t>(index3) * bf16RowElements;
                            Cast<float, T>(workFp32, accumulator[accumulatorOffset],
                                RoundMode::CAST_NONE, innerCount);
                            Cast<float, T>(nextWorkFp32, accumulator[accumulatorOffset1],
                                RoundMode::CAST_NONE, innerCount);
                            Cast<float, T>(thirdWorkFp32, accumulator[accumulatorOffset2],
                                RoundMode::CAST_NONE, innerCount);
                            Cast<float, T>(fourthWorkFp32, accumulator[accumulatorOffset3],
                                RoundMode::CAST_NONE, innerCount);
                            PipeBarrier<PIPE_V>();
                            Add<float>(workFp32, workFp32, sourceFp32[row * fp32RowElements], innerCount);
                            Add<float>(nextWorkFp32, nextWorkFp32,
                                sourceFp32[(row + 1) * fp32RowElements], innerCount);
                            Add<float>(thirdWorkFp32, thirdWorkFp32,
                                sourceFp32[(row + 2) * fp32RowElements], innerCount);
                            Add<float>(fourthWorkFp32, fourthWorkFp32,
                                sourceFp32[(row + 3) * fp32RowElements], innerCount);
                            PipeBarrier<PIPE_V>();
                            Cast<T, float>(accumulator[accumulatorOffset], workFp32,
                                RoundMode::CAST_RINT, innerCount);
                            Cast<T, float>(accumulator[accumulatorOffset1], nextWorkFp32,
                                RoundMode::CAST_RINT, innerCount);
                            Cast<T, float>(accumulator[accumulatorOffset2], thirdWorkFp32,
                                RoundMode::CAST_RINT, innerCount);
                            Cast<T, float>(accumulator[accumulatorOffset3], fourthWorkFp32,
                                RoundMode::CAST_RINT, innerCount);
                            PipeBarrier<PIPE_V>();
                            row += BF16_INDEPENDENT_WORK_ROWS;
                            continue;
                        }
                    }
                    // 四元组不独立时，仍可复用已验收的两行并行路径。
                    // 只有相邻两次更新落到不同输出行时才并发发射，保持重复 index 的逐次舍入顺序。
                    if (row + 1 < rows) {
                        const uint32_t nextOutputIndex =
                            static_cast<uint32_t>(indexAddr[mBase + row + 1]);
                        if (outputIndex != nextOutputIndex) {
                            const uint64_t nextAccumulatorOffset =
                                static_cast<uint64_t>(nextOutputIndex) * bf16RowElements;
                            Cast<float, T>(workFp32, accumulator[accumulatorOffset],
                                RoundMode::CAST_NONE, innerCount);
                            Cast<float, T>(nextWorkFp32, accumulator[nextAccumulatorOffset],
                                RoundMode::CAST_NONE, innerCount);
                            PipeBarrier<PIPE_V>();
                            Add<float>(workFp32, workFp32, sourceFp32[row * fp32RowElements], innerCount);
                            Add<float>(nextWorkFp32, nextWorkFp32,
                                sourceFp32[(row + 1) * fp32RowElements], innerCount);
                            PipeBarrier<PIPE_V>();
                            Cast<T, float>(accumulator[accumulatorOffset], workFp32,
                                RoundMode::CAST_RINT, innerCount);
                            Cast<T, float>(accumulator[nextAccumulatorOffset], nextWorkFp32,
                                RoundMode::CAST_RINT, innerCount);
                            PipeBarrier<PIPE_V>();
                            row += 2;
                            continue;
                        }
                    }
                    Cast<float, T>(workFp32, accumulator[accumulatorOffset],
                        RoundMode::CAST_NONE, innerCount);
                    PipeBarrier<PIPE_V>();
                    Add<float>(workFp32, workFp32, sourceFp32[row * fp32RowElements], innerCount);
                    PipeBarrier<PIPE_V>();
                    Cast<T, float>(accumulator[accumulatorOffset], workFp32,
                        RoundMode::CAST_RINT, innerCount);
                    PipeBarrier<PIPE_V>();
                    ++row;
                }

                mBase = nextBase;
                rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - mBase));
            }

            Sync<HardEvent::V_MTE3>();
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t outputRows =
                    static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t outputOffset =
                    (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsOutFixedPitch(outputGm_, accumulator[rowBase * bf16RowElements],
                    outputOffset, outputRows, validBytes, innerSize_, bf16RowBytes);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void ProcessInt8ResidentTask(uint64_t outerStart, uint32_t outerCount,
        uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> rawLocal = dataBuf_.Get<T>();
        LocalTensor<half> halfLocal = sourceBuf_.Get<half>();
        LocalTensor<int32_t> intLocal = denseBuf_.Get<int32_t>();
        LocalTensor<float> floatLocal = workBuf_.Get<float>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        __ubuf__ int32_t* indexAddr = (__ubuf__ int32_t*)indexLocal.GetPhyAddr();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t int8RowBytes = AlignBytes(innerTile_ * sizeof(T));
        const uint32_t halfRowBytes = AlignBytes(innerTile_ * sizeof(half));
        const uint32_t int32RowBytes = AlignBytes(innerTile_ * sizeof(int32_t));
        const uint32_t rowElements = int8RowBytes / sizeof(T);
        const uint32_t sourceIntOffset =
            static_cast<uint32_t>((axisSize_ * int32RowBytes + VECTOR_BYTES) / sizeof(int32_t));
        LocalTensor<int32_t> accumulator = intLocal;
        LocalTensor<int32_t> sourceInt = intLocal[sourceIntOffset];

        for (uint32_t outerOffset = 0; outerOffset < outerCount; ++outerOffset) {
            const uint64_t outer = outerStart + outerOffset;
            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t selfOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(rawLocal[rowBase * rowElements], selfGm_, selfOffset, rows,
                    validBytes, innerSize_, int8RowBytes);
            }
            Sync<HardEvent::MTE2_V>();
            const uint32_t accumulatorElements = static_cast<uint32_t>(axisSize_ * rowElements);
            Cast<half, T>(halfLocal, rawLocal, RoundMode::CAST_NONE, accumulatorElements);
            PipeBarrier<PIPE_V>();
            Cast<int32_t, half>(accumulator, halfLocal, RoundMode::CAST_RINT, accumulatorElements);
            Sync<HardEvent::V_MTE2>();

            for (uint32_t mBase = 0; mBase < indexSize_; mBase += sourceChunk_) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(sourceChunk_, indexSize_ - mBase));
                const uint64_t sourceOffset = (outer * indexSize_ + mBase) * innerSize_ + innerStart;
                CopyRowsInFixedPitch(rawLocal, sourceGm_, sourceOffset, rows, validBytes,
                    innerSize_, int8RowBytes);
                Sync<HardEvent::MTE2_V>();
                const uint32_t sourceElements = rows * rowElements;
                Cast<half, T>(halfLocal, rawLocal, RoundMode::CAST_NONE, sourceElements);
                PipeBarrier<PIPE_V>();
                Cast<int32_t, half>(sourceInt, halfLocal, RoundMode::CAST_RINT, sourceElements);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t outputIndex = static_cast<uint32_t>(indexAddr[mBase + row]);
                    Add<int32_t>(accumulator[outputIndex * rowElements],
                        accumulator[outputIndex * rowElements], sourceInt[row * rowElements], innerCount);
                }
                Sync<HardEvent::V_MTE2>();
            }

            ShiftLeft<int32_t>(accumulator, accumulator, static_cast<int32_t>(24), accumulatorElements);
            PipeBarrier<PIPE_V>();
            ShiftRight<int32_t>(accumulator, accumulator, static_cast<int32_t>(24), accumulatorElements);
            PipeBarrier<PIPE_V>();
            Cast<float, int32_t>(floatLocal, accumulator, RoundMode::CAST_NONE, accumulatorElements);
            PipeBarrier<PIPE_V>();
            Cast<half, float>(halfLocal, floatLocal, RoundMode::CAST_RINT, accumulatorElements);
            PipeBarrier<PIPE_V>();
            Cast<T, half>(rawLocal, halfLocal, RoundMode::CAST_NONE, accumulatorElements);
            Sync<HardEvent::V_MTE3>();

            for (uint64_t rowBase = 0; rowBase < axisSize_; rowBase += MAX_COPY_ROWS) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(MAX_COPY_ROWS, axisSize_ - rowBase));
                const uint64_t outputOffset = (outer * axisSize_ + rowBase) * innerSize_ + innerStart;
                CopyRowsOutFixedPitch(outputGm_, rawLocal[rowBase * rowElements], outputOffset, rows,
                    validBytes, innerSize_, int8RowBytes);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void ProcessInt8AxisTask(
        uint64_t outer, uint64_t outputAxis, uint64_t innerStart, uint32_t innerCount)
    {
        LocalTensor<T> rawLocal = dataBuf_.Get<T>();
        LocalTensor<half> halfLocal = sourceBuf_.Get<half>();
        LocalTensor<int32_t> intLocal = denseBuf_.Get<int32_t>();
        LocalTensor<uint8_t> byteLocal = denseBuf_.Get<uint8_t>();
        LocalTensor<float> floatLocal = workBuf_.Get<float>();
        LocalTensor<int32_t> indexLocal = indexBuf_.Get<int32_t>();
        const uint32_t validBytes = innerCount * sizeof(T);
        const uint32_t rowElements = AlignBytes(innerTile_ * sizeof(T)) / sizeof(T);
        const uint32_t int32RowBytes = AlignBytes(innerTile_ * sizeof(int32_t));
        const uint32_t compareElements = static_cast<uint32_t>(
            (indexSize_ + VECTOR_BYTES / sizeof(int32_t) - 1) /
            (VECTOR_BYTES / sizeof(int32_t)) * (VECTOR_BYTES / sizeof(int32_t)));
        const uint32_t compareMaskBytes = AlignBytes(compareElements / 8);
        const uint32_t maskGapBytes = compareMaskBytes > VECTOR_BYTES ?
            compareMaskBytes : VECTOR_BYTES;
        const uint32_t sourceIntOffset = (int32RowBytes + maskGapBytes) / sizeof(int32_t);
        LocalTensor<int32_t> accumulator = intLocal;
        LocalTensor<int32_t> sourceInt = intLocal[sourceIntOffset];
        LocalTensor<uint8_t> maskLocal = byteLocal[int32RowBytes];
        __ubuf__ uint32_t* maskAddr = (__ubuf__ uint32_t*)maskLocal.GetPhyAddr();
        DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        const uint64_t selfOffset = (outer * axisSize_ + outputAxis) * innerSize_ + innerStart;
        DataCopyPad(rawLocal, selfGm_[selfOffset], copyParams, padParams);
        CompareScalar<int32_t, uint8_t>(maskLocal, indexLocal, static_cast<int32_t>(outputAxis),
            CMPMODE::EQ, compareElements);
        Sync<HardEvent::V_S>();
        Sync<HardEvent::MTE2_V>();
        Cast<half, T>(halfLocal, rawLocal, RoundMode::CAST_NONE, rowElements);
        PipeBarrier<PIPE_V>();
        Cast<int32_t, half>(accumulator, halfLocal, RoundMode::CAST_RINT, rowElements);
        Sync<HardEvent::V_MTE2>();

        for (uint32_t word = 0; word < compareElements / 32; ++word) {
            uint32_t bits = maskAddr[word];
            while (bits != 0) {
                const uint32_t bit = static_cast<uint32_t>(
                    __builtin_ffs(static_cast<int32_t>(bits)) - 1);
                const uint32_t m = word * 32 + bit;
                const uint64_t sourceOffset = (outer * indexSize_ + m) * innerSize_ + innerStart;
                DataCopyPad(rawLocal, sourceGm_[sourceOffset], copyParams, padParams);
                Sync<HardEvent::MTE2_V>();
                Cast<half, T>(halfLocal, rawLocal, RoundMode::CAST_NONE, rowElements);
                PipeBarrier<PIPE_V>();
                Cast<int32_t, half>(sourceInt, halfLocal, RoundMode::CAST_RINT, rowElements);
                PipeBarrier<PIPE_V>();
                Add<int32_t>(accumulator, accumulator, sourceInt, innerCount);
                Sync<HardEvent::V_MTE2>();
                bits &= bits - 1;
            }
        }

        ShiftLeft<int32_t>(accumulator, accumulator, static_cast<int32_t>(24), rowElements);
        PipeBarrier<PIPE_V>();
        ShiftRight<int32_t>(accumulator, accumulator, static_cast<int32_t>(24), rowElements);
        PipeBarrier<PIPE_V>();
        Cast<float, int32_t>(floatLocal, accumulator, RoundMode::CAST_NONE, rowElements);
        PipeBarrier<PIPE_V>();
        Cast<half, float>(halfLocal, floatLocal, RoundMode::CAST_RINT, rowElements);
        PipeBarrier<PIPE_V>();
        Cast<T, half>(rawLocal, halfLocal, RoundMode::CAST_NONE, rowElements);
        Sync<HardEvent::V_MTE3>();

        DataCopyPad(outputGm_[selfOffset], rawLocal, copyParams);
        Sync<HardEvent::MTE3_MTE2>();
    }

    // 窄 Gather 的普通与转置溢出回退路径仅有 metadata 所在 buffer 不同，
    // 通过模板参数复用同一条向量累加流水，避免两份实现长期漂移。
    template <bool TRANSPOSE_FALLBACK>
    __aicore__ inline void ProcessNarrowMaskedTask(uint64_t outerStart, uint32_t outerCount)
    {
        LocalTensor<T> accLocal = dataBuf_.Get<T>();
        LocalTensor<T> sourceLocal;
        LocalTensor<int32_t> indexLocal;
        if constexpr (TRANSPOSE_FALLBACK) {
            sourceLocal = layerOffsetBuf_.Get<T>();
            indexLocal = sourceBuf_.Get<int32_t>();
        } else {
            sourceLocal = sourceBuf_.Get<T>();
            indexLocal = indexBuf_.Get<int32_t>();
        }
        const uint32_t accumulatorValidBytes = axisSize_ * sizeof(T);
        const uint32_t accumulatorRowBytes = AlignBytes(accumulatorValidBytes);
        const uint32_t accumulatorRowBlocks = accumulatorRowBytes / UB_BLOCK_BYTES;
        constexpr uint32_t vectorElements = VECTOR_BYTES / sizeof(T);
        constexpr uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);

        DataCopyExtParams selfCopyParams{
            static_cast<uint16_t>(outerCount), accumulatorValidBytes, 0, 0, 0};
        DataCopyPadExtParams<T> selfPadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(accLocal, selfGm_[outerStart * axisSize_], selfCopyParams, selfPadParams);
        Sync<HardEvent::MTE2_V>();

        const BinaryRepeatParams repeatParams{1, 1, 1,
            static_cast<uint8_t>(accumulatorRowBlocks), static_cast<uint8_t>(accumulatorRowBlocks), 8};
        for (uint64_t m = 0; m < indexSize_; ++m) {
            const uint32_t outputIndex = static_cast<uint32_t>(indexLocal.GetValue(m));
            const uint32_t blockBase = outputIndex / vectorElements * vectorElements;
            const uint32_t lane = outputIndex - blockBase;
            const uint32_t laneBlock = lane / blockElements;
            const uint32_t laneInBlock = lane - laneBlock * blockElements;

            DataCopyExtParams sourceCopyParams{
                static_cast<uint16_t>(outerCount), static_cast<uint32_t>(sizeof(T)),
                static_cast<uint32_t>((indexSize_ - 1) * sizeof(T)), 7, 0};
            DataCopyPadExtParams<T> sourcePadParams{true, static_cast<uint8_t>(laneInBlock),
                static_cast<uint8_t>(blockElements - laneInBlock - 1), static_cast<T>(0)};
            DataCopyPad(sourceLocal[laneBlock * blockElements],
                sourceGm_[outerStart * indexSize_ + m], sourceCopyParams, sourcePadParams);
            Sync<HardEvent::MTE2_V>();

            AddNarrowMasked(accLocal[blockBase], sourceLocal, lane, static_cast<uint8_t>(outerCount), repeatParams);
            Sync<HardEvent::V_MTE2>();
        }

        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams outputCopyParams{
            static_cast<uint16_t>(outerCount), accumulatorValidBytes, 0, 0, 0};
        DataCopyPad(outputGm_[outerStart * axisSize_], accLocal, outputCopyParams);
        Sync<HardEvent::MTE3_MTE2>();
    }

private:
    TPipe* pipe_{nullptr};
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECOUT> dataBuf_;
    TBuf<TPosition::VECIN> sourceBuf_;
    TBuf<TPosition::VECOUT> residentBuf_;
    TBuf<TPosition::VECCALC> counterBuf_;
    TBuf<TPosition::VECCALC> layerOffsetBuf_;
    TBuf<TPosition::VECCALC> denseBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> rowBaseBuf_;
    TBuf<TPosition::VECCALC> batchOffsetBuf_;
    TQue<TPosition::VECIN, 1> bf16SourceQueue_;
    GatherPipelineQueue gatherPipelineFirstQueue_;
    GatherPipelineQueue gatherPipelineSecondQueue_;

    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<uint16_t> workspaceOffsetGm_;

    uint64_t outerSize_{0};
    uint64_t axisSize_{0};
    uint64_t innerSize_{0};
    uint64_t indexSize_{0};
    uint64_t innerTile_{0};
    uint64_t outerTile_{0};
    uint64_t outerBatch_{0};
    uint32_t strategy_{0};
    uint32_t innerTileNum_{0};
    uint32_t taskNum_{0};
    uint32_t sourceChunk_{0};
    uint32_t blockDim_{0};
    uint32_t coopLayerCap_{0};
    uint32_t mixedInnerTile_{0};
    uint32_t mixedInnerTileNum_{0};
    uint32_t mixedOuterCount_{0};
    uint32_t residentBufferBytes_{0};
    uint32_t residentDoubleBuffer_{0};
    uint32_t gatherUbPaddedAxis_{0};
    uint32_t transposeBlockGatherMode_{0};
    uint32_t transposeSourceChunk_{0};
    uint32_t transposeLayerCap_{0};
    uint32_t transposePayloadLayerCap_{0};
    uint32_t transposeOffsetBytes_{0};
    uint32_t gatherSourceOffsetBytes_{0};
    uint32_t gatherDenseOffsetBytes_{0};
    bool cooperative_{false};
    bool useTransposeBlockGather_{false};
};

} // namespace IndexAdd

template <uint32_t schMode>
__global__ __aicore__ void index_add(
    GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (schMode == INDEX_ADD_KEY_COOP) {
        KERNEL_TASK_TYPE(INDEX_ADD_KEY_COOP, KERNEL_TYPE_MIX_AIV_1_0);
        if (workspace == nullptr) {
            return;
        }
        AscendC::TPipe pipe;
        if constexpr (AscendC::IsSameType<DTYPE_SELF, half>::value) {
            if (tilingData.strategy == IndexAdd::STRATEGY_GATHER_BATCH_WIDE) {
                IndexAdd::KernelIndexAdd<DTYPE_SELF, true> op;
                op.Init<false>(self, index, source, output, workspace, &tilingData, &pipe, true);
                op.Process<true>();
                return;
            }
        }
        if constexpr (AscendC::IsSameType<DTYPE_SELF, float>::value) {
            if (tilingData.strategy == IndexAdd::STRATEGY_GATHER_NARROW_DB &&
                tilingData.transposeBlockGather == IndexAdd::TRANSPOSE_BLOCK_GATHER_FP32) {
                IndexAdd::KernelIndexAdd<DTYPE_SELF, true> op;
                op.Init<false>(self, index, source, output, workspace, &tilingData, &pipe, true);
                op.Process<true>();
                return;
            }
        }
        IndexAdd::KernelIndexAdd<DTYPE_SELF> op;
        op.Init<false>(self, index, source, output, workspace, &tilingData, &pipe, true);
        op.Process<true>();
        return;
    }
    if constexpr (schMode == INDEX_ADD_KEY_SOURCE_L2_BYPASS) {
        KERNEL_TASK_TYPE(INDEX_ADD_KEY_SOURCE_L2_BYPASS, KERNEL_TYPE_AIV_ONLY);
        if (AscendC::GetBlockIdx() >= tilingData.taskNum) {
            return;
        }
        AscendC::TPipe pipe;
        IndexAdd::KernelIndexAdd<DTYPE_SELF> op;
        op.Init<true>(self, index, source, output, nullptr, &tilingData, &pipe, false);
        op.Process<false>();
        return;
    }
    if constexpr (schMode == INDEX_ADD_KEY_DEFAULT) {
        KERNEL_TASK_TYPE(INDEX_ADD_KEY_DEFAULT, KERNEL_TYPE_AIV_ONLY);
        if (AscendC::GetBlockIdx() >= tilingData.taskNum) {
            return;
        }
        AscendC::TPipe pipe;
        IndexAdd::KernelIndexAdd<DTYPE_SELF> op;
        op.Init<false>(self, index, source, output, nullptr, &tilingData, &pipe, false);
        op.Process<false>();
    }
}
