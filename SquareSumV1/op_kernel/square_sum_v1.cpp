#include "kernel_operator.h"
#include "square_sum_v1_tiling_key.h"

namespace SquareSum {
using namespace AscendC;

constexpr uint32_t TOPOLOGY_AR = 0;
constexpr uint32_t TOPOLOGY_ARA = 1;
constexpr uint32_t TOPOLOGY_RAR = 2;
constexpr uint32_t TOPOLOGY_RA = 3;
constexpr uint32_t TOPOLOGY_GENERIC = 4;
constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t UB_SKEW_BYTES = 256;
constexpr uint32_t MAX_AR_REUSE_ROWS = 248;
constexpr uint32_t MAX_VECTOR_REPEATS = 255;
constexpr uint32_t MAX_GATHER_ELEMENTS = MAX_VECTOR_REPEATS * 64U;
constexpr uint64_t MAX_COPY_STRIDE = 0xFFFFFFFFULL;
constexpr uint32_t COMPACT_ARA_PHASE_ACCUMULATORS = 2;
constexpr uint32_t COMPACT_RA_TILED = 1;
constexpr uint32_t COMPACT_RA_PHASE = 2;
constexpr uint32_t COMPACT_RA_PHASE_ACCUMULATORS = 1;
constexpr uint32_t COMPACT_RA_PHASE_MAX_SEGMENTS = 2;
constexpr uint32_t COMPACT_RA_PHASE_SCHEDULE_WAVE = 1;
constexpr uint32_t RAR_SCHEDULE_WAVE = 1;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignElements(uint32_t count)
{
    return (count + 7U) / 8U * 8U;
}

__aicore__ inline uint32_t GcdU32(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

template <typename T>
class KernelSquareSum {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR userWorkspace,
        const SquareSumV1TilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        if (tilingData_->bypassL2 != 0) {
            inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if (userWorkspace != nullptr) {
            workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace));
        }

        const bool directArHalf = std::is_same<T, half>::value &&
            tilingData_->topology == TOPOLOGY_AR &&
            tilingData_->alignedInner <= 64U &&
            tilingData_->compactAr == 0;
        if (directArHalf) {
            pipe_->InitBuffer(inputQueue_, 1,
                tilingData_->tilePrimary * tilingData_->alignedInner * sizeof(T) + UB_SKEW_BYTES);
            pipe_->InitBuffer(outputQueue_, 1, UB_BLOCK_BYTES);
            return;
        }

        const bool compactAr = tilingData_->compactAr != 0;
        const bool compactRa = tilingData_->compactRa != 0;
        const bool phaseRa = tilingData_->compactRa == COMPACT_RA_PHASE;
        const bool phaseAra = tilingData_->compactAra == 2;
        const uint32_t inputInner = compactRa ?
            static_cast<uint32_t>(tilingData_->dim1) : tilingData_->alignedInner;
        const uint32_t tileElements = tilingData_->tilePrimary * inputInner;
        const uint32_t vectorElements = tilingData_->tilePrimary > tilingData_->alignedInner ?
            tilingData_->tilePrimary : tilingData_->alignedInner;
        // phase 路径只需一个 32B 尾块承接 DataCopyPad 补齐，并让输入槽与持久 accumulator 错开 bank。
        const uint32_t inputExtraBytes = (phaseAra || phaseRa) ? UB_BLOCK_BYTES : UB_SKEW_BYTES;
        pipe_->InitBuffer(inputQueue_, tilingData_->inputBufferNum,
            tileElements * sizeof(T) + inputExtraBytes);
        const uint32_t phaseInner = phaseRa ?
            static_cast<uint32_t>(tilingData_->dim1) : tilingData_->alignedInner;
        const uint32_t phaseRows = phaseAra ?
            16U / GcdU32(phaseInner, 16U) : (phaseRa ? tilingData_->compactRaPhaseRows : 0U);
        const uint32_t phaseElements = phaseRows * phaseInner;
        if constexpr (!std::is_same<T, float>::value) {
            const uint32_t phaseWorkElements = 3U * AlignElements(tilingData_->alignedInner);
            const uint32_t araWorkElements = COMPACT_ARA_PHASE_ACCUMULATORS * phaseElements;
            const uint32_t workElements = phaseAra ?
                (araWorkElements > phaseWorkElements ? araWorkElements : phaseWorkElements) : tileElements;
            pipe_->InitBuffer(workBuf_, workElements * sizeof(float) +
                (phaseAra ? 0U : UB_SKEW_BYTES));
        }
        const uint32_t compactArRows = 2 * AlignElements(tilingData_->tilePrimary);
        const uint32_t sumElements = compactAr ? compactArRows : vectorElements;
        if (!phaseAra && !phaseRa) {
            pipe_->InitBuffer(sumBuf_, sumElements * sizeof(float) + UB_SKEW_BYTES);
        }
        const uint32_t accElements = compactAr ? compactArRows : (phaseAra ?
            COMPACT_ARA_PHASE_ACCUMULATORS * phaseElements : (phaseRa ?
            COMPACT_RA_PHASE_ACCUMULATORS * phaseElements : vectorElements));
        pipe_->InitBuffer(accBuf_, accElements * sizeof(float) +
            ((phaseAra || phaseRa) ? 0U : UB_SKEW_BYTES));
        if constexpr (std::is_same<T, float>::value) {
            if (phaseRa &&
                tilingData_->compactRaPhaseSegments == COMPACT_RA_PHASE_MAX_SEGMENTS) {
                // 保持 inputQueue→accBuf 的热路径地址不变，两个分段辅助区只追加在 accBuf 后。
                pipe_->InitBuffer(workBuf_,
                    3U * AlignElements(tilingData_->alignedInner) * sizeof(float));
                pipe_->InitBuffer(checkpointBuf_,
                    AlignElements(tilingData_->alignedInner) * sizeof(float));
            }
        }
        if (tilingData_->sharedTmpBytes > 0) {
            pipe_->InitBuffer(sharedTmpBuf_, tilingData_->sharedTmpBytes);
        }
        if (compactAr) {
            const uint32_t blockSumElements = (tileElements + 63U) / 64U * 8U + 8U;
            pipe_->InitBuffer(blockBuf_, blockSumElements * sizeof(float) + UB_SKEW_BYTES);
            const uint32_t packStride = AlignElements(tilingData_->alignedInner / 8U);
            pipe_->InitBuffer(compactBuf_,
                tilingData_->tilePrimary * packStride * sizeof(float) + UB_SKEW_BYTES);
        }
        if (!compactRa) {
            const bool isTelescopingBf16 = std::is_same<T, bfloat16_t>::value &&
                tilingData_->topology == TOPOLOGY_AR &&
                (tilingData_->alignedInner & 7U) == 1U &&
                ((tilingData_->alignedInner / 8U) % 64U) == 0U;
            const uint32_t telescopingMaxRows = isTelescopingBf16 && tilingData_->blockDim > 0 ?
                static_cast<uint32_t>((tilingData_->dim0 / 16U / tilingData_->blockDim + 2U) * 16U) : 0U;
            const uint32_t outputElements = isTelescopingBf16 ? telescopingMaxRows :
                (compactAr ? tilingData_->tilePrimary :
                (phaseAra ? tilingData_->alignedInner : vectorElements));
            pipe_->InitBuffer(outputQueue_, 1,
                ((outputElements * sizeof(T) + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->topology == TOPOLOGY_AR) {
            ProcessAR();
        } else if (tilingData_->topology == TOPOLOGY_ARA) {
            ProcessARA();
        } else if (tilingData_->topology == TOPOLOGY_RAR) {
            ProcessRAR();
        } else if (tilingData_->topology == TOPOLOGY_GENERIC) {
            ProcessGeneric();
        }
    }

    __aicore__ inline void ProcessSync()
    {
        ProcessRA();
    }

    __aicore__ inline void ProcessCompactArDirect()
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            const uint32_t inner = static_cast<uint32_t>(tilingData_->dim1);
            const uint32_t packStride = AlignElements(inner / 8U);
            if ((inner & 7U) == 1U && ((inner / 8U) % 64U) == 0U && packStride >= 64U) {
                ProcessARCompactTelescoping();
                return;
            }
            ProcessARCompactDirect();
        }
    }

    __aicore__ inline void ProcessARCompactTelescoping()
    {
        const uint64_t normalSize = tilingData_->dim0;
        const uint32_t inner = static_cast<uint32_t>(tilingData_->dim1);
        const uint32_t Q = inner / 8U;
        const uint32_t repeatsPerInterval = Q / 64U;
        const uint32_t alignedInner = AlignElements(inner);

        const uint64_t blockDim = tilingData_->blockDim;
        const uint64_t physicalBlockIdx = GetBlockIdx();
        uint64_t blockIdx = physicalBlockIdx;
        if ((blockDim & 1U) == 0 && blockDim >= 4U) {
            const uint64_t pairCount = blockDim / 2U;
            const uint64_t pairRank = physicalBlockIdx / 2U;
            const uint64_t pairLowerHalf = (pairCount + 1U) / 2U;
            const uint64_t logicalPairRank = (pairRank & 1U) == 0 ?
                pairRank / 2U : pairLowerHalf + pairRank / 2U;
            blockIdx = (physicalBlockIdx & 1U) == 0 ?
                logicalPairRank : pairCount + logicalPairRank;
        }
        const uint64_t quantum = 16U;
        const uint64_t totalQuanta = normalSize / quantum;
        const uint64_t baseQuanta = totalQuanta / blockDim;
        const uint64_t extraQuanta = totalQuanta % blockDim;
        uint64_t coreStart = (blockIdx * baseQuanta + MinU64(blockIdx, extraQuanta)) * quantum;
        uint64_t coreEnd = coreStart + (baseQuanta + (blockIdx < extraQuanta ? 1U : 0U)) * quantum;
        if (blockIdx + 1U == blockDim) {
            coreEnd = normalSize;
        }

        if (coreStart >= coreEnd) {
            return;
        }
        const uint32_t totalCoreRows = static_cast<uint32_t>(coreEnd - coreStart);
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();

        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - rowBase));
        CopyInCompact(rowBase * inner, rows, inner);

        const uint32_t maxTileRows = tilingData_->tilePrimary;
        const uint32_t alignedRows = AlignElements(maxTileRows);

        LocalTensor<float> blockSums = blockBuf_.Get<float>();
        LocalTensor<float> intervalSums = compactBuf_.Get<float>();

        LocalTensor<float> S = sumBuf_.Get<float>();
        LocalTensor<float> tail = sumBuf_.Get<float>()[alignedRows];
        LocalTensor<float> prevTail = accBuf_.Get<float>();
        LocalTensor<uint32_t> prevTailOffsets = sharedTmpBuf_.Get<uint32_t>();

        if (maxTileRows == 16U) {
            prevTailOffsets.SetValue(0, 64U);
            prevTailOffsets.SetValue(1, 0U);
            prevTailOffsets.SetValue(2, 4U);
            prevTailOffsets.SetValue(3, 8U);
            prevTailOffsets.SetValue(4, 12U);
            prevTailOffsets.SetValue(5, 16U);
            prevTailOffsets.SetValue(6, 20U);
            prevTailOffsets.SetValue(7, 24U);
            prevTailOffsets.SetValue(8, 64U);
            prevTailOffsets.SetValue(9, 32U);
            prevTailOffsets.SetValue(10, 36U);
            prevTailOffsets.SetValue(11, 40U);
            prevTailOffsets.SetValue(12, 44U);
            prevTailOffsets.SetValue(13, 48U);
            prevTailOffsets.SetValue(14, 52U);
            prevTailOffsets.SetValue(15, 56U);
        } else {
            for (uint32_t r = 0; r < maxTileRows; ++r) {
                if ((r & 7U) == 0U) {
                    prevTailOffsets.SetValue(r, alignedRows * static_cast<uint32_t>(sizeof(float)));
                } else {
                    prevTailOffsets.SetValue(r, (r - 1U) * static_cast<uint32_t>(sizeof(float)));
                }
            }
        }

        Duplicate<float>(tail, 0.0f, alignedRows + 8U);
        PipeBarrier<PIPE_V>();

        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                CopyInCompact(nextBase * inner, nextRows, inner);
            }

            SquareInputCompact(inputLocal, rows * inner);
            LocalTensor<float> source = GetSquaredTensor(inputLocal);
            inputQueue_.FreeTensor(inputLocal);

            if (rows == 16U) {
                BlockReduceSum<float>(blockSums, source, 128, 64, 1, 1, 8);
                BlockReduceSum<float>(blockSums[alignedInner], source[8U * inner], 128, 64, 1, 1, 8);

                WholeReduceSum<float>(tail, source[1024U], 1, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[1U], source[2048U], 2, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[2U], source[3072U], 3, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[3U], source[4096U], 4, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[4U], source[5120U], 5, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[5U], source[6144U], 6, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[6U], source[7168U], 7, 2, 8, 1, 1025);
                WholeReduceSum<float>(tail[7U], source[8192U], 8, 2, 8, 1, 1025);

                PipeBarrier<PIPE_V>();

                WholeReduceSum<float>(intervalSums, blockSums, 64, 16, 1, 1, 8);
                WholeReduceSum<float>(intervalSums[16U], blockSums[alignedInner], 64, 16, 1, 1, 8);

                PairReduceSum<float>(S, intervalSums, 1, 32, 1, 1, 8);
            } else {
                const uint32_t numGroups = (rows + 7U) / 8U;
                for (uint32_t g = 0; g < numGroups; ++g) {
                    const uint32_t groupStart = g * 8U;
                    const uint32_t groupRows = static_cast<uint32_t>(MinU64(8U, rows - groupStart));
                    const uint32_t groupSourceOffset = g * 8U * inner;
                    const uint32_t groupBlockOffset = g * alignedInner;

                    const uint32_t groupElements = groupRows * inner;
                    const uint32_t fullRepeats = groupElements / 64U;
                    if (fullRepeats > 0) {
                        BlockReduceSum<float>(blockSums[groupBlockOffset], source[groupSourceOffset],
                            static_cast<int32_t>(fullRepeats), 64, 1, 1, 8);
                    }
                }
                PipeBarrier<PIPE_V>();

                for (uint32_t g = 0; g < numGroups; ++g) {
                    const uint32_t groupStart = g * 8U;
                    const uint32_t groupRows = static_cast<uint32_t>(MinU64(8U, rows - groupStart));
                    const uint32_t groupBlockOffset = g * alignedInner;
                    const uint32_t intervalOffset = g * 16U;
                    WholeReduceSum<float>(intervalSums[intervalOffset], blockSums[groupBlockOffset], 64,
                        static_cast<int32_t>(groupRows * repeatsPerInterval), 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();

                for (uint32_t g = 0; g < numGroups; ++g) {
                    const uint32_t groupStart = g * 8U;
                    const uint32_t groupRows = static_cast<uint32_t>(MinU64(8U, rows - groupStart));
                    const uint32_t intervalOffset = g * 16U;
                    PairReduceSum<float>(S[groupStart], intervalSums[intervalOffset], 1,
                        static_cast<int32_t>(groupRows * repeatsPerInterval), 1, 1, 8);
                }

                const uint32_t phaseRows = 8U;
                const uint32_t srcRepStride = inner;
                for (uint32_t firstRow = 0; firstRow < phaseRows && firstRow < rows; ++firstRow) {
                    const uint32_t classRows = (rows - firstRow + phaseRows - 1U) / phaseRows;
                    const uint32_t phase = firstRow * inner % 8U;
                    const uint32_t endPhase = (phase + inner) % 8U;
                    if (endPhase != 0) {
                        uint64_t mask[2] = {(1ULL << endPhase) - 1ULL, 0};
                        WholeReduceSum<float>(tail[firstRow], source[firstRow * inner + inner - endPhase],
                            mask, static_cast<int32_t>(classRows),
                            static_cast<int32_t>(phaseRows), 1, static_cast<int32_t>(srcRepStride));
                    } else {
                        WholeReduceSum<float>(tail[firstRow], source[firstRow * inner + inner - 8U], 8,
                            static_cast<int32_t>(classRows), static_cast<int32_t>(phaseRows), 1,
                            static_cast<int32_t>(srcRepStride));
                    }
                }
                PipeBarrier<PIPE_V>();
            }

            // 4. Gather prevTail from tail in Vector SIMD
            Gather<float>(prevTail, tail, prevTailOffsets, 0, rows);

            // 5. Final result = S - prevTail + tail (computed in-place in prevTail)
            Sub<float>(prevTail, S, prevTail, rows);
            Add<float>(prevTail, prevTail, tail, rows);

            const uint32_t outOffset = static_cast<uint32_t>(rowBase - coreStart);
            Cast<T, float>(outputLocal[outOffset], prevTail, RoundMode::CAST_RINT, rows);
            rowBase = nextBase;
            rows = nextRows;
        }

        outputQueue_.EnQue(outputLocal);
        outputLocal = outputQueue_.DeQue<T>();
        const uint32_t totalBytes = totalCoreRows * static_cast<uint32_t>(sizeof(T));
        if (totalBytes % UB_BLOCK_BYTES == 0) {
            DataCopy(outputGm_[coreStart], outputLocal, totalCoreRows);
        } else {
            DataCopyExtParams copyParams{1, totalBytes, 0, 0, 0};
            DataCopyPad(outputGm_[coreStart], outputLocal, copyParams);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    __aicore__ inline void SyncVToS()
    {
        const event_t eventId = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncSToV()
    {
        const event_t eventId = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        const event_t eventId = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        const event_t eventId = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte2()
    {
        const event_t eventId = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SplitRange(uint64_t total, uint64_t& start, uint64_t& end) const
    {
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t base = total / tilingData_->blockDim;
        const uint64_t extra = total % tilingData_->blockDim;
        start = blockIdx * base + MinU64(blockIdx, extra);
        end = start + base + (blockIdx < extra ? 1 : 0);
    }

    __aicore__ inline void SplitRangeAligned(uint64_t total, uint32_t quantum,
        uint64_t& start, uint64_t& end) const
    {
        start = 0;
        end = 0;
        const uint64_t blockDim = tilingData_->blockDim;
        if (blockDim == 0 || quantum == 0) {
            return;
        }
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t totalQuanta = (total + quantum - 1U) / quantum;
        const uint64_t baseQuanta = totalQuanta / blockDim;
        const uint64_t extraQuanta = totalQuanta % blockDim;
        const uint64_t startQuanta = blockIdx * baseQuanta + MinU64(blockIdx, extraQuanta);
        const uint64_t endQuanta = startQuanta + baseQuanta + (blockIdx < extraQuanta ? 1U : 0U);
        start = MinU64(total, startQuanta * quantum);
        end = MinU64(total, endQuanta * quantum);
    }

    __aicore__ inline void SplitRangeAlignedIdx(uint64_t total, uint32_t quantum,
        uint64_t blockIdx, uint64_t blockDim, uint64_t& start, uint64_t& end) const
    {
        start = 0;
        end = 0;
        if (blockDim == 0 || quantum == 0) {
            return;
        }
        const uint64_t totalQuanta = (total + quantum - 1U) / quantum;
        const uint64_t baseQuanta = totalQuanta / blockDim;
        const uint64_t extraQuanta = totalQuanta % blockDim;
        const uint64_t startQuanta = blockIdx * baseQuanta + MinU64(blockIdx, extraQuanta);
        const uint64_t endQuanta = startQuanta + baseQuanta + (blockIdx < extraQuanta ? 1U : 0U);
        start = MinU64(total, startQuanta * quantum);
        end = MinU64(total, endQuanta * quantum);
    }

    __aicore__ inline bool SplitRangePhaseGroups(uint64_t total, uint32_t phaseRows,
        uint64_t& start, uint64_t& end) const
    {
        start = 0;
        end = 0;
        const uint64_t blockDim = tilingData_->blockDim;
        const uint64_t physicalBlockIdx = GetBlockIdx();
        if (phaseRows == 0 || blockDim == 0 || physicalBlockIdx >= blockDim) {
            return false;
        }
        // 相邻物理核交错领取逻辑低/高半区，打散同相位的大块连续读。
        // 偶数核时再对物理核对编号做一次半区交织，使相邻核对继续分散到不同四分区；
        // 每个逻辑区间内部仍保持单调连续访问，且映射始终为一一置换。
        const uint64_t lowerHalf = (blockDim + 1U) / 2U;
        uint64_t blockIdx = 0;
        if ((blockDim & 1U) == 0) {
            const uint64_t pairCount = blockDim / 2U;
            const uint64_t pairRank = physicalBlockIdx / 2U;
            const uint64_t pairLowerHalf = (pairCount + 1U) / 2U;
            const uint64_t logicalPairRank = (pairRank & 1U) == 0 ?
                pairRank / 2U : pairLowerHalf + pairRank / 2U;
            blockIdx = (physicalBlockIdx & 1U) == 0 ?
                logicalPairRank : pairCount + logicalPairRank;
        } else {
            blockIdx = (physicalBlockIdx & 1U) == 0 ?
                physicalBlockIdx / 2U : lowerHalf + physicalBlockIdx / 2U;
        }
        const uint64_t fullGroups = total / phaseRows;
        const uint64_t residualRows = total % phaseRows;
        if (fullGroups < blockDim) {
            return false;
        }
        const uint64_t baseGroups = fullGroups / blockDim;
        const uint64_t extraGroups = fullGroups % blockDim;
        const uint64_t groupStart = blockIdx * baseGroups + MinU64(blockIdx, extraGroups);
        const uint64_t groupCount = baseGroups + (blockIdx < extraGroups ? 1U : 0U);
        if (groupStart > fullGroups || groupCount > fullGroups - groupStart) {
            return false;
        }
        start = groupStart * phaseRows;
        end = start + groupCount * phaseRows;
        if (blockIdx + 1U == blockDim) {
            if (end > total - residualRows) {
                return false;
            }
            end += residualRows;
        }
        return start <= end && end <= total;
    }

    __aicore__ inline void CopyIn(uint64_t gmOffset, uint32_t rows,
        uint32_t validInner, uint64_t gmRowStride)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        const uint64_t gapBytes = (gmRowStride - validInner) * sizeof(T);
        DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(tilingData_->alignedInner - validInner), static_cast<T>(0)};
        if (gapBytes <= MAX_COPY_STRIDE) {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(validInner * sizeof(T)),
                static_cast<uint32_t>(gapBytes), 0, 0};
            DataCopyPad(inputLocal, inputGm_[gmOffset], copyParams, padParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(validInner * sizeof(T)), 0, 0, 0};
            for (uint32_t row = 0; row < rows; ++row) {
                DataCopyPad(inputLocal[row * tilingData_->alignedInner],
                    inputGm_[gmOffset + static_cast<uint64_t>(row) * gmRowStride], copyParams, padParams);
            }
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void CopyInCompactTo(LocalTensor<T> inputLocal,
        uint64_t gmOffset, uint32_t rows, uint32_t validInner)
    {
        const uint32_t count = rows * validInner;
        const uint32_t copyBytes = count * sizeof(T);
        const bool gmAligned = (gmOffset * sizeof(T)) % UB_BLOCK_BYTES == 0;
        if (gmAligned && copyBytes % UB_BLOCK_BYTES == 0) {
            // 基础搬运要求 GM 起址和长度都满足 32B 对齐；否则使用补齐搬运。
            DataCopy(inputLocal, inputGm_[gmOffset], count);
            return;
        }
        const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
        const uint32_t alignedCount = (count + blockElements - 1) / blockElements * blockElements;
        DataCopyExtParams copyParams{
            1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(alignedCount - count), static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyInCompact(uint64_t gmOffset, uint32_t rows, uint32_t validInner)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        CopyInCompactTo(inputLocal, gmOffset, rows, validInner);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void SquareInput(LocalTensor<T> inputLocal, uint32_t rows)
    {
        const uint32_t count = rows * tilingData_->alignedInner;
        if constexpr (std::is_same<T, float>::value) {
            Mul<float>(inputLocal, inputLocal, inputLocal, count);
        } else {
            LocalTensor<float> workLocal = workBuf_.Get<float>();
            Cast<float, T>(workLocal, inputLocal, RoundMode::CAST_NONE, count);
            Mul<float>(workLocal, workLocal, workLocal, count);
        }
    }

    __aicore__ inline void SquareInputCompact(LocalTensor<T> inputLocal, uint32_t count)
    {
        if constexpr (std::is_same<T, float>::value) {
            Mul<float>(inputLocal, inputLocal, inputLocal, count);
        } else {
            LocalTensor<float> workLocal = workBuf_.Get<float>();
            Cast<float, T>(workLocal, inputLocal, RoundMode::CAST_NONE, count);
            Mul<float>(workLocal, workLocal, workLocal, count);
        }
    }

    __aicore__ inline LocalTensor<T> CopyAndSquare(uint64_t gmOffset, uint32_t rows,
        uint32_t validInner, uint64_t gmRowStride)
    {
        CopyIn(gmOffset, rows, validInner, gmRowStride);
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        SquareInput(inputLocal, rows);
        return inputLocal;
    }

    __aicore__ inline LocalTensor<float> GetSquaredTensor(const LocalTensor<T>& inputLocal)
    {
        if constexpr (std::is_same<T, float>::value) {
            return inputLocal;
        }
        return workBuf_.Get<float>();
    }

    __aicore__ inline LocalTensor<uint8_t> GetReduceTmpTensor()
    {
        if (tilingData_->sharedTmpBytes > 0) {
            return sharedTmpBuf_.Get<uint8_t>();
        }
        const uint32_t vectorElements = tilingData_->tilePrimary > tilingData_->alignedInner ?
            tilingData_->tilePrimary : tilingData_->alignedInner;
        const uint32_t scratchOffset = AlignElements(vectorElements) * sizeof(float);
        // min tmp 为 0 时复用 sumBuf_ 的预留尾部，保持 LocalTensor 地址有效且不与结果重叠。
        return sumBuf_.Get<uint8_t>()[scratchOffset];
    }

    __aicore__ inline LocalTensor<float> ReduceAR(uint32_t rows, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        if (tilingData_->topology == TOPOLOGY_AR && tilingData_->alignedInner <= 64U) {
            const uint32_t alignedInner = tilingData_->alignedInner;
            const uint32_t srcRepStride = alignedInner / 8U;
            const int32_t validInner = static_cast<int32_t>(tilingData_->dim1);
            for (uint32_t repeatBase = 0; repeatBase < rows; repeatBase += MAX_VECTOR_REPEATS) {
                const uint32_t repeats = static_cast<uint32_t>(
                    MinU64(MAX_VECTOR_REPEATS, rows - repeatBase));
                WholeReduceSum<float>(sumLocal[repeatBase], source[repeatBase * alignedInner],
                    validInner, static_cast<int32_t>(repeats), 1, 1,
                    static_cast<int32_t>(srcRepStride));
            }
            return sumLocal;
        }
        LocalTensor<uint8_t> sharedTmp = GetReduceTmpTensor();
        // v220 的 AR 源复用路径以 248 行为内部批次，外层分段可保证带 padding 的多批次偏移正确。
        for (uint32_t rowBase = 0; rowBase < rows; rowBase += MAX_AR_REUSE_ROWS) {
            const uint32_t currentRows = static_cast<uint32_t>(
                MinU64(MAX_AR_REUSE_ROWS, rows - rowBase));
            uint32_t shape[2] = {currentRows, tilingData_->alignedInner};
            ReduceSum<float, Pattern::Reduce::AR, true>(sumLocal[rowBase],
                source[rowBase * tilingData_->alignedInner], sharedTmp, shape, true);
        }
        PipeBarrier<PIPE_V>();
        return sumLocal;
    }

    __aicore__ inline LocalTensor<float> ReduceRA(uint32_t rows, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        LocalTensor<uint8_t> sharedTmp = GetReduceTmpTensor();
        uint32_t shape[2] = {rows, tilingData_->alignedInner};
        ReduceSum<float, Pattern::Reduce::RA, true>(
            sumLocal, source, sharedTmp, shape, true);
        PipeBarrier<PIPE_V>();
        return sumLocal;
    }

    __aicore__ inline LocalTensor<float> ReduceRAInPlace(uint32_t rows, LocalTensor<float> source)
    {
        BinaryRepeatParams repeatParams;
        uint32_t splitRows = 1;
        while ((splitRows << 1U) <= rows) {
            splitRows <<= 1U;
        }
        const uint32_t remainRows = rows - splitRows;
        SetMaskCount();
        if (remainRows > 0) {
            SetVectorMask<float, MaskMode::COUNTER>(
                0, remainRows * tilingData_->alignedInner);
            Add<float, false>(source, source,
                source[splitRows * tilingData_->alignedInner], MASK_PLACEHOLDER, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        while (splitRows > 1) {
            splitRows >>= 1U;
            SetVectorMask<float, MaskMode::COUNTER>(
                0, splitRows * tilingData_->alignedInner);
            Add<float, false>(source, source,
                source[splitRows * tilingData_->alignedInner], MASK_PLACEHOLDER, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        ResetMask();
        return source;
    }

    __aicore__ inline LocalTensor<float> ReduceCompactAR(
        uint32_t rows, uint32_t inner, LocalTensor<float> source)
    {
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        LocalTensor<float> partialLocal = accBuf_.Get<float>();
        const uint32_t gcdVal = (inner & 1U) != 0 ? 1U : ((inner & 2U) != 0 ? 2U : ((inner & 4U) != 0 ? 4U : 8U));
        const uint32_t phaseStep = 8U / gcdVal;
        const uint32_t srcRepStride = inner / gcdVal;
        const uint32_t chunkCount = (inner + 7U + 63U) / 64U;
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t chunkBegin = chunk * 64U;
            const uint32_t chunkEnd = chunkBegin + 64U;
            if (chunk > 0 && inner <= chunkBegin) {
                Duplicate<float>(partialLocal, 0.0f, AlignElements(rows));
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t firstRow = 0; firstRow < phaseStep && firstRow < rows; ++firstRow) {
                const uint32_t phase = firstRow * inner % 8U;
                const uint32_t validBegin = phase > chunkBegin ? phase : chunkBegin;
                const uint32_t rowEnd = phase + inner;
                const uint32_t validEnd = rowEnd < chunkEnd ? rowEnd : chunkEnd;
                if (validBegin >= validEnd) {
                    continue;
                }
                const uint32_t maskOffset = validBegin - chunkBegin;
                const uint32_t maskCount = validEnd - validBegin;
                const uint64_t lowBits = maskCount == 64U ? UINT64_MAX : ((1ULL << maskCount) - 1ULL);
                uint64_t mask[2] = {lowBits << maskOffset, 0};
                const uint32_t phaseRows = (rows - firstRow + phaseStep - 1U) / phaseStep;
                if (phaseRows <= MAX_VECTOR_REPEATS) {
                    const uint32_t alignedBase = (firstRow * inner) & ~7U;
                    if (chunk == 0) {
                        WholeReduceSum<float>(sumLocal[firstRow], source[alignedBase + chunkBegin], mask,
                            static_cast<int32_t>(phaseRows), static_cast<int32_t>(phaseStep), 1,
                            static_cast<int32_t>(srcRepStride));
                    } else {
                        WholeReduceSum<float>(partialLocal[firstRow], source[alignedBase + chunkBegin], mask,
                            static_cast<int32_t>(phaseRows), static_cast<int32_t>(phaseStep), 1,
                            static_cast<int32_t>(srcRepStride));
                    }
                } else {
                    for (uint32_t repeatBase = 0; repeatBase < phaseRows;
                         repeatBase += MAX_VECTOR_REPEATS) {
                        const uint32_t repeats = static_cast<uint32_t>(
                            MinU64(MAX_VECTOR_REPEATS, phaseRows - repeatBase));
                        const uint32_t row = firstRow + repeatBase * phaseStep;
                        const uint32_t alignedBase = row * inner - phase;
                        if (chunk == 0) {
                            WholeReduceSum<float>(sumLocal[row], source[alignedBase + chunkBegin], mask,
                                static_cast<int32_t>(repeats), static_cast<int32_t>(phaseStep), 1,
                                static_cast<int32_t>(srcRepStride));
                        } else {
                            WholeReduceSum<float>(partialLocal[row], source[alignedBase + chunkBegin], mask,
                                static_cast<int32_t>(repeats), static_cast<int32_t>(phaseStep), 1,
                                static_cast<int32_t>(srcRepStride));
                        }
                    }
                }
            }
            PipeBarrier<PIPE_V>();
            if (chunk > 0) {
                Add<float>(sumLocal, sumLocal, partialLocal, rows);
            }
        }
        return sumLocal;
    }

    __aicore__ inline void StoreOutput(uint64_t gmOffset, LocalTensor<float> source, uint32_t count)
    {
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds<float>(outputLocal, source, 0.0f, count);
        } else {
            Cast<T, float>(outputLocal, source, RoundMode::CAST_RINT, count);
        }
        outputQueue_.EnQue(outputLocal);
        outputLocal = outputQueue_.DeQue<T>();
        const uint32_t copyBytes = static_cast<uint32_t>(count * sizeof(T));
        if ((gmOffset * sizeof(T)) % UB_BLOCK_BYTES == 0 && copyBytes % UB_BLOCK_BYTES == 0) {
            DataCopy(outputGm_[gmOffset], outputLocal, count);
        } else {
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPad(outputGm_[gmOffset], outputLocal, copyParams);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ProcessAR()
    {
        if (tilingData_->compactAr != 0) {
            ProcessARCompact();
            return;
        }
        const uint64_t normalSize = tilingData_->dim0;
        const uint64_t reduceSize = tilingData_->dim1;
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        const uint32_t quantum = UB_BLOCK_BYTES / sizeof(T);
        SplitRangeAligned(normalSize, quantum, coreStart, coreEnd);
        if (coreStart >= coreEnd) {
            return;
        }
        if (coreEnd - coreStart <= tilingData_->tilePrimary) {
            const uint32_t rows = static_cast<uint32_t>(coreEnd - coreStart);
            if constexpr (std::is_same<T, half>::value) {
                if (tilingData_->alignedInner <= 64U) {
                    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
                    LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
                    DataCopyExtParams copyParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(reduceSize * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> padParams{
                        true, 0, static_cast<uint8_t>(tilingData_->alignedInner - reduceSize), static_cast<T>(0)};
                    DataCopyPad(inputLocal, inputGm_[coreStart * reduceSize], copyParams, padParams);
                    inputQueue_.EnQue(inputLocal);
                    inputLocal = inputQueue_.DeQue<T>();
                    const uint32_t alignedInner = tilingData_->alignedInner;
                    Mul<half>(inputLocal, inputLocal, inputLocal, rows * alignedInner);
                    WholeReduceSum<half>(outputLocal, inputLocal, static_cast<int32_t>(reduceSize),
                        static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(alignedInner / 16U));
                    inputQueue_.FreeTensor(inputLocal);
                    outputQueue_.EnQue(outputLocal);
                    outputLocal = outputQueue_.DeQue<T>();
                    if (rows * sizeof(T) == UB_BLOCK_BYTES) {
                        DataCopy(outputGm_[coreStart], outputLocal, rows);
                    } else {
                        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
                        DataCopyPad(outputGm_[coreStart], outputLocal, outCopyParams);
                    }
                    outputQueue_.FreeTensor(outputLocal);
                    return;
                }
            }
            CopyIn(coreStart * reduceSize, rows, static_cast<uint32_t>(reduceSize), reduceSize);
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            SquareInput(inputLocal, rows);
            LocalTensor<float> partial = ReduceAR(rows, GetSquaredTensor(inputLocal));
            inputQueue_.FreeTensor(inputLocal);
            StoreOutput(coreStart, partial, rows);
            return;
        }
        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - rowBase));
        CopyIn(rowBase * reduceSize, rows, static_cast<uint32_t>(reduceSize), reduceSize);
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                CopyIn(nextBase * reduceSize, nextRows, static_cast<uint32_t>(reduceSize), reduceSize);
            }
            SquareInput(inputLocal, rows);
            LocalTensor<float> partial = ReduceAR(rows, GetSquaredTensor(inputLocal));
            inputQueue_.FreeTensor(inputLocal);
            StoreOutput(rowBase, partial, rows);
            if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                CopyIn(nextBase * reduceSize, nextRows, static_cast<uint32_t>(reduceSize), reduceSize);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline LocalTensor<float> CompressARBlocks(
        LocalTensor<float> source, uint32_t rows, uint32_t inner)
    {
        const uint32_t totalElements = rows * inner;
        const uint32_t fullRepeats = totalElements / 64U;
        const uint32_t tailElements = totalElements % 64U;
        LocalTensor<float> blockSums = blockBuf_.Get<float>();
        blockSums.SetSize((fullRepeats + (tailElements > 0 ? 1U : 0U)) * 8U);
        for (uint32_t repeatBase = 0; repeatBase < fullRepeats;
             repeatBase += MAX_VECTOR_REPEATS) {
            const uint32_t repeats = static_cast<uint32_t>(
                MinU64(MAX_VECTOR_REPEATS, fullRepeats - repeatBase));
            BlockReduceSum<float>(blockSums[repeatBase * 8U], source[repeatBase * 64U],
                static_cast<int32_t>(repeats), 64, 1, 1, 8);
        }
        if (tailElements > 0) {
            BlockReduceSum<float>(blockSums[fullRepeats * 8U], source[fullRepeats * 64U],
                1, static_cast<int32_t>(tailElements), 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        return blockSums;
    }

    __aicore__ inline void InitARPackOffsets(LocalTensor<uint32_t> gatherOffsets,
        uint32_t inner, uint32_t packStride, uint32_t zeroBlock, uint32_t initRows)
    {
        const uint32_t offsetCount = initRows * packStride;
        Duplicate<uint32_t>(gatherOffsets, zeroBlock * sizeof(float), offsetCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < initRows; ++row) {
            const uint32_t rowBegin = row * inner;
            const uint32_t rowEnd = rowBegin + inner;
            const uint32_t firstBlock = (rowBegin + 7U) / 8U;
            const uint32_t lastBlock = rowEnd / 8U;
            const uint32_t blockCount = lastBlock - firstBlock;
            const uint32_t alignedBlockCount = blockCount / 8U * 8U;
            if (alignedBlockCount > 0) {
                Arange<int32_t>(gatherOffsets[row * packStride].ReinterpretCast<int32_t>(),
                    static_cast<int32_t>(firstBlock * sizeof(float)),
                    static_cast<int32_t>(sizeof(float)), alignedBlockCount);
            }
            // 单独生成不足一个 DataBlock 的短尾，避免覆盖预置的零槽 padding offset。
            if (alignedBlockCount < blockCount) {
                Arange<int32_t>(gatherOffsets[row * packStride + alignedBlockCount].ReinterpretCast<int32_t>(),
                    static_cast<int32_t>((firstBlock + alignedBlockCount) * sizeof(float)),
                    static_cast<int32_t>(sizeof(float)), blockCount - alignedBlockCount);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildARRowPack(LocalTensor<float> rowPack,
        LocalTensor<float> blockSums, LocalTensor<uint32_t> gatherOffsets,
        uint32_t rows, uint32_t packStride)
    {
        const uint32_t count = rows * packStride;
        for (uint32_t base = 0; base < count; base += MAX_GATHER_ELEMENTS) {
            const uint32_t current = static_cast<uint32_t>(
                MinU64(MAX_GATHER_ELEMENTS, count - base));
            Gather<float>(rowPack[base], blockSums, gatherOffsets[base], 0, current);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildARRowPackDirect(LocalTensor<float> rowPack,
        LocalTensor<float> source, uint32_t rows, uint32_t inner, uint32_t packStride)
    {
        // 对大 BF16 紧凑 AR，按行直接压缩完整 DataBlock，避免整 tile 的中间块表和 Gather。
        // 行首/行尾边界仍由 ReduceAREdges 处理，块内归约顺序保持不变。
        Duplicate<float>(rowPack, 0.0f, rows * packStride);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rowBegin = row * inner;
            const uint32_t rowFirstBlock = (rowBegin + 7U) / 8U;
            const uint32_t rowLastBlock = (rowBegin + inner) / 8U;
            const uint32_t rowBlockCount = rowLastBlock - rowFirstBlock;
            const uint32_t rowFullGroups = rowBlockCount / 8U;
            const uint32_t rowTailBlocks = rowBlockCount % 8U;
            if (rowFullGroups > 0) {
                BlockReduceSum<float>(rowPack[row * packStride],
                    source[rowFirstBlock * 8U], static_cast<int32_t>(rowFullGroups), 64, 1, 1, 8);
            }
            if (rowTailBlocks > 0) {
                BlockReduceSum<float>(rowPack[row * packStride + rowFullGroups * 8U],
                    source[(rowFirstBlock + rowFullGroups * 8U) * 8U], 1,
                    static_cast<int32_t>(rowTailBlocks * 8U), 1, 1, 8);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceAREdges(LocalTensor<float> head,
        LocalTensor<float> tail, LocalTensor<float> source, uint32_t rows, uint32_t inner)
    {
        const uint32_t phaseRows = 8U / GcdU32(inner, 8U);
        const uint32_t srcRepStride = inner / GcdU32(inner, 8U);
        const uint32_t alignedRows = AlignElements(rows);
        Duplicate<float>(head, 0.0f, alignedRows);
        Duplicate<float>(tail, 0.0f, alignedRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t firstRow = 0; firstRow < phaseRows && firstRow < rows; ++firstRow) {
            const uint32_t phase = firstRow * inner % 8U;
            const uint32_t endPhase = (phase + inner) % 8U;
            const uint32_t classRows = (rows - firstRow + phaseRows - 1U) / phaseRows;
            const uint64_t lowBitsHead = phase != 0 ? (1ULL << (8U - phase)) - 1ULL : 0ULL;
            const uint64_t maskHead[2] = {lowBitsHead << phase, 0};
            const uint64_t lowBitsTail = endPhase != 0 ? (1ULL << endPhase) - 1ULL : 0ULL;
            const uint64_t maskTail[2] = {lowBitsTail, 0};
            for (uint32_t repeatBase = 0; repeatBase < classRows;
                 repeatBase += MAX_VECTOR_REPEATS) {
                const uint32_t repeats = static_cast<uint32_t>(
                    MinU64(MAX_VECTOR_REPEATS, classRows - repeatBase));
                const uint32_t row = firstRow + repeatBase * phaseRows;
                if (phase != 0) {
                    WholeReduceSum<float>(head[row], source[row * inner - phase], maskHead,
                        static_cast<int32_t>(repeats), static_cast<int32_t>(phaseRows), 1,
                        static_cast<int32_t>(srcRepStride));
                }
                if (endPhase != 0) {
                    WholeReduceSum<float>(tail[row], source[row * inner + inner - endPhase], maskTail,
                        static_cast<int32_t>(repeats), static_cast<int32_t>(phaseRows), 1,
                        static_cast<int32_t>(srcRepStride));
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> ReduceARRowPack(LocalTensor<float> rowPack,
        LocalTensor<float> head, LocalTensor<float> tail, uint32_t rows, uint32_t packStride)
    {
        const uint32_t alignedRows = AlignElements(rows);
        LocalTensor<float> result = accBuf_.Get<float>();
        LocalTensor<float> partial = accBuf_.Get<float>()[alignedRows];
        for (uint32_t column = 0; column < packStride; column += 64U) {
            const uint32_t count = static_cast<uint32_t>(MinU64(64U, packStride - column));
            LocalTensor<float> destination = column == 0 ? result : partial;
            WholeReduceSum<float>(destination, rowPack[column], static_cast<int32_t>(count),
                static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(packStride / 8U));
            if (column != 0) {
                PipeBarrier<PIPE_V>();
                Add<float>(result, result, partial, rows);
            }
        }
        PipeBarrier<PIPE_V>();
        Add<float>(result, result, head, rows);
        PipeBarrier<PIPE_V>();
        Add<float>(result, result, tail, rows);
        return result;
    }

    __aicore__ inline void ProcessARCompact()
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            const uint32_t inner = static_cast<uint32_t>(tilingData_->dim1);
            const uint32_t packStride = AlignElements(inner / 8U);
            if ((inner & 7U) != 0 && packStride >= 64U) {
                ProcessARCompactDirect();
                return;
            }
        }
        ProcessARCompactGather();
    }

    __aicore__ inline void ProcessARCompactDirect()
    {
        const uint64_t normalSize = tilingData_->dim0;
        const uint32_t inner = static_cast<uint32_t>(tilingData_->dim1);
        const uint32_t packStride = AlignElements(inner / 8U);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(normalSize, coreStart, coreEnd);
        LocalTensor<float> rowPack = compactBuf_.Get<float>();
        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - rowBase));
        CopyInCompact(rowBase * inner, rows, inner);
        const uint32_t maxTileRows = tilingData_->tilePrimary;
        const uint32_t alignedRows = AlignElements(maxTileRows);
        LocalTensor<float> head = sumBuf_.Get<float>();
        LocalTensor<float> tail = sumBuf_.Get<float>()[alignedRows];
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                CopyInCompact(nextBase * inner, nextRows, inner);
            }
            SquareInput(inputLocal, rows);
            LocalTensor<float> source = GetSquaredTensor(inputLocal);
            inputQueue_.FreeTensor(inputLocal);
            BuildARRowPackDirect(rowPack, source, rows, inner, packStride);
            ReduceAREdges(head, tail, source, rows, inner);
            LocalTensor<float> result = ReduceARRowPack(
                rowPack, head, tail, rows, packStride);
            StoreOutput(rowBase, result, rows);
            if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                CopyInCompact(nextBase * inner, nextRows, inner);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void ProcessARCompactGather()
    {
        const uint64_t normalSize = tilingData_->dim0;
        const uint32_t inner = static_cast<uint32_t>(tilingData_->dim1);
        const uint32_t packStride = AlignElements(inner / 8U);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(normalSize, coreStart, coreEnd);
        if (coreStart >= coreEnd) {
            return;
        }
        LocalTensor<float> rowPack = compactBuf_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = sharedTmpBuf_.Get<uint32_t>();
        LocalTensor<float> blockSums = blockBuf_.Get<float>();
        const uint32_t maxTileElements = tilingData_->tilePrimary * inner;
        const uint32_t zeroBlock = (maxTileElements + 63U) / 64U * 8U;
        const uint32_t maxCoreRows = static_cast<uint32_t>(
            MinU64(tilingData_->tilePrimary, coreEnd - coreStart));
        InitARPackOffsets(gatherOffsets, inner, packStride, zeroBlock, maxCoreRows);
        Duplicate<float>(blockSums[zeroBlock], 0.0f, 8);
        PipeBarrier<PIPE_V>();

        uint64_t rowBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - rowBase));
        CopyInCompact(rowBase * inner, rows, inner);
        while (rowBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = rowBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                CopyInCompact(nextBase * inner, nextRows, inner);
            }
            SquareInput(inputLocal, rows);
            LocalTensor<float> source = GetSquaredTensor(inputLocal);
            inputQueue_.FreeTensor(inputLocal);
            blockSums = CompressARBlocks(source, rows, inner);
            BuildARRowPack(rowPack, blockSums, gatherOffsets, rows, packStride);

            const uint32_t alignedRows = AlignElements(rows);
            LocalTensor<float> head = sumBuf_.Get<float>();
            LocalTensor<float> tail = sumBuf_.Get<float>()[alignedRows];
            ReduceAREdges(head, tail, source, rows, inner);
            LocalTensor<float> result = ReduceARRowPack(
                rowPack, head, tail, rows, packStride);
            StoreOutput(rowBase, result, rows);
            if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                CopyInCompact(nextBase * inner, nextRows, inner);
            }
            rowBase = nextBase;
            rows = nextRows;
        }
    }

    __aicore__ inline void ProcessARA()
    {
        if (tilingData_->compactAra == 2) {
            ProcessARAPhase();
            return;
        }
        if (tilingData_->compactAra == 1) {
            ProcessARACompact();
            return;
        }
        const uint64_t outerSize = tilingData_->dim0;
        const uint64_t reduceSize = tilingData_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(outerSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(accumulator, 0.0f, tilingData_->alignedInner);
            PipeBarrier<PIPE_V>();
            uint64_t reduceBase = 0;
            uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, reduceSize));
            CopyIn(outer * reduceSize * innerSize, rows, innerSize, innerSize);
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? static_cast<uint32_t>(
                    MinU64(tilingData_->tilePrimary, reduceSize - nextBase)) : 0;
                if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                    CopyIn((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize, innerSize);
                }
                SquareInput(inputLocal, rows);
                LocalTensor<float> partial = ReduceRA(rows, GetSquaredTensor(inputLocal));
                inputQueue_.FreeTensor(inputLocal);
                Add<float>(accumulator, accumulator, partial, tilingData_->alignedInner);
                PipeBarrier<PIPE_V>();
                if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                    CopyIn((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            StoreOutput(outer * innerSize, accumulator, innerSize);
        }
    }

    __aicore__ inline void AccumulateCompactRA(uint32_t rows, uint32_t inner,
        LocalTensor<float> source, LocalTensor<float> accumulator,
        LocalTensor<uint32_t> gatherOffsets)
    {
        const uint32_t phaseRows = 8U / GcdU32(inner, 8U);
        const uint32_t groupElements = phaseRows * inner;
        const uint32_t fullGroups = rows / phaseRows;
        const uint32_t tailRows = rows % phaseRows;
        // 每组元素数均为 8 的倍数，可在对齐地址上原地二叉折叠为最多 8 条相位行。
        uint32_t activeGroups = fullGroups;
        while (activeGroups > 1) {
            const uint32_t leftGroups = (activeGroups + 1U) / 2U;
            const uint32_t pairGroups = activeGroups / 2U;
            Add<float>(source, source, source[leftGroups * groupElements],
                pairGroups * groupElements);
            PipeBarrier<PIPE_V>();
            activeGroups = leftGroups;
        }
        if (fullGroups > 0 && tailRows > 0) {
            Add<float>(source, source, source[fullGroups * groupElements], tailRows * inner);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> gatherLocal = sumBuf_.Get<float>();
        const uint32_t remainingRows = fullGroups > 0 ? phaseRows : tailRows;
        for (uint32_t row = 0; row < remainingRows; ++row) {
            for (uint32_t column = 0; column < inner; column += MAX_GATHER_ELEMENTS) {
                const uint32_t count = static_cast<uint32_t>(
                    MinU64(MAX_GATHER_ELEMENTS, inner - column));
                Gather<float>(gatherLocal[column], source, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Add<float>(accumulator[column], accumulator[column], gatherLocal[column], count);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessARACompact()
    {
        const uint64_t outerSize = tilingData_->dim0;
        const uint64_t reduceSize = tilingData_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(outerSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = sharedTmpBuf_.Get<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), innerSize);
        PipeBarrier<PIPE_V>();
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(accumulator, 0.0f, AlignElements(innerSize));
            PipeBarrier<PIPE_V>();
            uint64_t reduceBase = 0;
            uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, reduceSize));
            CopyInCompact(outer * reduceSize * innerSize, rows, innerSize);
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? static_cast<uint32_t>(
                    MinU64(tilingData_->tilePrimary, reduceSize - nextBase)) : 0;
                if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                    CopyInCompact((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                SquareInput(inputLocal, rows);
                AccumulateCompactRA(rows, innerSize, GetSquaredTensor(inputLocal),
                    accumulator, gatherOffsets);
                inputQueue_.FreeTensor(inputLocal);
                if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                    CopyInCompact((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            StoreOutput(outer * innerSize, accumulator, innerSize);
        }
    }

    __aicore__ inline void AccumulateARAPhases(LocalTensor<T> inputLocal, uint32_t rows,
        uint32_t inner, uint32_t phaseRows, uint32_t phaseElements,
        LocalTensor<float> phaseAccumulator, uint32_t& groupIndex)
    {
        LocalTensor<float> castLocal = workBuf_.Get<float>();
        const uint32_t fullPhaseRows = rows / phaseRows * phaseRows;
        uint32_t offset = 0;
        for (uint32_t row = 0; row < fullPhaseRows; row += phaseRows) {
            const uint32_t laneOffset = (groupIndex & 1U) != 0 ? phaseElements : 0U;
            LocalTensor<float> currentCast = castLocal[laneOffset];
            Cast<float, T>(currentCast, inputLocal[offset], RoundMode::CAST_NONE, phaseElements);
            MulAddDst<float, float>(phaseAccumulator[laneOffset],
                currentCast, currentCast, static_cast<int32_t>(phaseElements));
            ++groupIndex;
            offset += phaseElements;
        }
        if (rows > fullPhaseRows) {
            const uint32_t tailCount = (rows - fullPhaseRows) * inner;
            const uint32_t laneOffset = (groupIndex & 1U) != 0 ? phaseElements : 0U;
            LocalTensor<float> currentCast = castLocal[laneOffset];
            Cast<float, T>(currentCast, inputLocal[offset], RoundMode::CAST_NONE, tailCount);
            MulAddDst<float, float>(phaseAccumulator[laneOffset],
                currentCast, currentCast, static_cast<int32_t>(tailCount));
            ++groupIndex;
        }
    }

    __aicore__ inline void AccumulateCompactRAPhases(LocalTensor<float> inputLocal,
        uint32_t rows, uint32_t inner, uint32_t phaseRows, uint32_t phaseElements,
        LocalTensor<float> phaseAccumulator, uint32_t& groupIndex)
    {
        const uint32_t fullPhaseRows = rows / phaseRows * phaseRows;
        uint32_t offset = 0;
        for (uint32_t row = 0; row < fullPhaseRows; row += phaseRows) {
            const uint32_t lane = groupIndex % COMPACT_RA_PHASE_ACCUMULATORS;
            MulAddDst<float, float>(phaseAccumulator[lane * phaseElements],
                inputLocal[offset], inputLocal[offset],
                static_cast<int32_t>(phaseElements));
            ++groupIndex;
            offset += phaseElements;
        }
        if (rows > fullPhaseRows) {
            const uint32_t tailCount = (rows - fullPhaseRows) * inner;
            const uint32_t lane = groupIndex % COMPACT_RA_PHASE_ACCUMULATORS;
            MulAddDst<float, float>(phaseAccumulator[lane * phaseElements],
                inputLocal[offset], inputLocal[offset],
                static_cast<int32_t>(tailCount));
            ++groupIndex;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> FinalizeCompactRAPhases(uint32_t inner,
        uint32_t phaseRows, uint32_t phaseElements, LocalTensor<float> phaseAccumulator,
        LocalTensor<float> phaseWork)
    {
        const uint32_t alignedInner = AlignElements(inner);
        if constexpr (COMPACT_RA_PHASE_ACCUMULATORS > 1) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[phaseElements], phaseElements);
        }
        LocalTensor<uint32_t> gatherOffsets = phaseWork.ReinterpretCast<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), inner);
        LocalTensor<float> result = phaseWork[alignedInner];
        LocalTensor<float> gatherLocal = phaseWork[2U * alignedInner];
        Duplicate<float>(result, 0.0f, alignedInner);
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < inner; column += MAX_GATHER_ELEMENTS) {
            const uint32_t count = static_cast<uint32_t>(
                MinU64(MAX_GATHER_ELEMENTS, inner - column));
            Gather<float>(result[column], phaseAccumulator, gatherOffsets[column], 0, count);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < phaseRows; ++row) {
            for (uint32_t column = 0; column < inner; column += MAX_GATHER_ELEMENTS) {
                const uint32_t count = static_cast<uint32_t>(
                    MinU64(MAX_GATHER_ELEMENTS, inner - column));
                Gather<float>(gatherLocal[column], phaseAccumulator, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                PipeBarrier<PIPE_V>();
                Add<float>(result[column], result[column], gatherLocal[column], count);
                PipeBarrier<PIPE_V>();
            }
        }
        return result;
    }

    __aicore__ inline LocalTensor<float> FinalizeARAPhases(uint32_t inner,
        uint32_t phaseRows, uint32_t phaseElements, LocalTensor<float> phaseAccumulator)
    {
        if constexpr (COMPACT_ARA_PHASE_ACCUMULATORS > 1) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[phaseElements], phaseElements);
        }
        const uint32_t remainingRows = 8U / GcdU32(inner, 8U);
        if (phaseRows > remainingRows) {
            Add<float>(phaseAccumulator, phaseAccumulator,
                phaseAccumulator[remainingRows * inner], remainingRows * inner);
        }

        const uint32_t alignedInner = AlignElements(inner);
        LocalTensor<float> phaseWork = workBuf_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = phaseWork.ReinterpretCast<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), inner);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> gatherLocal = phaseWork[alignedInner];
        for (uint32_t row = 1; row < remainingRows; ++row) {
            for (uint32_t column = 0; column < inner; column += MAX_GATHER_ELEMENTS) {
                const uint32_t count = static_cast<uint32_t>(
                    MinU64(MAX_GATHER_ELEMENTS, inner - column));
                Gather<float>(gatherLocal[column], phaseAccumulator, gatherOffsets[column],
                    row * inner * sizeof(float), count);
                Add<float>(phaseAccumulator[column], phaseAccumulator[column], gatherLocal[column], count);
            }
        }
        return phaseAccumulator;
    }

    __aicore__ inline uint32_t SelectARAPhaseRows(uint64_t outer, uint64_t reduceBase,
        uint64_t reduceSize, uint32_t inner, uint32_t phaseRows) const
    {
        const uint64_t remaining = reduceSize - reduceBase;
        if (reduceBase == 0) {
            const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(T);
            const uint32_t outerPhase = static_cast<uint32_t>(
                (outer * reduceSize * inner) % blockElements);
            if (outerPhase != 0) {
                for (uint32_t leadingRows = 1; leadingRows < phaseRows; ++leadingRows) {
                    if ((outerPhase + leadingRows * inner) % blockElements == 0) {
                        return static_cast<uint32_t>(MinU64(leadingRows, remaining));
                    }
                }
            }
        }
        return static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, remaining));
    }

    __aicore__ inline void ProcessARAPhase()
    {
        const uint64_t outerSize = tilingData_->dim0;
        const uint64_t reduceSize = tilingData_->dim1;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim2);
        const uint32_t phaseRows = 16U / GcdU32(innerSize, 16U);
        const uint32_t phaseElements = phaseRows * innerSize;
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(outerSize, coreStart, coreEnd);
        LocalTensor<float> phaseAccumulator = accBuf_.Get<float>();
        bool inputPrefetched = false;
        uint32_t prefetchedRows = 0;
        if (coreStart < coreEnd) {
            prefetchedRows = SelectARAPhaseRows(
                coreStart, 0, reduceSize, innerSize, phaseRows);
            CopyInCompact(coreStart * reduceSize * innerSize, prefetchedRows, innerSize);
            inputPrefetched = true;
        }
        for (uint64_t outer = coreStart; outer < coreEnd; ++outer) {
            Duplicate<float>(phaseAccumulator, 0.0f,
                COMPACT_ARA_PHASE_ACCUMULATORS * phaseElements);
            uint32_t groupIndex = 0;
            uint64_t reduceBase = 0;
            uint32_t rows = 0;
            if (inputPrefetched) {
                rows = prefetchedRows;
                inputPrefetched = false;
            } else {
                rows = SelectARAPhaseRows(
                    outer, reduceBase, reduceSize, innerSize, phaseRows);
                CopyInCompact(outer * reduceSize * innerSize, rows, innerSize);
            }
            while (reduceBase < reduceSize) {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < reduceSize ? SelectARAPhaseRows(
                    outer, nextBase, reduceSize, innerSize, phaseRows) : 0;
                if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                    CopyInCompact((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                } else if (nextRows == 0 && tilingData_->inputBufferNum == 2 &&
                    outer + 1U < coreEnd) {
                    prefetchedRows = SelectARAPhaseRows(
                        outer + 1U, 0, reduceSize, innerSize, phaseRows);
                    CopyInCompact((outer + 1U) * reduceSize * innerSize,
                        prefetchedRows, innerSize);
                    inputPrefetched = true;
                }
                AccumulateARAPhases(inputLocal, rows, innerSize, phaseRows,
                    phaseElements, phaseAccumulator, groupIndex);
                inputQueue_.FreeTensor(inputLocal);
                if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                    CopyInCompact((outer * reduceSize + nextBase) * innerSize,
                        nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<float> result = FinalizeARAPhases(
                innerSize, phaseRows, phaseElements, phaseAccumulator);
            StoreOutput(outer * innerSize, result, innerSize);
        }
    }

    __aicore__ inline void ProcessRAR()
    {
        if constexpr (std::is_same<T, half>::value) {
            if (tilingData_->compactRar != 0) {
                ProcessRARCompact();
                return;
            }
        }
        const uint64_t outerReduce = tilingData_->dim0;
        const uint64_t normalSize = tilingData_->dim1;
        const uint32_t innerReduce = static_cast<uint32_t>(tilingData_->dim2);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(normalSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        for (uint64_t normalBase = coreStart; normalBase < coreEnd; normalBase += tilingData_->tilePrimary) {
            const uint32_t rows = static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - normalBase));
            Duplicate<float>(accumulator, 0.0f, AlignElements(rows));
            PipeBarrier<PIPE_V>();
            uint64_t outer = 0;
            CopyIn(normalBase * innerReduce, rows, innerReduce, innerReduce);
            while (outer < outerReduce) {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                const uint64_t nextOuter = outer + 1;
                if (nextOuter < outerReduce && tilingData_->inputBufferNum == 2) {
                    CopyIn((nextOuter * normalSize + normalBase) * innerReduce,
                        rows, innerReduce, innerReduce);
                }
                SquareInput(inputLocal, rows);
                LocalTensor<float> partial = ReduceAR(rows, GetSquaredTensor(inputLocal));
                inputQueue_.FreeTensor(inputLocal);
                Add<float>(accumulator, accumulator, partial, rows);
                PipeBarrier<PIPE_V>();
                if (nextOuter < outerReduce && tilingData_->inputBufferNum == 1) {
                    CopyIn((nextOuter * normalSize + normalBase) * innerReduce,
                        rows, innerReduce, innerReduce);
                }
                outer = nextOuter;
            }
            StoreOutput(normalBase, accumulator, rows);
        }
    }

    __aicore__ inline void ProcessRARCompactTile(uint64_t normalBase, uint32_t rows,
        uint64_t outerReduce, uint64_t normalSize, uint32_t innerReduce,
        uint64_t outerStart, LocalTensor<float> matrixAccumulator,
        bool isFirstTile = true, bool hasNextTile = false,
        uint64_t nextNormalBase = 0, uint32_t nextTileRows = 0)
    {
        const uint32_t matrixElements = rows * innerReduce;
        const uint64_t outerStride = normalSize * innerReduce;
        const uint64_t tileBaseOffset = normalBase * innerReduce;
        const uint64_t nextTileFirstGmOffset = hasNextTile ?
            (outerStart * outerStride + nextNormalBase * innerReduce) : 0U;
        uint64_t outer = outerStart;
        uint64_t gmOffset = outerStart * outerStride + tileBaseOffset;
        if (isFirstTile) {
            CopyInCompact(gmOffset, rows, innerReduce);
        }
        Duplicate<float>(matrixAccumulator, 0.0f, matrixElements);
        do {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            uint64_t nextOuter = outer + 1U;
            uint64_t nextGmOffset = gmOffset + outerStride;
            if (nextOuter == outerReduce) {
                nextOuter = 0;
                nextGmOffset = tileBaseOffset;
            }
            const bool hasNext = nextOuter != outerStart;
            if (hasNext && tilingData_->inputBufferNum == 2) {
                CopyInCompact(nextGmOffset, rows, innerReduce);
            } else if (!hasNext && tilingData_->inputBufferNum == 2 && hasNextTile) {
                CopyInCompact(nextTileFirstGmOffset, nextTileRows, innerReduce);
            }
            MulAddDst<float, T>(
                matrixAccumulator, inputLocal, inputLocal, static_cast<int32_t>(matrixElements));
            inputQueue_.FreeTensor(inputLocal);
            if (hasNext && tilingData_->inputBufferNum == 1) {
                CopyInCompact(nextGmOffset, rows, innerReduce);
            }
            outer = nextOuter;
            gmOffset = nextGmOffset;
        } while (outer != outerStart);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> partial = ReduceCompactAR(rows, innerReduce, matrixAccumulator);
        StoreOutput(normalBase, partial, rows);
    }

    __aicore__ inline void ProcessRARCompact()
    {
        const uint64_t outerReduce = tilingData_->dim0;
        const uint64_t normalSize = tilingData_->dim1;
        const uint32_t innerReduce = static_cast<uint32_t>(tilingData_->dim2);
        LocalTensor<float> matrixAccumulator = workBuf_.Get<float>();
        if (tilingData_->rarScheduleMode == RAR_SCHEDULE_WAVE) {
            const uint64_t blockDim = tilingData_->blockDim;
            const uint64_t physicalBlockIdx = GetBlockIdx();
            uint64_t blockIdx = physicalBlockIdx;
            if ((blockDim & 1U) == 0 && blockDim >= 4U) {
                const uint64_t pairCount = blockDim / 2U;
                const uint64_t pairRank = physicalBlockIdx / 2U;
                const uint64_t pairLowerHalf = (pairCount + 1U) / 2U;
                const uint64_t logicalPairRank = (pairRank & 1U) == 0 ?
                    pairRank / 2U : pairLowerHalf + pairRank / 2U;
                blockIdx = (physicalBlockIdx & 1U) == 0 ?
                    logicalPairRank : pairCount + logicalPairRank;
            }
            const uint64_t secondGroupBegin =
                (static_cast<uint64_t>(tilingData_->blockDim) + 1U) / 2U;
            // 两组 block 从相隔半个 outer 轴的位置起步，分散大工作集同时访问的 GM slab。
            // 每组仍循环完整 outerReduce 次，保持总搬运字节、DMA 次数与 FP32 累加深度不变。
            const uint64_t outerStart =
                blockIdx >= secondGroupBegin && outerReduce > 1U ? outerReduce / 2U : 0U;
            const uint64_t tileRows = tilingData_->tilePrimary;
            const uint64_t waveRows = static_cast<uint64_t>(tilingData_->blockDim) * tileRows;
            const uint64_t fullWaves = normalSize / waveRows;
            const uint64_t prefixRows = fullWaves * waveRows;
            const uint64_t tailRows = normalSize - prefixRows;
            uint64_t tailStart = 0;
            uint64_t tailEnd = 0;
            const uint32_t quantum = UB_BLOCK_BYTES / sizeof(T);
            SplitRangeAlignedIdx(tailRows, quantum, blockIdx, blockDim, tailStart, tailEnd);
            const bool hasTail = tailStart < tailEnd;
            const uint32_t tailTileRows = hasTail ? static_cast<uint32_t>(tailEnd - tailStart) : 0U;
            const uint64_t tailNormalBase = prefixRows + tailStart;

            uint64_t normalBase = blockIdx * tileRows;
            bool isFirst = true;
            for (uint64_t wave = 0; wave < fullWaves; ++wave) {
                const bool hasNextWave = wave + 1U < fullWaves;
                const bool hasNext = hasNextWave || hasTail;
                const uint64_t nextBase = hasNextWave ? (normalBase + waveRows) : tailNormalBase;
                const uint32_t nextRows = hasNextWave ? static_cast<uint32_t>(tileRows) : tailTileRows;
                ProcessRARCompactTile(normalBase, static_cast<uint32_t>(tileRows),
                    outerReduce, normalSize, innerReduce, outerStart, matrixAccumulator,
                    isFirst, hasNext, nextBase, nextRows);
                normalBase += waveRows;
                isFirst = false;
            }
            if (hasTail) {
                ProcessRARCompactTile(tailNormalBase, tailTileRows, outerReduce,
                    normalSize, innerReduce, outerStart, matrixAccumulator,
                    isFirst, false, 0, 0);
            }
            return;
        }

        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(normalSize, coreStart, coreEnd);
        bool isFirst = true;
        for (uint64_t normalBase = coreStart; normalBase < coreEnd; normalBase += tilingData_->tilePrimary) {
            const uint32_t rows = static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - normalBase));
            const uint64_t nextBase = normalBase + tilingData_->tilePrimary;
            const bool hasNext = nextBase < coreEnd;
            const uint32_t nextRows = hasNext ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0U;
            ProcessRARCompactTile(normalBase, rows, outerReduce,
                normalSize, innerReduce, 0, matrixAccumulator,
                isFirst, hasNext, nextBase, nextRows);
            isFirst = false;
        }
    }

    __aicore__ inline void ProcessRACompactTiled()
    {
        const uint64_t reduceSize = tilingData_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim1);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(reduceSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        LocalTensor<uint32_t> gatherOffsets = sharedTmpBuf_.Get<uint32_t>();
        Arange<int32_t>(gatherOffsets.ReinterpretCast<int32_t>(), 0,
            static_cast<int32_t>(sizeof(float)), innerSize);
        Duplicate<float>(accumulator, 0.0f, tilingData_->alignedInner);
        PipeBarrier<PIPE_V>();

        uint64_t reduceBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - reduceBase));
        const uint32_t staggerGroups = tilingData_->raStaggerGroups;
        if (staggerGroups > 1 && rows == tilingData_->tilePrimary) {
            const uint64_t coreRows = coreEnd - coreStart;
            const uint64_t rounds = (coreRows + tilingData_->tilePrimary - 1) /
                tilingData_->tilePrimary;
            const uint32_t tailRows = static_cast<uint32_t>(
                coreRows - (rounds - 1) * tilingData_->tilePrimary);
            const uint32_t shiftBudget = tilingData_->tilePrimary - tailRows;
            const uint32_t group = GetBlockIdx() % staggerGroups;
            const uint32_t firstShift = shiftBudget * group / (staggerGroups - 1);
            rows = tilingData_->tilePrimary - firstShift;
        }
        CopyInCompact(reduceBase * innerSize, rows, innerSize);
        while (reduceBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = reduceBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            SquareInputCompact(inputLocal, rows * innerSize);
            AccumulateCompactRA(rows, innerSize, GetSquaredTensor(inputLocal),
                accumulator, gatherOffsets);
            inputQueue_.FreeTensor(inputLocal);
            if (nextRows > 0) {
                CopyInCompact(nextBase * innerSize, nextRows, innerSize);
            }
            reduceBase = nextBase;
            rows = nextRows;
        }

        const uint64_t workspaceStrideElements = tilingData_->workspaceStride / sizeof(float);
        SyncVToMte3();
        DataCopy(workspaceGm_[static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements],
            accumulator, tilingData_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<T> mergeInput = inputQueue_.AllocTensor<T>();
        inputQueue_.EnQue(mergeInput);
        mergeInput = inputQueue_.DeQue<T>();
        MergeWorkspace(accumulator, mergeInput, workspaceStrideElements);
        inputQueue_.FreeTensor(mergeInput);
        SyncVToMte3();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(innerSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm_, accumulator, copyParams);
    }

    __aicore__ inline void ProcessRACompactPhase()
    {
        const uint64_t reduceSize = tilingData_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim1);
        const uint32_t phaseRows = tilingData_->compactRaPhaseRows;
        const uint32_t phaseElements = phaseRows * innerSize;
        const bool segmented =
            tilingData_->compactRaPhaseSegments == COMPACT_RA_PHASE_MAX_SEGMENTS;
        const uint32_t checkpointGroups = tilingData_->compactRaPhaseCheckpointGroups;
        const uint64_t workspaceStrideElements = tilingData_->workspaceStride / sizeof(float);
        const uint64_t workspaceOffset =
            static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements;
        LocalTensor<float> phaseAccumulator = accBuf_.Get<float>();
        Duplicate<float>(phaseAccumulator, 0.0f,
            COMPACT_RA_PHASE_ACCUMULATORS * phaseElements);
        PipeBarrier<PIPE_V>();

        uint32_t groupIndex = 0;
        bool checkpointStored = false;
        if (tilingData_->compactRaPhaseScheduleMode == COMPACT_RA_PHASE_SCHEDULE_WAVE) {
            const uint64_t tileRows = tilingData_->tilePrimary;
            const uint64_t fullTiles = reduceSize / tileRows;
            const uint64_t tailRows = reduceSize % tileRows;
            const uint64_t physicalBlockIdx = GetBlockIdx();
            const bool ownsTail = tailRows != 0 &&
                physicalBlockIdx == fullTiles % tilingData_->blockDim;
            uint64_t tileIndex = physicalBlockIdx;
            uint64_t reduceBase = tileIndex * tileRows;
            uint32_t rows = static_cast<uint32_t>(tileRows);
            bool currentIsTail = false;
            bool tailIssued = false;
            CopyInCompact(reduceBase * innerSize, rows, innerSize);
            do {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                uint64_t nextBase = 0;
                uint64_t nextTileIndex = tileIndex;
                uint32_t nextRows = 0;
                bool nextIsTail = false;
                if (!currentIsTail) {
                    nextTileIndex = tileIndex + tilingData_->blockDim;
                    if (nextTileIndex < fullTiles) {
                        nextBase = nextTileIndex * tileRows;
                        nextRows = static_cast<uint32_t>(tileRows);
                    } else if (ownsTail && !tailIssued) {
                        // 不完整 wave 后的尾 tile 交给下一物理核，维持该波的连续 GM 访问。
                        nextBase = fullTiles * tileRows;
                        nextRows = static_cast<uint32_t>(tailRows);
                        nextIsTail = true;
                        tailIssued = true;
                    }
                }
                if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                    CopyInCompact(nextBase * innerSize, nextRows, innerSize);
                }
                AccumulateCompactRAPhases(inputLocal, rows, innerSize,
                    phaseRows, phaseElements, phaseAccumulator, groupIndex);
                inputQueue_.FreeTensor(inputLocal);
                if (segmented && !checkpointStored && groupIndex == checkpointGroups) {
                    LocalTensor<float> phaseWork = workBuf_.Get<float>();
                    LocalTensor<float> checkpointResult = FinalizeCompactRAPhases(
                        innerSize, phaseRows, phaseElements, phaseAccumulator, phaseWork);
                    LocalTensor<float> checkpoint = checkpointBuf_.Get<float>();
                    Adds<float>(checkpoint, checkpointResult, 0.0f, tilingData_->alignedInner);
                    Duplicate<float>(phaseAccumulator, 0.0f,
                        COMPACT_RA_PHASE_ACCUMULATORS * phaseElements);
                    PipeBarrier<PIPE_V>();
                    groupIndex = 0;
                    checkpointStored = true;
                }
                if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                    CopyInCompact(nextBase * innerSize, nextRows, innerSize);
                }
                // 统一在循环尾部更新下一状态，避免提前跳出遗漏同步生命周期。
                reduceBase = nextBase;
                tileIndex = nextTileIndex;
                rows = nextRows;
                currentIsTail = nextIsTail;
            } while (rows > 0);
        } else {
            uint64_t coreStart = 0;
            uint64_t coreEnd = 0;
            if (tilingData_->compactRaPhaseAlignedSplit == 0 ||
                !SplitRangePhaseGroups(reduceSize, phaseRows, coreStart, coreEnd)) {
                SplitRange(reduceSize, coreStart, coreEnd);
            }
            uint64_t reduceBase = coreStart;
            uint32_t rows = static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - reduceBase));
            CopyInCompact(reduceBase * innerSize, rows, innerSize);
            while (reduceBase < coreEnd) {
                LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
                const uint64_t nextBase = reduceBase + rows;
                const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                    MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
                if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                    CopyInCompact(nextBase * innerSize, nextRows, innerSize);
                }
                AccumulateCompactRAPhases(inputLocal, rows, innerSize,
                    phaseRows, phaseElements, phaseAccumulator, groupIndex);
                inputQueue_.FreeTensor(inputLocal);
                if (segmented && !checkpointStored && groupIndex == checkpointGroups) {
                    LocalTensor<float> phaseWork = workBuf_.Get<float>();
                    LocalTensor<float> checkpointResult = FinalizeCompactRAPhases(
                        innerSize, phaseRows, phaseElements, phaseAccumulator, phaseWork);
                    LocalTensor<float> checkpoint = checkpointBuf_.Get<float>();
                    Adds<float>(checkpoint, checkpointResult, 0.0f, tilingData_->alignedInner);
                    Duplicate<float>(phaseAccumulator, 0.0f,
                        COMPACT_RA_PHASE_ACCUMULATORS * phaseElements);
                    PipeBarrier<PIPE_V>();
                    groupIndex = 0;
                    checkpointStored = true;
                }
                if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                    CopyInCompact(nextBase * innerSize, nextRows, innerSize);
                }
                reduceBase = nextBase;
                rows = nextRows;
            }
        }

        if (segmented) {
            LocalTensor<float> phaseWork = workBuf_.Get<float>();
            LocalTensor<float> result = FinalizeCompactRAPhases(innerSize, phaseRows,
                phaseElements, phaseAccumulator, phaseWork);
            if (checkpointStored) {
                Add<float>(phaseAccumulator, result, checkpointBuf_.Get<float>(),
                    tilingData_->alignedInner);
            } else {
                Adds<float>(phaseAccumulator, result, 0.0f, tilingData_->alignedInner);
            }
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<T> phaseWorkTensor = inputQueue_.AllocTensor<T>();
            inputQueue_.EnQue(phaseWorkTensor);
            phaseWorkTensor = inputQueue_.DeQue<T>();
            LocalTensor<float> result = FinalizeCompactRAPhases(innerSize, phaseRows,
                phaseElements, phaseAccumulator, phaseWorkTensor);
            Adds<float>(phaseAccumulator, result, 0.0f, tilingData_->alignedInner);
            PipeBarrier<PIPE_V>();
            inputQueue_.FreeTensor(phaseWorkTensor);
        }

        // phaseAccumulator 由 Vector 生成，显式建立 V 到 MTE3 的写回依赖。
        const event_t workspaceStoreEvent = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(workspaceStoreEvent);
        WaitFlag<HardEvent::V_MTE3>(workspaceStoreEvent);
        DataCopy(workspaceGm_[workspaceOffset], phaseAccumulator, tilingData_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        // 主输入已全部消费且显式完成 MTE2->V 同步，直接取得空闲槽作为归并缓冲，
        // 避免无数据的 EnQue/DeQue 事件干扰收尾流水。
        LocalTensor<T> mergeInput = inputQueue_.AllocTensor<T>();
        MergeWorkspace(phaseAccumulator, mergeInput, workspaceStrideElements);
        inputQueue_.FreeTensor(mergeInput);
        SyncVToMte3();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(innerSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm_, phaseAccumulator, copyParams);
    }

    __aicore__ inline void ProcessRA()
    {
        if constexpr (std::is_same<T, float>::value) {
            if (tilingData_->compactRa == COMPACT_RA_PHASE) {
                ProcessRACompactPhase();
                return;
            }
            if (tilingData_->compactRa == COMPACT_RA_TILED) {
                ProcessRACompactTiled();
                return;
            }
        }
        const uint64_t reduceSize = tilingData_->dim0;
        const uint32_t innerSize = static_cast<uint32_t>(tilingData_->dim1);
        uint64_t coreStart = 0;
        uint64_t coreEnd = 0;
        SplitRange(reduceSize, coreStart, coreEnd);
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        Duplicate<float>(accumulator, 0.0f, tilingData_->alignedInner);
        PipeBarrier<PIPE_V>();
        uint64_t reduceBase = coreStart;
        uint32_t rows = static_cast<uint32_t>(MinU64(tilingData_->tilePrimary, coreEnd - reduceBase));
        const uint32_t staggerGroups = tilingData_->raStaggerGroups;
        if (staggerGroups > 1 && rows == tilingData_->tilePrimary) {
            const uint64_t coreRows = coreEnd - coreStart;
            const uint64_t rounds = (coreRows + tilingData_->tilePrimary - 1) /
                tilingData_->tilePrimary;
            const uint32_t tailRows = static_cast<uint32_t>(
                coreRows - (rounds - 1) * tilingData_->tilePrimary);
            const uint32_t shiftBudget = tilingData_->tilePrimary - tailRows;
            const uint32_t group = GetBlockIdx() % staggerGroups;
            const uint32_t firstShift = shiftBudget * group / (staggerGroups - 1);
            // 只调整首尾块边界，保持每核总轮数、总字节数和连续处理区间不变。
            rows = tilingData_->tilePrimary - firstShift;
        }
        CopyIn(reduceBase * innerSize, rows, innerSize, innerSize);
        while (reduceBase < coreEnd) {
            LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
            const uint64_t nextBase = reduceBase + rows;
            const uint32_t nextRows = nextBase < coreEnd ? static_cast<uint32_t>(
                MinU64(tilingData_->tilePrimary, coreEnd - nextBase)) : 0;
            if (nextRows > 0 && tilingData_->inputBufferNum == 2) {
                CopyIn(nextBase * innerSize, nextRows, innerSize, innerSize);
            }
            SquareInput(inputLocal, rows);
            LocalTensor<float> partial = ReduceRA(rows, GetSquaredTensor(inputLocal));
            inputQueue_.FreeTensor(inputLocal);
            Add<float>(accumulator, accumulator, partial, tilingData_->alignedInner);
            PipeBarrier<PIPE_V>();
            if (nextRows > 0 && tilingData_->inputBufferNum == 1) {
                CopyIn(nextBase * innerSize, nextRows, innerSize, innerSize);
            }
            reduceBase = nextBase;
            rows = nextRows;
        }

        const uint64_t workspaceStrideElements = tilingData_->workspaceStride / sizeof(float);
        SyncVToMte3();
        DataCopy(workspaceGm_[static_cast<uint64_t>(GetBlockIdx()) * workspaceStrideElements],
            accumulator, tilingData_->alignedInner);
        SyncAll();

        if (GetBlockIdx() != 0) {
            return;
        }
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<T> mergeInput = inputQueue_.AllocTensor<T>();
            inputQueue_.EnQue(mergeInput);
            mergeInput = inputQueue_.DeQue<T>();
            MergeWorkspace(accumulator, mergeInput, workspaceStrideElements);
            inputQueue_.FreeTensor(mergeInput);
        } else {
            MergeWorkspace(accumulator, workBuf_.Get<float>(), workspaceStrideElements);
        }
        StoreOutput(0, accumulator, innerSize);
    }

    __aicore__ inline void MergeWorkspace(LocalTensor<float> accumulator,
        LocalTensor<float> mergeLocal, uint64_t workspaceStrideElements)
    {
        const uint32_t rowBytes = tilingData_->alignedInner * sizeof(float);
        const uint32_t srcStride = static_cast<uint32_t>(tilingData_->workspaceStride - rowBytes);
        uint32_t mergeRows = tilingData_->tilePrimary;
        if (tilingData_->compactRa != 0) {
            const uint64_t compactElements = static_cast<uint64_t>(tilingData_->tilePrimary) *
                tilingData_->dim1;
            mergeRows = static_cast<uint32_t>(compactElements / tilingData_->alignedInner);
            if (mergeRows == 0) {
                mergeRows = 1;
            }
        }
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        // 批量搬入多个核的 partial，减少逐核 DMA 和归约启动开销。
        for (uint32_t coreBase = 1; coreBase < tilingData_->blockDim;
             coreBase += mergeRows) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(
                mergeRows, tilingData_->blockDim - coreBase));
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), rowBytes, srcStride, 0, 0};
            DataCopyPad(mergeLocal,
                workspaceGm_[static_cast<uint64_t>(coreBase) * workspaceStrideElements],
                copyParams, padParams);
            SyncMte2ToV();
            LocalTensor<float> partial = tilingData_->compactRa != 0 ?
                ReduceRAInPlace(rows, mergeLocal) : ReduceRA(rows, mergeLocal);
            Add<float>(accumulator, accumulator, partial, tilingData_->alignedInner);
            SyncVToMte2();
        }
    }

    __aicore__ inline uint64_t DecodeOutputBase(uint64_t linear, bool skipLastNormal) const
    {
        uint64_t base = 0;
        for (int32_t dim = static_cast<int32_t>(tilingData_->rank) - 1; dim >= 0; --dim) {
            const bool isReduce = (tilingData_->axisMask & (1U << dim)) != 0;
            if (isReduce || (skipLastNormal && dim == static_cast<int32_t>(tilingData_->rank) - 1)) {
                continue;
            }
            const uint64_t coordinate = linear % tilingData_->shape[dim];
            linear /= tilingData_->shape[dim];
            base += coordinate * tilingData_->stride[dim];
        }
        return base;
    }

    __aicore__ inline uint64_t DecodeReduceOffset(uint64_t linear, bool skipLastReduce) const
    {
        uint64_t offset = 0;
        for (int32_t dim = static_cast<int32_t>(tilingData_->rank) - 1; dim >= 0; --dim) {
            const bool isReduce = (tilingData_->axisMask & (1U << dim)) != 0;
            if (!isReduce || (skipLastReduce && dim == static_cast<int32_t>(tilingData_->rank) - 1)) {
                continue;
            }
            const uint64_t coordinate = linear % tilingData_->shape[dim];
            linear /= tilingData_->shape[dim];
            offset += coordinate * tilingData_->stride[dim];
        }
        return offset;
    }

    __aicore__ inline void ProcessGeneric()
    {
        const uint32_t lastDim = tilingData_->rank - 1;
        const uint32_t lastSize = static_cast<uint32_t>(tilingData_->shape[lastDim]);
        const bool lastReduced = (tilingData_->axisMask & (1U << lastDim)) != 0;
        LocalTensor<float> accumulator = accBuf_.Get<float>();
        if (lastReduced) {
            const uint64_t outerReduce = tilingData_->reduceSize / lastSize;
            for (uint64_t output = 0; output < tilingData_->outputSize; ++output) {
                Duplicate<float>(accumulator, 0.0f, 8);
                PipeBarrier<PIPE_V>();
                const uint64_t outputBase = DecodeOutputBase(output, false);
                for (uint64_t reduce = 0; reduce < outerReduce; ++reduce) {
                    const uint64_t gmOffset = outputBase + DecodeReduceOffset(reduce, true);
                    LocalTensor<T> inputLocal = CopyAndSquare(gmOffset, 1, lastSize, lastSize);
                    LocalTensor<float> partial = ReduceAR(1, GetSquaredTensor(inputLocal));
                    inputQueue_.FreeTensor(inputLocal);
                    Add<float>(accumulator, accumulator, partial, 1);
                    PipeBarrier<PIPE_V>();
                }
                StoreOutput(output, accumulator, 1);
            }
            return;
        }

        const uint64_t outputGroups = tilingData_->outputSize / lastSize;
        for (uint64_t group = 0; group < outputGroups; ++group) {
            Duplicate<float>(accumulator, 0.0f, tilingData_->alignedInner);
            PipeBarrier<PIPE_V>();
            const uint64_t outputBase = DecodeOutputBase(group, true);
            for (uint64_t reduce = 0; reduce < tilingData_->reduceSize; ++reduce) {
                const uint64_t gmOffset = outputBase + DecodeReduceOffset(reduce, false);
                LocalTensor<T> inputLocal = CopyAndSquare(gmOffset, 1, lastSize, lastSize);
                Add<float>(accumulator, accumulator, GetSquaredTensor(inputLocal), tilingData_->alignedInner);
                PipeBarrier<PIPE_V>();
                inputQueue_.FreeTensor(inputLocal);
            }
            StoreOutput(group * lastSize, accumulator, lastSize);
        }
    }

private:
    const SquareSumV1TilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<float> workspaceGm_;
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> checkpointBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> sharedTmpBuf_;
    TBuf<TPosition::VECCALC> blockBuf_;
    TBuf<TPosition::VECCALC> compactBuf_;
};
}

template <uint32_t schMode>
__global__ __aicore__ void square_sum_v1(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 模板函数内的 if constexpr 不会被 op 编译器当作任务元数据分支解析；
    // 将需要跨核同步的 key 在入口无条件登记，其他 key 仍保持 AIV-only。
    KERNEL_TASK_TYPE(SQUARE_SUM_KEY_SYNC, KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (schMode == SQUARE_SUM_KEY_SYNC) {
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        AscendC::TPipe pipe;
        SquareSum::KernelSquareSum<DTYPE_INPUT> op;
        op.Init(input, output, userWorkspace, &tilingData, &pipe);
        op.ProcessSync();
        return;
    }
#if defined(ORIG_DTYPE_INPUT) && defined(DT_BF16) && (ORIG_DTYPE_INPUT == DT_BF16)
    if constexpr (schMode == SQUARE_SUM_KEY_COMPACT_AR_DIRECT) {
        KERNEL_TASK_TYPE(SQUARE_SUM_KEY_COMPACT_AR_DIRECT, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        SquareSum::KernelSquareSum<DTYPE_INPUT> op;
        op.Init(input, output, nullptr, &tilingData, &pipe);
        op.ProcessCompactArDirect();
        return;
    }
#endif
    if constexpr (schMode == SQUARE_SUM_KEY_NORMAL) {
        KERNEL_TASK_TYPE(SQUARE_SUM_KEY_NORMAL, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        SquareSum::KernelSquareSum<DTYPE_INPUT> op;
        op.Init(input, output, nullptr, &tilingData, &pipe);
        op.Process();
    }
}
