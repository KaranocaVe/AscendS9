#include "kernel_operator.h"
#include "adv_api/pad/broadcast.h"

#define GREATER_KEY_V2 6
#define GREATER_KEY_V2_WHOLE_LINEAR 7

using namespace AscendC;

namespace GreaterKernel {

constexpr uint32_t MAX_RANK = 25;
constexpr uint32_t ROW_PARTITION_GROUP_MAJOR = 0;
constexpr uint32_t ROW_PARTITION_ROW_MAJOR = 1;
constexpr uint32_t SCHEDULE_LINEAR_SPAN = 0;
constexpr uint32_t SCHEDULE_RESIDENT_ROW = 1;
constexpr uint32_t SCHEDULE_AFFINE_ROW = 2;
constexpr uint32_t SCHEDULE_RESIDENT_PACKED = 3;
constexpr uint32_t SCHEDULE_RESIDENT_PITCHED = 4;
constexpr uint32_t BROADCAST_SELF = 1;
constexpr uint32_t BROADCAST_OTHER = 2;
constexpr uint32_t SELECT_RESERVED_BYTES = 8 * 1024;
constexpr uint32_t BUFFER_PHASE_BYTES = 256;
constexpr uint32_t UB_DATA_BLOCK_BYTES = 32;
constexpr uint32_t UB_BANK_GROUP_COUNT = 16;
constexpr uint32_t LINEAR_BANK_SKEW_BYTES = 4 * UB_DATA_BLOCK_BYTES;
constexpr uint32_t INT32_STRIPE_ELEMENTS = BUFFER_PHASE_BYTES / sizeof(int32_t);
constexpr uint32_t MAX_VECTOR_REPEAT = 255;
constexpr uint32_t MAX_DMA_BLOCK_COUNT = 4095;
constexpr uint32_t INT32_COMPARE_ALIGN = 64;

template <typename T>
__aicore__ inline uint32_t ComputeAlign()
{
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int8_t>) {
        return 128;
    }
    return 64;
}

__aicore__ inline uint32_t MinU32(uint64_t lhs, uint32_t rhs)
{
    return static_cast<uint32_t>(lhs < rhs ? lhs : rhs);
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline uint32_t GreatestCommonDivisor(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

template <typename T>
__aicore__ inline void CopyInPad(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyOutPad(GlobalTensor<int8_t> dst, LocalTensor<int8_t> src,
    uint32_t count)
{
    DataCopyExtParams copyParams{1, count, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

template <typename T>
__aicore__ inline void ComputePacked(LocalTensor<int8_t> output, LocalTensor<T> self,
    LocalTensor<T> other, TBuf<TPosition::VECCALC>& work0,
    TBuf<TPosition::VECCALC>& work1, uint32_t computeCount,
    bool selfScalar = false, bool otherScalar = false)
{
    LocalTensor<uint8_t> mask = output.template ReinterpretCast<uint8_t>();
    if constexpr (std::is_same_v<T, half>) {
        Compare(mask, self, other, CMPMODE::GT, computeCount);
        Duplicate(self, static_cast<half>(1), computeCount);
        Select(self, mask, self, static_cast<half>(0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        Cast(output, self, RoundMode::CAST_NONE, computeCount);
    } else if constexpr (std::is_same_v<T, float>) {
        Compare(mask, self, other, CMPMODE::GT, computeCount);
        LocalTensor<half> selected = self.template ReinterpretCast<half>();
        Duplicate(selected, static_cast<half>(1), computeCount);
        Select(selected, mask, selected, static_cast<half>(0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        Cast(output, selected, RoundMode::CAST_NONE, computeCount);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> selfFp32 = work0.Get<float>();
        LocalTensor<float> otherFp32 = work1.Get<float>();
        Cast(selfFp32, self, RoundMode::CAST_NONE, computeCount);
        Cast(otherFp32, other, RoundMode::CAST_NONE, computeCount);
        Compare(mask, selfFp32, otherFp32, CMPMODE::GT, computeCount);
        LocalTensor<half> selected = selfFp32.template ReinterpretCast<half>();
        Duplicate(selected, static_cast<half>(1), computeCount);
        Select(selected, mask, selected, static_cast<half>(0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        Cast(output, selected, RoundMode::CAST_NONE, computeCount);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        LocalTensor<half> selfFp16 = work0.Get<half>();
        LocalTensor<half> otherFp16 = work1.Get<half>();
        if (selfScalar || otherScalar) {
            PipeBarrier<PIPE_ALL>();
        }
        if (selfScalar) {
            Duplicate(selfFp16, static_cast<half>(self.GetValue(0)), computeCount);
        } else {
            Cast(selfFp16, self, RoundMode::CAST_NONE, computeCount);
        }
        if (otherScalar) {
            Duplicate(otherFp16, static_cast<half>(other.GetValue(0)), computeCount);
        } else {
            Cast(otherFp16, other, RoundMode::CAST_NONE, computeCount);
        }
        Compare(mask, selfFp16, otherFp16, CMPMODE::GT, computeCount);
        Duplicate(selfFp16, static_cast<half>(1), computeCount);
        Select(selfFp16, mask, selfFp16, static_cast<half>(0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        Cast(output, selfFp16, RoundMode::CAST_NONE, computeCount);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        // DAV_2201 上沿用已经验证的 Min/Max + EQ 语义变换，避免依赖未验证的 INT32 GT。
        Min(other, self, other, computeCount);
        Compare(mask, self, other, CMPMODE::EQ, computeCount);
        LocalTensor<half> selected = self.template ReinterpretCast<half>();
        Duplicate(selected, static_cast<half>(0), computeCount);
        Select(selected, mask, selected, static_cast<half>(1),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        Cast(output, selected, RoundMode::CAST_NONE, computeCount);
    }
}

// 整块连续数据已经由 host 的 UB 账本确认可驻留时，绕过队列调度开销。
template <typename T>
class KernelWholeLinear {
public:
    __aicore__ inline void Process(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        uint64_t totalElements)
    {
        if (totalElements == 0) {
            return;
        }

        const uint32_t validCount = static_cast<uint32_t>(totalElements);
        const uint32_t computeCount = AlignUpU32(validCount, ComputeAlign<T>());
        const uint32_t inputBytes = computeCount * static_cast<uint32_t>(sizeof(T));
        const uint32_t outputBytes = computeCount;
        const uint32_t selfAddr = 0;
        const uint32_t otherAddr =
            AlignUpU32(selfAddr + inputBytes, BUFFER_PHASE_BYTES) + BUFFER_PHASE_BYTES;
        const uint32_t outputAddr =
            AlignUpU32(otherAddr + inputBytes, BUFFER_PHASE_BYTES) + BUFFER_PHASE_BYTES;
        const uint32_t work0Addr =
            AlignUpU32(outputAddr + outputBytes, BUFFER_PHASE_BYTES) + BUFFER_PHASE_BYTES;

        GlobalTensor<T> selfGm;
        GlobalTensor<T> otherGm;
        GlobalTensor<int8_t> outputGm;
        selfGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        otherGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(other));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));

        LocalTensor<T> selfLocal(TPosition::VECIN, selfAddr, computeCount);
        LocalTensor<T> otherLocal(TPosition::VECIN, otherAddr, computeCount);
        LocalTensor<int8_t> outputLocal(TPosition::VECOUT, outputAddr, computeCount);
        const bool inputAligned =
            validCount * static_cast<uint32_t>(sizeof(T)) % 32 == 0;
        if (inputAligned) {
            DataCopy(selfLocal, selfGm, validCount);
            DataCopy(otherLocal, otherGm, validCount);
        } else {
            CopyInPad(selfLocal, selfGm, validCount);
            CopyInPad(otherLocal, otherGm, validCount);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Compute(outputLocal, selfLocal, otherLocal, computeCount, work0Addr);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (validCount % 32 == 0) {
            DataCopy(outputGm, outputLocal, validCount);
        } else {
            CopyOutPad(outputGm, outputLocal, validCount);
        }
    }

private:
    __aicore__ inline void Compute(LocalTensor<int8_t> output, LocalTensor<T> self,
        LocalTensor<T> other, uint32_t computeCount, uint32_t work0Addr)
    {
        LocalTensor<uint8_t> mask = output.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, half>) {
            Compare(mask, self, other, CMPMODE::GT, computeCount);
            Duplicate(self, static_cast<half>(1), computeCount);
            Select(self, mask, self, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, self, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, float>) {
            Compare(mask, self, other, CMPMODE::GT, computeCount);
            LocalTensor<half> selected = self.template ReinterpretCast<half>();
            Duplicate(selected, static_cast<half>(1), computeCount);
            Select(selected, mask, selected, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t work1Addr =
                AlignUpU32(work0Addr + computeCount * sizeof(float), BUFFER_PHASE_BYTES) +
                BUFFER_PHASE_BYTES;
            LocalTensor<float> selfFp32(TPosition::VECCALC, work0Addr, computeCount);
            LocalTensor<float> otherFp32(TPosition::VECCALC, work1Addr, computeCount);
            Cast(selfFp32, self, RoundMode::CAST_NONE, computeCount);
            Cast(otherFp32, other, RoundMode::CAST_NONE, computeCount);
            Compare(mask, selfFp32, otherFp32, CMPMODE::GT, computeCount);
            LocalTensor<half> selected = selfFp32.template ReinterpretCast<half>();
            Duplicate(selected, static_cast<half>(1), computeCount);
            Select(selected, mask, selected, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            const uint32_t work1Addr =
                AlignUpU32(work0Addr + computeCount * sizeof(half), BUFFER_PHASE_BYTES) +
                BUFFER_PHASE_BYTES;
            LocalTensor<half> selfFp16(TPosition::VECCALC, work0Addr, computeCount);
            LocalTensor<half> otherFp16(TPosition::VECCALC, work1Addr, computeCount);
            Cast(selfFp16, self, RoundMode::CAST_NONE, computeCount);
            Cast(otherFp16, other, RoundMode::CAST_NONE, computeCount);
            Compare(mask, selfFp16, otherFp16, CMPMODE::GT, computeCount);
            Duplicate(selfFp16, static_cast<half>(1), computeCount);
            Select(selfFp16, mask, selfFp16, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selfFp16, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            Min(other, self, other, computeCount);
            Compare(mask, self, other, CMPMODE::EQ, computeCount);
            LocalTensor<half> selected = self.template ReinterpretCast<half>();
            Duplicate(selected, static_cast<half>(0), computeCount);
            Select(selected, mask, selected, static_cast<half>(1),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        }
    }
};

template <typename T, uint32_t BUFFER_NUM>
class KernelLinearSpan {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        const uint32_t tileElements, bool outputCacheEnabled, TPipe* pipe)
    {
        pipe_ = pipe;
        tileElements_ = tileElements;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        otherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(other));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (!outputCacheEnabled) {
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        const uint32_t inputBytes = tileElements_ * sizeof(T);
        pipe_->InitBuffer(selfQueue_, BUFFER_NUM, inputBytes);
        if constexpr (BUFFER_NUM == 1) {
            const uint32_t inputBlocks = inputBytes / UB_DATA_BLOCK_BYTES;
            const uint32_t outputBlocks = tileElements_ / UB_DATA_BLOCK_BYTES;
            // 这三类无转换路径在 host 的线性 UB 账本中已预留 phase 空间；
            // 转换路径没有该余量，继续保持原始排布。
            constexpr bool hasPhaseReserve = std::is_same_v<T, half> ||
                std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
            // DAV2201 的 VECIN、VECOUT、VECCALC 共用 UB 地址池。若两路输入的
            // 跨度正好回到同一 bank group，插入 4 个 DataBlock 后三操作数落在
            // 相对 group 0、4、12，避免 Compare/Select 的读读和读写同组冲突。
            const bool useBankSkew = hasPhaseReserve &&
                inputBlocks % UB_BANK_GROUP_COUNT == 0 &&
                outputBlocks % UB_BANK_GROUP_COUNT == 8;
            if (useBankSkew) {
                pipe_->InitBuffer(bankGap_, LINEAR_BANK_SKEW_BYTES);
                pipe_->InitBuffer(outputQueue_, BUFFER_NUM, tileElements_);
                pipe_->InitBuffer(otherQueue_, BUFFER_NUM, inputBytes);
            } else {
                pipe_->InitBuffer(otherQueue_, BUFFER_NUM, inputBytes);
                pipe_->InitBuffer(outputQueue_, BUFFER_NUM, tileElements_);
            }
        } else {
            pipe_->InitBuffer(otherQueue_, BUFFER_NUM, inputBytes);
            pipe_->InitBuffer(outputQueue_, BUFFER_NUM, tileElements_);
            if constexpr (!std::is_same_v<T, bfloat16_t> && !std::is_same_v<T, int8_t>) {
                pipe_->InitBuffer(bankGap_, BUFFER_PHASE_BYTES);
            }
        }
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_->InitBuffer(work0_, tileElements_ * sizeof(float));
            pipe_->InitBuffer(work1_, tileElements_ * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            pipe_->InitBuffer(work0_, tileElements_ * sizeof(half));
            pipe_->InitBuffer(work1_, tileElements_ * sizeof(half));
        }
    }

    __aicore__ inline void Process(uint64_t totalElements, uint32_t activeCores)
    {
        if (totalElements == 0) {
            return;
        }
        const uint64_t block = GetBlockIdx();
        const uint64_t blockNum = activeCores == 0 ? GetBlockNum() : activeCores;
        const uint32_t alignment = ComputeAlign<T>();
        const uint64_t unitCount = (totalElements + alignment - 1) / alignment;
        const uint64_t unitStart = unitCount * block / blockNum;
        const uint64_t unitEnd = unitCount * (block + 1) / blockNum;
        uint64_t offset = unitStart * alignment;
        const uint64_t end = MinU64(totalElements, unitEnd * alignment);
        while (offset < end) {
            const uint32_t validCount = MinU32(end - offset, tileElements_);
            const uint32_t computeCount = AlignUpU32(validCount, alignment);
            LocalTensor<T> self = selfQueue_.template AllocTensor<T>();
            LocalTensor<T> other = otherQueue_.template AllocTensor<T>();
            CopyIn(self, other, offset, validCount);
            selfQueue_.EnQue(self);
            otherQueue_.EnQue(other);
            self = selfQueue_.template DeQue<T>();
            other = otherQueue_.template DeQue<T>();

            LocalTensor<int8_t> output = outputQueue_.template AllocTensor<int8_t>();
            ComputePacked(output, self, other, work0_, work1_, computeCount);
            outputQueue_.EnQue(output);
            selfQueue_.FreeTensor(self);
            otherQueue_.FreeTensor(other);

            output = outputQueue_.template DeQue<int8_t>();
            CopyOut(outputGm_[offset], output, validCount);
            outputQueue_.FreeTensor(output);
            offset += validCount;
        }
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<T> self, LocalTensor<T> other,
        uint64_t offset, uint32_t count)
    {
        const bool aligned = count * static_cast<uint32_t>(sizeof(T)) % 32 == 0;
        if (aligned) {
            DataCopy(self, selfGm_[offset], count);
            DataCopy(other, otherGm_[offset], count);
        } else {
            CopyInPad(self, selfGm_[offset], count);
            CopyInPad(other, otherGm_[offset], count);
        }
    }

    __aicore__ inline void CopyOut(GlobalTensor<int8_t> dst, LocalTensor<int8_t> output,
        uint32_t count)
    {
        if (count % 32 == 0) {
            DataCopy(dst, output, count);
        } else {
            CopyOutPad(dst, output, count);
        }
    }

    TPipe* pipe_{nullptr};
    uint32_t tileElements_{0};
    GlobalTensor<T> selfGm_;
    GlobalTensor<T> otherGm_;
    GlobalTensor<int8_t> outputGm_;
    TQue<TPosition::VECIN, BUFFER_NUM> selfQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM> otherQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<TPosition::VECCALC> bankGap_;
    TBuf<TPosition::VECCALC> work0_;
    TBuf<TPosition::VECCALC> work1_;
};

template <typename T>
class KernelAffineRow {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        const uint64_t* dims, const uint64_t* selfStrides, const uint64_t* otherStrides,
        uint32_t rank, uint64_t inner, uint32_t tileElements, bool outputCacheEnabled,
        TPipe* pipe)
    {
        pipe_ = pipe;
        rank_ = rank;
        inner_ = inner;
        tileElements_ = tileElements;
        for (uint32_t axis = 0; axis < rank_; ++axis) {
            dims_[axis] = dims[axis];
            selfStrides_[axis] = selfStrides[axis];
            otherStrides_[axis] = otherStrides[axis];
        }
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        otherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(other));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (!outputCacheEnabled) {
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_->InitBuffer(selfQueue_, 1, tileElements_ * sizeof(T));
        pipe_->InitBuffer(otherQueue_, 1, tileElements_ * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 1, tileElements_);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_->InitBuffer(work0_, tileElements_ * sizeof(float));
            pipe_->InitBuffer(work1_, tileElements_ * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            pipe_->InitBuffer(work0_, tileElements_ * sizeof(half));
            pipe_->InitBuffer(work1_, tileElements_ * sizeof(half));
        }
    }

    __aicore__ inline void Process(uint64_t rowCount, uint32_t activeCores)
    {
        if (rowCount == 0 || inner_ == 0) {
            return;
        }
        const uint64_t block = GetBlockIdx();
        const uint64_t blockNum = activeCores == 0 ? GetBlockNum() : activeCores;
        const uint64_t rowStart = rowCount * block / blockNum;
        const uint64_t rowEnd = rowCount * (block + 1) / blockNum;
        uint64_t selfBase = 0;
        uint64_t otherBase = 0;
        InitCursor(rowStart, selfBase, otherBase);
        const uint32_t alignment = ComputeAlign<T>();
        for (uint64_t row = rowStart; row < rowEnd; ++row) {
            uint64_t innerOffset = 0;
            while (innerOffset < inner_) {
                const uint32_t validCount = MinU32(inner_ - innerOffset, tileElements_);
                const uint32_t computeCount = AlignUpU32(validCount, alignment);
                const bool selfScalar = selfStrides_[rank_ - 1] == 0;
                const bool otherScalar = otherStrides_[rank_ - 1] == 0;
                LocalTensor<T> self = selfQueue_.template AllocTensor<T>();
                LocalTensor<T> other = otherQueue_.template AllocTensor<T>();
                CopyInput(self, selfGm_[selfBase + (selfScalar ? 0 : innerOffset)],
                    selfScalar ? 1 : validCount);
                CopyInput(other, otherGm_[otherBase + (otherScalar ? 0 : innerOffset)],
                    otherScalar ? 1 : validCount);
                selfQueue_.EnQue(self);
                otherQueue_.EnQue(other);
                self = selfQueue_.template DeQue<T>();
                other = otherQueue_.template DeQue<T>();
                PrepareScalar(self, other, computeCount, selfScalar, otherScalar);

                LocalTensor<int8_t> output = outputQueue_.template AllocTensor<int8_t>();
                ComputePacked(output, self, other, work0_, work1_, computeCount,
                    selfScalar, otherScalar);
                outputQueue_.EnQue(output);
                selfQueue_.FreeTensor(self);
                otherQueue_.FreeTensor(other);
                output = outputQueue_.template DeQue<int8_t>();
                CopyOutput(outputGm_[row * inner_ + innerOffset], output, validCount);
                outputQueue_.FreeTensor(output);
                innerOffset += validCount;
            }
            AdvanceCursor(selfBase, otherBase);
        }
    }

private:
    __aicore__ inline void CopyInput(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
    {
        if (count * static_cast<uint32_t>(sizeof(T)) % 32 == 0) {
            DataCopy(dst, src, count);
        } else {
            CopyInPad(dst, src, count);
        }
    }

    __aicore__ inline void CopyOutput(GlobalTensor<int8_t> dst, LocalTensor<int8_t> src,
        uint32_t count)
    {
        if (count % 32 == 0) {
            DataCopy(dst, src, count);
        } else {
            CopyOutPad(dst, src, count);
        }
    }

    __aicore__ inline void PrepareScalar(LocalTensor<T> self, LocalTensor<T> other,
        uint32_t computeCount, bool selfScalar, bool otherScalar)
    {
        if constexpr (!std::is_same_v<T, int8_t>) {
            if (selfScalar || otherScalar) {
                PipeBarrier<PIPE_ALL>();
                if (selfScalar) {
                    Duplicate(self, self.GetValue(0), computeCount);
                }
                if (otherScalar) {
                    Duplicate(other, other.GetValue(0), computeCount);
                }
            }
        }
    }

    __aicore__ inline void InitCursor(uint64_t row, uint64_t& selfBase, uint64_t& otherBase)
    {
        uint64_t residual = row;
        for (int32_t axis = static_cast<int32_t>(rank_) - 2; axis >= 0; --axis) {
            const uint64_t index = dims_[axis] == 0 ? 0 : residual % dims_[axis];
            residual = dims_[axis] == 0 ? 0 : residual / dims_[axis];
            indices_[axis] = index;
            selfBase += index * selfStrides_[axis];
            otherBase += index * otherStrides_[axis];
        }
    }

    __aicore__ inline void AdvanceCursor(uint64_t& selfBase, uint64_t& otherBase)
    {
        for (int32_t axis = static_cast<int32_t>(rank_) - 2; axis >= 0; --axis) {
            ++indices_[axis];
            selfBase += selfStrides_[axis];
            otherBase += otherStrides_[axis];
            if (indices_[axis] < dims_[axis]) {
                return;
            }
            indices_[axis] = 0;
            selfBase -= selfStrides_[axis] * dims_[axis];
            otherBase -= otherStrides_[axis] * dims_[axis];
        }
    }

    TPipe* pipe_{nullptr};
    uint32_t rank_{0};
    uint32_t tileElements_{0};
    uint64_t inner_{0};
    uint64_t dims_[MAX_RANK]{};
    uint64_t selfStrides_[MAX_RANK]{};
    uint64_t otherStrides_[MAX_RANK]{};
    uint64_t indices_[MAX_RANK]{};
    GlobalTensor<T> selfGm_;
    GlobalTensor<T> otherGm_;
    GlobalTensor<int8_t> outputGm_;
    TQue<TPosition::VECIN, 1> selfQueue_;
    TQue<TPosition::VECIN, 1> otherQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> work0_;
    TBuf<TPosition::VECCALC> work1_;
};

template <typename T, bool SELF_BROADCAST>
class KernelResidentRow {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        const GreaterTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        tileElements_ = tiling_->tileElements;
        rowBatch_ = tiling_->rowBatch;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        otherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(other));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (tiling_->outputCacheEnabled == 0) {
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_->InitBuffer(residentQueue_, 1, tileElements_ * sizeof(T) + BUFFER_PHASE_BYTES);
        pipe_->InitBuffer(fullQueue_, 2,
            rowBatch_ * tileElements_ * sizeof(T));
        pipe_->InitBuffer(outputQueue_, 2, rowBatch_ * tileElements_);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_->InitBuffer(residentWork_, tileElements_ * sizeof(float));
            pipe_->InitBuffer(fullWork_, tileElements_ * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            pipe_->InitBuffer(residentWork_, tileElements_ * sizeof(half));
            pipe_->InitBuffer(fullWork_, tileElements_ * sizeof(half));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            pipe_->InitBuffer(zeroWork_, tileElements_ * sizeof(half));
            LocalTensor<half> zero = zeroWork_.Get<half>();
            // 预置比较掩码和零向量；批量路径随后复用同一块 UB。
            Duplicate(zero, static_cast<half>(1), BUFFER_PHASE_BYTES / sizeof(half));
            PipeBarrier<PIPE_V>();
            SetCmpMask(zero);
            PipeBarrier<PIPE_V>();
            Duplicate(zero, static_cast<half>(0), tileElements_);
            SetMaskNorm();
            SetVectorMask<int32_t, MaskMode::NORMAL>(INT32_STRIPE_ELEMENTS);
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        if (tiling_->partitionMode == ROW_PARTITION_GROUP_MAJOR) {
            const uint64_t groupStart = tiling_->outer * block / blockNum;
            const uint64_t groupEnd = tiling_->outer * (block + 1) / blockNum;
            for (uint64_t group = groupStart; group < groupEnd; ++group) {
                ProcessGroup(group, 0, tiling_->repeat);
            }
            if constexpr (std::is_same_v<T, int32_t>) {
                ResetMask();
            }
            return;
        }

        const uint64_t totalRows = tiling_->rowCount;
        const uint64_t rowStart = totalRows * block / blockNum;
        const uint64_t rowEnd = totalRows * (block + 1) / blockNum;
        uint64_t row = rowStart;
        while (row < rowEnd) {
            const uint64_t group = tiling_->repeat == 0 ? 0 : row / tiling_->repeat;
            const uint64_t repeatStart = tiling_->repeat == 0 ? 0 : row % tiling_->repeat;
            const uint64_t repeatEnd = MinU64(
                tiling_->repeat, repeatStart + rowEnd - row);
            ProcessGroup(group, repeatStart, repeatEnd);
            row += repeatEnd - repeatStart;
        }
        if constexpr (std::is_same_v<T, int32_t>) {
            ResetMask();
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint64_t group, uint64_t repeatStart,
        uint64_t repeatEnd)
    {
        const uint32_t alignment = ComputeAlign<T>();
        uint64_t innerOffset = 0;
        while (innerOffset < tiling_->inner) {
            const uint32_t validCount = MinU32(tiling_->inner - innerOffset, tileElements_);
            const uint32_t computeCount = AlignUpU32(validCount, alignment);
            const uint64_t residentOffset = group * tiling_->inner + innerOffset;
            LocalTensor<T> resident = residentQueue_.template AllocTensor<T>();
            if constexpr (SELF_BROADCAST) {
                CopyInPad(resident, selfGm_[residentOffset], validCount);
            } else {
                CopyInPad(resident, otherGm_[residentOffset], validCount);
            }
            residentQueue_.EnQue(resident);
            resident = residentQueue_.template DeQue<T>();
            PrepareResident(resident, computeCount);

            uint64_t nextRepeat = repeatStart;
            uint32_t currentBatch = 0;
            if (nextRepeat < repeatEnd) {
                currentBatch = static_cast<uint32_t>(
                    MinU64(repeatEnd - nextRepeat, rowBatch_));
                CopyFullBatch(group, nextRepeat, innerOffset, validCount, currentBatch);
                nextRepeat += currentBatch;
            }

            uint64_t repeatIndex = repeatStart;
            while (repeatIndex < repeatEnd) {
                const uint32_t batch = currentBatch;
                LocalTensor<T> full = fullQueue_.template DeQue<T>();
                uint32_t nextBatch = 0;
                if (nextRepeat < repeatEnd) {
                    nextBatch = static_cast<uint32_t>(
                        MinU64(repeatEnd - nextRepeat, rowBatch_));
                    CopyFullBatch(group, nextRepeat, innerOffset, validCount, nextBatch);
                    nextRepeat += nextBatch;
                }

                LocalTensor<int8_t> output = outputQueue_.template AllocTensor<int8_t>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    if (CanUseInt32PhasedBatch(computeCount, batch)) {
                        ComputeInt32BatchPhased(resident, full, output, computeCount, batch);
                    } else {
                        for (uint32_t row = 0; row < batch; ++row) {
                            if constexpr (SELF_BROADCAST) {
                                ComputeResidentRow(output[row * tileElements_], resident,
                                    full[row * tileElements_], computeCount);
                            } else {
                                ComputeResidentRow(output[row * tileElements_],
                                    full[row * tileElements_], resident, computeCount);
                            }
                        }
                    }
                } else {
                    for (uint32_t row = 0; row < batch; ++row) {
                        if constexpr (SELF_BROADCAST) {
                            ComputeResidentRow(output[row * tileElements_], resident,
                                full[row * tileElements_], computeCount);
                        } else {
                            ComputeResidentRow(output[row * tileElements_],
                                full[row * tileElements_], resident, computeCount);
                        }
                    }
                }
                outputQueue_.EnQue(output);
                fullQueue_.FreeTensor(full);

                const uint64_t outputOffset =
                    (group * tiling_->repeat + repeatIndex) * tiling_->inner + innerOffset;
                CopyOutBatch(outputOffset, validCount, batch);
                repeatIndex += batch;
                currentBatch = nextBatch;
            }
            residentQueue_.FreeTensor(resident);
            innerOffset += validCount;
        }
    }

    __aicore__ inline void CopyFullBatch(uint64_t group, uint64_t repeatStart,
        uint64_t innerOffset, uint32_t validCount, uint32_t batch)
    {
        LocalTensor<T> full = fullQueue_.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, int32_t>) {
            if (CopyFullBatchPhaseGrouped(full, group, repeatStart, innerOffset,
                validCount, batch)) {
                fullQueue_.EnQue(full);
                return;
            }
        }
        const uint32_t validBytes = validCount * static_cast<uint32_t>(sizeof(T));
        const uint32_t alignedBytes = AlignUpU32(validBytes, 32);
        const uint32_t localPitchBytes = tileElements_ * static_cast<uint32_t>(sizeof(T));
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        const uint64_t offset =
            (group * tiling_->repeat + repeatStart) * tiling_->inner + innerOffset;
        const uint32_t rowBytes = static_cast<uint32_t>(tiling_->inner * sizeof(T));
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch), validBytes, rowBytes - validBytes,
            (localPitchBytes - alignedBytes) / 32, 0};
        if constexpr (SELF_BROADCAST) {
            DataCopyPad(full, otherGm_[offset], copyParams, padParams);
        } else {
            DataCopyPad(full, selfGm_[offset], copyParams, padParams);
        }
        fullQueue_.EnQue(full);
    }

    // 对同一 32B 相位的行使用普通二维 DataCopy。每行末尾的对齐读取只落在
    // 本行独立 UB slot 的无效 padding 中；末个 GM 元素无法安全多读时保持 Pad 路径。
    __aicore__ inline bool CopyFullBatchPhaseGrouped(LocalTensor<T> full,
        uint64_t group, uint64_t repeatStart, uint64_t innerOffset,
        uint32_t validCount, uint32_t batch)
    {
        const uint32_t validBytes = validCount * sizeof(T);
        const uint32_t alignedBytes = AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES);
        const uint32_t localPitchBytes = tileElements_ * sizeof(T);
        if (batch == 0 || validBytes == alignedBytes || alignedBytes > localPitchBytes ||
            alignedBytes % UB_DATA_BLOCK_BYTES != 0) {
            return false;
        }

        if (tiling_->inner == 0 || tiling_->inner > 0x3fffffffULL) {
            return false;
        }
        const uint32_t rowBytes = static_cast<uint32_t>(tiling_->inner) * sizeof(T);
        // 将相同相位跨行距聚合前，group 跨度必须回到 32B 边界，才能让该
        // 映射在所有 group 上保持相同的数据流；小批量不足以摊销八路 DMA。
        const uint32_t repeatPhase = static_cast<uint32_t>(tiling_->repeat) &
            (UB_DATA_BLOCK_BYTES - 1);
        if ((repeatPhase * (rowBytes & (UB_DATA_BLOCK_BYTES - 1))) &
            (UB_DATA_BLOCK_BYTES - 1)) {
            return false;
        }
        const uint32_t phasePeriod = UB_DATA_BLOCK_BYTES /
            GreatestCommonDivisor(rowBytes, UB_DATA_BLOCK_BYTES);
        if (phasePeriod == 0 || batch < phasePeriod * 2 || rowBytes > 0x07ffffffU) {
            return false;
        }

        const uint32_t sourceStrideBytes = phasePeriod * rowBytes;
        const uint32_t destinationStrideBytes = phasePeriod * localPitchBytes;
        if (sourceStrideBytes < alignedBytes || destinationStrideBytes < alignedBytes ||
            (sourceStrideBytes - alignedBytes) % UB_DATA_BLOCK_BYTES != 0 ||
            (destinationStrideBytes - alignedBytes) % UB_DATA_BLOCK_BYTES != 0) {
            return false;
        }
        const uint32_t sourceGap =
            (sourceStrideBytes - alignedBytes) / UB_DATA_BLOCK_BYTES;
        const uint32_t destinationGap =
            (destinationStrideBytes - alignedBytes) / UB_DATA_BLOCK_BYTES;
        const uint32_t blockLen = alignedBytes / UB_DATA_BLOCK_BYTES;
        if (sourceGap > 0xffffU || destinationGap > 0xffffU || blockLen == 0 ||
            blockLen > 0xffffU) {
            return false;
        }

        const uint64_t offset =
            (group * tiling_->repeat + repeatStart) * tiling_->inner + innerOffset;
        // 不读取完整输入逻辑末尾之外的 4~28 B；那一行回退到 DataCopyPad。
        if (group + 1 == tiling_->outer && repeatStart + batch == tiling_->repeat &&
            innerOffset + validCount == tiling_->inner) {
            return false;
        }

        uint64_t phaseOffset = offset;
        for (uint32_t phase = 0; phase < phasePeriod && phase < batch; ++phase) {
            const uint32_t phaseCount =
                (batch - phase + phasePeriod - 1) / phasePeriod;
            if (phaseCount == 0 || phaseCount > MAX_DMA_BLOCK_COUNT) {
                return false;
            }
            const DataCopyParams copyParams{
                static_cast<uint16_t>(phaseCount), static_cast<uint16_t>(blockLen),
                static_cast<uint16_t>(sourceGap), static_cast<uint16_t>(destinationGap)};
            if constexpr (SELF_BROADCAST) {
                DataCopy(full[phase * tileElements_], otherGm_[phaseOffset], copyParams);
            } else {
                DataCopy(full[phase * tileElements_], selfGm_[phaseOffset], copyParams);
            }
            phaseOffset += tiling_->inner;
        }
        return true;
    }

    __aicore__ inline bool CanUseInt32PhasedBatch(uint32_t computeCount,
        uint32_t batch) const
    {
        if (computeCount == 0 || computeCount % INT32_STRIPE_ELEMENTS != 0 ||
            tileElements_ < 2 * INT32_STRIPE_ELEMENTS ||
            tileElements_ % INT32_STRIPE_ELEMENTS != 0 || batch == 0 ||
            batch > MAX_VECTOR_REPEAT) {
            return false;
        }
        // BinaryRepeatParams 的 stride 字段按 32B DataBlock 计数且为 uint8_t。
        return tileElements_ / 8 <= MAX_VECTOR_REPEAT;
    }

    __aicore__ inline void ComputeInt32BatchPhased(LocalTensor<T> resident,
        LocalTensor<T> full, LocalTensor<int8_t> output, uint32_t computeCount,
        uint32_t batch)
    {
        constexpr uint32_t maskBytesPerRow = INT32_STRIPE_ELEMENTS / 8;
        const uint8_t repeat = static_cast<uint8_t>(batch);
        const uint32_t pairedRows = AlignUpU32(batch, 2);
        const uint8_t selectRepeat = static_cast<uint8_t>(pairedRows / 2);
        const uint8_t fullPitch = static_cast<uint8_t>(tileElements_ / 8);
        const uint8_t outputPitch = static_cast<uint8_t>(tileElements_ / 32);
        const uint32_t maskStripeBytes = AlignUpU32(
            pairedRows * maskBytesPerRow, 32);
        const BinaryRepeatParams minMaxParams{1, 1, 1, fullPitch,
            static_cast<uint8_t>(SELF_BROADCAST ? 0 : fullPitch),
            static_cast<uint8_t>(SELF_BROADCAST ? fullPitch : 0)};
        const BinaryRepeatParams compareParams{0, 1, 1, maskBytesPerRow, 0,
            fullPitch};
        const BinaryRepeatParams selectParams{1, 1, 0, 8, 0, 8};
        const UnaryRepeatParams castParams{1, 1, outputPitch,
            INT32_STRIPE_ELEMENTS * sizeof(half) / 32};
        LocalTensor<uint8_t> mask = output.template ReinterpretCast<uint8_t>();
        LocalTensor<half> selected = full.template ReinterpretCast<half>();
        LocalTensor<half> zero = zeroWork_.Get<half>();

        // 先做整批 Min/Max，避免原位压缩覆盖后续仍需比较的 INT32 数据。
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            if constexpr (SELF_BROADCAST) {
                Min<int32_t, false>(full[stripe], resident[stripe], full[stripe],
                    MASK_PLACEHOLDER, repeat, minMaxParams);
            } else {
                Max<int32_t, false>(full[stripe], full[stripe], resident[stripe],
                    MASK_PLACEHOLDER, repeat, minMaxParams);
            }
        }
        PipeBarrier<PIPE_V>();

        // 每个 256B stripe 的比较结果是每行 8B 的 bit mask。
        uint32_t stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Compare<int32_t, uint8_t, false>(
                mask[stripeIndex * maskStripeBytes], resident[stripe],
                full[stripe], CMPMODE::EQ, MASK_PLACEHOLDER, repeat,
                compareParams);
            ++stripeIndex;
        }
        PipeBarrier<PIPE_V>();

        // Select 每次处理相邻两行的 128 个 half，输出按 stripe 紧凑排列。
        SetVectorMask<half, MaskMode::NORMAL>(2 * INT32_STRIPE_ELEMENTS);
        stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Select(selected[stripeIndex * pairedRows * INT32_STRIPE_ELEMENTS],
                mask[stripeIndex * maskStripeBytes], zero, selectRepeat,
                selectParams);
            ++stripeIndex;
        }
        PipeBarrier<PIPE_V>();

        // Cast 将 stripe-major 的 half 结果按原始行距写回 output。
        SetVectorMask<half, MaskMode::NORMAL>(INT32_STRIPE_ELEMENTS);
        stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Cast<int8_t, half, false>(output[stripe],
                selected[stripeIndex * pairedRows * INT32_STRIPE_ELEMENTS],
                RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeat, castParams);
            ++stripeIndex;
        }
    }

    __aicore__ inline void PrepareResident(LocalTensor<T> resident, uint32_t computeCount)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            Cast(residentWork_.Get<float>(), resident, RoundMode::CAST_NONE, computeCount);
            Duplicate(resident.template ReinterpretCast<half>()[BUFFER_PHASE_BYTES / sizeof(half)],
                static_cast<half>(1), computeCount);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            Cast(residentWork_.Get<half>(), resident, RoundMode::CAST_NONE, computeCount);
        }
    }

    __aicore__ inline void ComputeResidentRow(LocalTensor<int8_t> output,
        LocalTensor<T> self, LocalTensor<T> other, uint32_t computeCount)
    {
        LocalTensor<uint8_t> mask = output.template ReinterpretCast<uint8_t>();
        LocalTensor<T> resident;
        LocalTensor<T> full;
        if constexpr (SELF_BROADCAST) {
            resident = self;
            full = other;
        } else {
            resident = other;
            full = self;
        }
        if constexpr (std::is_same_v<T, half>) {
            Compare(mask, resident, full, CMPMODE::GT, computeCount);
            Duplicate(full, static_cast<half>(1), computeCount);
            Select(full, mask, full, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, full, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, float>) {
            Compare(mask, resident, full, CMPMODE::GT, computeCount);
            LocalTensor<half> selected = full.template ReinterpretCast<half>();
            Duplicate(selected, static_cast<half>(1), computeCount);
            Select(selected, mask, selected, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<T> full;
            if constexpr (SELF_BROADCAST) {
                full = other;
            } else {
                full = self;
            }
            Cast(fullWork_.Get<float>(), full, RoundMode::CAST_NONE, computeCount);
            if constexpr (SELF_BROADCAST) {
                Compare(mask, residentWork_.Get<float>(), fullWork_.Get<float>(),
                    CMPMODE::GT, computeCount);
            } else {
                Compare(mask, fullWork_.Get<float>(), residentWork_.Get<float>(),
                    CMPMODE::GT, computeCount);
            }
            LocalTensor<half> selected = fullWork_.Get<float>().template ReinterpretCast<half>();
            LocalTensor<half> ones = resident.template ReinterpretCast<half>()[BUFFER_PHASE_BYTES / sizeof(half)];
            Select(selected, mask, ones, static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<T> full;
            if constexpr (SELF_BROADCAST) {
                full = other;
            } else {
                full = self;
            }
            Cast(fullWork_.Get<half>(), full, RoundMode::CAST_NONE, computeCount);
            if constexpr (SELF_BROADCAST) {
                Compare(mask, residentWork_.Get<half>(), fullWork_.Get<half>(),
                    CMPMODE::GT, computeCount);
            } else {
                Compare(mask, fullWork_.Get<half>(), residentWork_.Get<half>(),
                    CMPMODE::GT, computeCount);
            }
            Duplicate(fullWork_.Get<half>(), static_cast<half>(1), computeCount);
            Select(fullWork_.Get<half>(), mask, fullWork_.Get<half>(),
                static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, fullWork_.Get<half>(), RoundMode::CAST_NONE, computeCount);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if constexpr (SELF_BROADCAST) {
                Min(full, resident, full, computeCount);
                Compare(mask, resident, full, CMPMODE::EQ, computeCount);
            } else {
                Max(full, full, resident, computeCount);
                Compare(mask, resident, full, CMPMODE::EQ, computeCount);
            }
            LocalTensor<half> selected = full.template ReinterpretCast<half>();
            Select(selected, mask, zeroWork_.Get<half>(), static_cast<half>(1),
                SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            Cast(output, selected, RoundMode::CAST_NONE, computeCount);
        }
    }

    __aicore__ inline void CopyOutBatch(uint64_t offset, uint32_t count, uint32_t batch)
    {
        LocalTensor<int8_t> output = outputQueue_.template DeQue<int8_t>();
        const uint32_t alignedBytes = AlignUpU32(count, 32);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch),
            count,
            (tileElements_ - alignedBytes) / 32,
            static_cast<uint32_t>(tiling_->inner - count),
            0};
        DataCopyPad(outputGm_[offset], output, copyParams);
        outputQueue_.FreeTensor(output);
    }

    const GreaterTilingData* tiling_{nullptr};
    uint32_t tileElements_{0};
    uint32_t rowBatch_{1};
    GlobalTensor<T> selfGm_;
    GlobalTensor<T> otherGm_;
    GlobalTensor<int8_t> outputGm_;
    TQue<TPosition::VECIN, 1> residentQueue_;
    TQue<TPosition::VECIN, 2> fullQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> residentWork_;
    TBuf<TPosition::VECCALC> fullWork_;
    TBuf<TPosition::VECCALC> zeroWork_;
    TPipe* pipe_{nullptr};
};

// INT32 非对齐行直接以独立的 32B 行距搬入 UB。full 行不再经过 Broadcast/GatherMask，
// 比较阶段通过 BinaryRepeatParams 的零 resident stride 复用常驻行。
template <bool SELF_BROADCAST>
class KernelResidentPitchedRow {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        const GreaterTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        computeElements_ = AlignUpU32(static_cast<uint32_t>(tiling_->inner),
            INT32_COMPARE_ALIGN);
        // full/output 行额外保留一个 DataBlock，既保持每行起址对齐，也打散
        // 连续行在同一 UB bank group 上的映射。
        fullPitchElements_ = computeElements_ + 8;
        outputPitchElements_ = AlignUpU32(static_cast<uint32_t>(tiling_->inner),
            UB_DATA_BLOCK_BYTES) + UB_DATA_BLOCK_BYTES;
        packetRows_ = tiling_->packedPacketRows;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self));
        otherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(other));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (tiling_->outputCacheEnabled == 0) {
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        pipe_->InitBuffer(residentQueue_, 1, computeElements_ * sizeof(int32_t));
        pipe_->InitBuffer(fullQueue_, 2,
            packetRows_ * fullPitchElements_ * sizeof(int32_t));
        pipe_->InitBuffer(outputQueue_, 2, packetRows_ * outputPitchElements_);
        pipe_->InitBuffer(zeroWork_, computeElements_ * sizeof(half));

        LocalTensor<half> zero = zeroWork_.Get<half>();
        // 该路径使用 Compare 的 tensor mask，不依赖 CMPMASK SPR；zero buffer
        // 只需初始化为 Select 的零向量，避免无效的 mask 预置和屏障。
        Duplicate(zero, static_cast<half>(0), computeElements_);
        SetMaskNorm();
        SetVectorMask<int32_t, MaskMode::NORMAL>(INT32_STRIPE_ELEMENTS);
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t groupStart = tiling_->outer * block / blockNum;
        const uint64_t groupEnd = tiling_->outer * (block + 1) / blockNum;
        for (uint64_t group = groupStart; group < groupEnd; ++group) {
            ProcessGroup(group);
        }
        ResetMask();
    }

private:
    __aicore__ inline void ProcessGroup(uint64_t group)
    {
        LocalTensor<int32_t> resident = residentQueue_.AllocTensor<int32_t>();
        const uint64_t residentOffset = group * tiling_->inner;
        if constexpr (SELF_BROADCAST) {
            CopyInPad(resident, selfGm_[residentOffset],
                static_cast<uint32_t>(tiling_->inner));
        } else {
            CopyInPad(resident, otherGm_[residentOffset],
                static_cast<uint32_t>(tiling_->inner));
        }
        residentQueue_.EnQue(resident);
        resident = residentQueue_.DeQue<int32_t>();

        uint64_t nextRepeat = 0;
        if (nextRepeat < tiling_->repeat) {
            const uint32_t batch = static_cast<uint32_t>(MinU64(
                tiling_->repeat - nextRepeat, packetRows_));
            CopyFullBatch(group, nextRepeat, batch);
            nextRepeat += batch;
        }

        uint64_t repeatIndex = 0;
        while (repeatIndex < tiling_->repeat) {
            const uint32_t batch = static_cast<uint32_t>(MinU64(
                tiling_->repeat - repeatIndex, packetRows_));
            LocalTensor<int32_t> full = fullQueue_.DeQue<int32_t>();
            if (nextRepeat < tiling_->repeat) {
                const uint32_t nextBatch = static_cast<uint32_t>(MinU64(
                    tiling_->repeat - nextRepeat, packetRows_));
                CopyFullBatch(group, nextRepeat, nextBatch);
                nextRepeat += nextBatch;
            }

            LocalTensor<int8_t> output = outputQueue_.AllocTensor<int8_t>();
            ComputeInt32BatchPitched(resident, full, output, computeElements_, batch);
            outputQueue_.EnQue(output);
            fullQueue_.FreeTensor(full);

            output = outputQueue_.DeQue<int8_t>();
            CopyOutBatch(group, repeatIndex, output, batch);
            outputQueue_.FreeTensor(output);
            repeatIndex += batch;
        }
        residentQueue_.FreeTensor(resident);
    }

    __aicore__ inline void CopyFullBatch(uint64_t group, uint64_t repeatStart,
        uint32_t batch)
    {
        LocalTensor<int32_t> full = fullQueue_.AllocTensor<int32_t>();
        if (CopyFullBatchContiguousPad(full, group, repeatStart, batch)) {
            fullQueue_.EnQue(full);
            return;
        }
        if (CopyFullBatchPhaseGrouped(full, group, repeatStart, batch)) {
            fullQueue_.EnQue(full);
            return;
        }
        // INT32 行的有效长度不是 32B 整数倍时，逐行使用对齐长度的普通 DMA。
        // 除最后一个逻辑行外，多读的尾部元素仍在输入张量内；这样避开
        // DataCopyPad 对每个非对齐 block 的逐行补齐开销，尾行仍保留安全回退。
        const uint32_t validBytes = static_cast<uint32_t>(tiling_->inner * sizeof(int32_t));
        const uint32_t alignedElements = AlignUpU32(
            static_cast<uint32_t>(tiling_->inner), UB_DATA_BLOCK_BYTES / sizeof(int32_t));
        const uint64_t totalRows = tiling_->outer * tiling_->repeat;
        const uint64_t firstRow = group * tiling_->repeat + repeatStart;
        const bool canCopyRows = validBytes % UB_DATA_BLOCK_BYTES != 0 &&
            alignedElements <= computeElements_;
        if (canCopyRows) {
            for (uint32_t row = 0; row < batch; ++row) {
                const uint64_t globalRow = firstRow + row;
                const uint64_t offset = globalRow * tiling_->inner;
                if (globalRow + 1 < totalRows) {
                    if constexpr (SELF_BROADCAST) {
                        DataCopy(full[row * fullPitchElements_], otherGm_[offset], alignedElements);
                    } else {
                        DataCopy(full[row * fullPitchElements_], selfGm_[offset], alignedElements);
                    }
                } else {
                    const DataCopyExtParams copyParams{
                        1, validBytes, 0,
                        static_cast<uint32_t>((fullPitchElements_ * sizeof(int32_t) -
                            AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES)) / UB_DATA_BLOCK_BYTES), 0};
                    const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
                    if constexpr (SELF_BROADCAST) {
                        DataCopyPad(full[row * fullPitchElements_], otherGm_[offset],
                            copyParams, padParams);
                    } else {
                        DataCopyPad(full[row * fullPitchElements_], selfGm_[offset],
                            copyParams, padParams);
                    }
                }
            }
            fullQueue_.EnQue(full);
            return;
        }
        const uint32_t alignedBytes = AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES);
        const uint32_t fullPitchBytes = fullPitchElements_ * sizeof(int32_t);
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch), validBytes, 0,
            (fullPitchBytes - alignedBytes) / UB_DATA_BLOCK_BYTES, 0};
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        const uint64_t offset =
            (group * tiling_->repeat + repeatStart) * tiling_->inner;
        if constexpr (SELF_BROADCAST) {
            DataCopyPad(full, otherGm_[offset], copyParams, padParams);
        } else {
            DataCopyPad(full, selfGm_[offset], copyParams, padParams);
        }
        fullQueue_.EnQue(full);
    }

    // 行在 GM 中本身连续，只是单行长度不是 32B 整数倍。DataCopyPad 的
    // blockCount 可以直接沿用有效行宽和连续源地址，并在 UB 端按物理行距
    // 跳过每行的对齐填充；相比按相位拆成多条二维 DMA，可将一整个 batch
    // 压缩为一条搬运命令。这里不读取行尾之外的 GM，最后逻辑行也安全。
    __aicore__ inline bool CopyFullBatchContiguousPad(LocalTensor<int32_t> full,
        uint64_t group, uint64_t repeatStart, uint32_t batch)
    {
        const uint32_t validBytes = static_cast<uint32_t>(tiling_->inner * sizeof(int32_t));
        const uint32_t alignedBytes = AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES);
        if (validBytes == 0 || validBytes % UB_DATA_BLOCK_BYTES == 0 || batch == 0 ||
            batch > MAX_DMA_BLOCK_COUNT || alignedBytes > fullPitchElements_ * sizeof(int32_t)) {
            return false;
        }

        const uint32_t fullPitchBytes = fullPitchElements_ * sizeof(int32_t);
        if ((fullPitchBytes - alignedBytes) % UB_DATA_BLOCK_BYTES != 0) {
            return false;
        }
        const uint64_t offset =
            (group * tiling_->repeat + repeatStart) * tiling_->inner;
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch), validBytes, 0,
            (fullPitchBytes - alignedBytes) / UB_DATA_BLOCK_BYTES, 0};
        const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        if constexpr (SELF_BROADCAST) {
            DataCopyPad(full, otherGm_[offset], copyParams, padParams);
        } else {
            DataCopyPad(full, selfGm_[offset], copyParams, padParams);
        }
        return true;
    }

    // 将同一 32B 相位的行聚合成带 blockCount 的普通二维 DataCopy。相邻相位
    // 的 GM 起址可能不对齐，但每个相位内跨 phaseRows 行后会重新回到 32B
    // 边界；这样在 packetRows 大于相位周期时，DMA 命令数从 batch 降为相位数。
    // 逻辑末行不能多读时返回 false，交给上面的逐行安全回退处理。
    __aicore__ inline bool CopyFullBatchPhaseGrouped(LocalTensor<int32_t> full,
        uint64_t group, uint64_t repeatStart, uint32_t batch)
    {
        const uint32_t validBytes = static_cast<uint32_t>(tiling_->inner * sizeof(int32_t));
        const uint32_t alignedBytes = AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES);
        const uint32_t phaseRows = UB_DATA_BLOCK_BYTES /
            GreatestCommonDivisor(validBytes, UB_DATA_BLOCK_BYTES);
        if (batch < phaseRows || phaseRows <= 1 || alignedBytes > fullPitchElements_ * sizeof(int32_t) ||
            alignedBytes % UB_DATA_BLOCK_BYTES != 0 || validBytes == 0) {
            return false;
        }

        const uint64_t totalRows = tiling_->outer * tiling_->repeat;
        const uint64_t firstRow = group * tiling_->repeat + repeatStart;
        if (firstRow + batch >= totalRows) {
            return false;
        }
        const uint32_t rowBytes = validBytes;
        const uint32_t sourceStrideBytes = phaseRows * rowBytes;
        const uint32_t destinationStrideBytes = phaseRows * fullPitchElements_ *
            static_cast<uint32_t>(sizeof(int32_t));
        if (sourceStrideBytes < alignedBytes || destinationStrideBytes < alignedBytes ||
            (sourceStrideBytes - alignedBytes) % UB_DATA_BLOCK_BYTES != 0 ||
            (destinationStrideBytes - alignedBytes) % UB_DATA_BLOCK_BYTES != 0) {
            return false;
        }

        const uint32_t sourceGap = (sourceStrideBytes - alignedBytes) / UB_DATA_BLOCK_BYTES;
        const uint32_t destinationGap =
            (destinationStrideBytes - alignedBytes) / UB_DATA_BLOCK_BYTES;
        const uint32_t blockLen = alignedBytes / UB_DATA_BLOCK_BYTES;
        if (sourceGap > 0xffffU || destinationGap > 0xffffU ||
            blockLen == 0 || blockLen > 0xffffU) {
            return false;
        }

        uint64_t phaseOffset = firstRow * tiling_->inner;
        for (uint32_t phase = 0; phase < phaseRows && phase < batch; ++phase) {
            const uint32_t phaseCount = (batch - phase + phaseRows - 1) / phaseRows;
            if (phaseCount == 0 || phaseCount > MAX_DMA_BLOCK_COUNT) {
                return false;
            }
            const DataCopyParams copyParams{
                static_cast<uint16_t>(phaseCount), static_cast<uint16_t>(blockLen),
                static_cast<uint16_t>(sourceGap), static_cast<uint16_t>(destinationGap)};
            if constexpr (SELF_BROADCAST) {
                DataCopy(full[phase * fullPitchElements_], otherGm_[phaseOffset], copyParams);
            } else {
                DataCopy(full[phase * fullPitchElements_], selfGm_[phaseOffset], copyParams);
            }
            phaseOffset += tiling_->inner;
        }
        return true;
    }

    __aicore__ inline void ComputeInt32BatchPitched(LocalTensor<int32_t> resident,
        LocalTensor<int32_t> full, LocalTensor<int8_t> output, uint32_t computeCount,
        uint32_t batch)
    {
        constexpr uint32_t maskBytesPerRow = INT32_STRIPE_ELEMENTS / 8;
        const uint8_t repeat = static_cast<uint8_t>(batch);
        const uint32_t pairedRows = AlignUpU32(batch, 2);
        const uint8_t selectRepeat = static_cast<uint8_t>(pairedRows / 2);
        const uint8_t fullPitch = static_cast<uint8_t>(fullPitchElements_ / 8);
        const uint8_t outputPitch = static_cast<uint8_t>(outputPitchElements_ / 32);
        const uint32_t maskStripeBytes = AlignUpU32(
            pairedRows * maskBytesPerRow, UB_DATA_BLOCK_BYTES);
        const BinaryRepeatParams minMaxParams{1, 1, 1, fullPitch,
            static_cast<uint8_t>(SELF_BROADCAST ? 0 : fullPitch),
            static_cast<uint8_t>(SELF_BROADCAST ? fullPitch : 0)};
        const BinaryRepeatParams compareParams{0, 1, 1, maskBytesPerRow, 0,
            fullPitch};
        const BinaryRepeatParams selectParams{1, 1, 0, 8, 0, 8};
        const UnaryRepeatParams castParams{1, 1, outputPitch,
            INT32_STRIPE_ELEMENTS * sizeof(half) / 32};
        LocalTensor<uint8_t> mask = output.ReinterpretCast<uint8_t>();
        LocalTensor<half> selected = full.ReinterpretCast<half>();
        LocalTensor<half> zero = zeroWork_.Get<half>();

        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            if constexpr (SELF_BROADCAST) {
                Min<int32_t, false>(full[stripe], resident[stripe], full[stripe],
                    MASK_PLACEHOLDER, repeat, minMaxParams);
            } else {
                Max<int32_t, false>(full[stripe], full[stripe], resident[stripe],
                    MASK_PLACEHOLDER, repeat, minMaxParams);
            }
        }
        PipeBarrier<PIPE_V>();

        uint32_t stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Compare<int32_t, uint8_t, false>(
                mask[stripeIndex * maskStripeBytes], resident[stripe], full[stripe],
                CMPMODE::EQ, MASK_PLACEHOLDER, repeat, compareParams);
            ++stripeIndex;
        }
        PipeBarrier<PIPE_V>();

        SetVectorMask<half, MaskMode::NORMAL>(2 * INT32_STRIPE_ELEMENTS);
        stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Select(selected[stripeIndex * pairedRows * INT32_STRIPE_ELEMENTS],
                mask[stripeIndex * maskStripeBytes], zero, selectRepeat, selectParams);
            ++stripeIndex;
        }
        PipeBarrier<PIPE_V>();

        SetVectorMask<half, MaskMode::NORMAL>(INT32_STRIPE_ELEMENTS);
        stripeIndex = 0;
        for (uint32_t stripe = 0; stripe < computeCount;
            stripe += INT32_STRIPE_ELEMENTS) {
            Cast<int8_t, half, false>(output[stripe],
                selected[stripeIndex * pairedRows * INT32_STRIPE_ELEMENTS],
                RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeat, castParams);
            ++stripeIndex;
        }
    }

    __aicore__ inline void CopyOutBatch(uint64_t group, uint64_t repeatStart,
        LocalTensor<int8_t> output, uint32_t batch)
    {
        const uint32_t validBytes = static_cast<uint32_t>(tiling_->inner);
        const uint32_t alignedBytes = AlignUpU32(validBytes, UB_DATA_BLOCK_BYTES);
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch), validBytes,
            (outputPitchElements_ - alignedBytes) / UB_DATA_BLOCK_BYTES, 0, 0};
        const uint64_t offset =
            (group * tiling_->repeat + repeatStart) * tiling_->inner;
        DataCopyPad(outputGm_[offset], output, copyParams);
    }

    const GreaterTilingData* tiling_{nullptr};
    uint32_t computeElements_{0};
    uint32_t fullPitchElements_{0};
    uint32_t outputPitchElements_{0};
    uint32_t packetRows_{1};
    GlobalTensor<int32_t> selfGm_;
    GlobalTensor<int32_t> otherGm_;
    GlobalTensor<int8_t> outputGm_;
    TQue<TPosition::VECIN, 1> residentQueue_;
    TQue<TPosition::VECIN, 2> fullQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> zeroWork_;
    TPipe* pipe_{nullptr};
};

// INT32 非对齐行按最小 32B 相位包连续读取。完整输入保持紧凑 row-major，
// 常驻行在 UB 内以同一紧凑布局展开，从而避免每个包的 phase-strided MTE2。
template <bool SELF_BROADCAST>
class KernelResidentPackedRow {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR output,
        const GreaterTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        tileElements_ = tiling_->tileElements;
        const uint32_t rowBytes = static_cast<uint32_t>(tiling_->inner) * sizeof(int32_t);
        phaseRows_ = UB_DATA_BLOCK_BYTES / GreatestCommonDivisor(rowBytes, UB_DATA_BLOCK_BYTES);
        packetRows_ = tiling_->packedPacketRows == 0 ? phaseRows_ : tiling_->packedPacketRows;
        packetElements_ = packetRows_ * static_cast<uint32_t>(tiling_->inner);
        computeElements_ = AlignUpU32(packetElements_, INT32_COMPARE_ALIGN);
        paddedPacketElements_ = packetRows_ * tileElements_;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self));
        otherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(other));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (tiling_->outputCacheEnabled == 0) {
            outputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        pipe_->InitBuffer(residentQueue_, 1, tileElements_ * sizeof(int32_t));
        pipe_->InitBuffer(fullQueue_, 2, computeElements_ * sizeof(int32_t));
        pipe_->InitBuffer(outputQueue_, 2, computeElements_);
        // GatherMask 的模式地址会落在源 tensor 后一个 256B 窗口内，显式保留两相位。
        pipe_->InitBuffer(residentPadded_,
            paddedPacketElements_ * sizeof(int32_t) + 2 * BUFFER_PHASE_BYTES);
        pipe_->InitBuffer(residentPacked_, computeElements_ * sizeof(int32_t));
        pipe_->InitBuffer(broadcastTmp_, BUFFER_PHASE_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t groupStart = tiling_->outer * block / blockNum;
        const uint64_t groupEnd = tiling_->outer * (block + 1) / blockNum;
        for (uint64_t group = groupStart; group < groupEnd; ++group) {
            ProcessGroup(group);
        }
        ResetMask();
    }

private:
    __aicore__ inline void ProcessGroup(uint64_t group)
    {
        LocalTensor<int32_t> resident = residentQueue_.template AllocTensor<int32_t>();
        const uint64_t residentOffset = group * tiling_->inner;
        if constexpr (SELF_BROADCAST) {
            CopyInPad(resident, selfGm_[residentOffset], static_cast<uint32_t>(tiling_->inner));
        } else {
            CopyInPad(resident, otherGm_[residentOffset], static_cast<uint32_t>(tiling_->inner));
        }
        residentQueue_.EnQue(resident);
        resident = residentQueue_.template DeQue<int32_t>();

        // resident 的 MTE2 已完成后立即发起首个连续 full 读取；随后用 Vector 展开 resident，
        // 两条流水可重叠，避免把一次性的 UB 重排放到 MTE2 关键路径之后。
        CopyFullPacket(group, 0);
        BuildPackedResident(resident);
        residentQueue_.FreeTensor(resident);

        uint64_t nextRepeat = packetRows_;
        uint64_t repeatIndex = 0;
        while (repeatIndex < tiling_->repeat) {
            LocalTensor<int32_t> full = fullQueue_.template DeQue<int32_t>();
            if (nextRepeat < tiling_->repeat) {
                CopyFullPacket(group, nextRepeat);
                nextRepeat += packetRows_;
            }

            LocalTensor<int8_t> output = outputQueue_.template AllocTensor<int8_t>();
            ComputePacked(output, residentPacked_.Get<int32_t>(), full);
            outputQueue_.EnQue(output);
            fullQueue_.FreeTensor(full);

            output = outputQueue_.template DeQue<int8_t>();
            CopyOutPacket(group, repeatIndex, output);
            outputQueue_.FreeTensor(output);
            repeatIndex += packetRows_;
        }
    }

    __aicore__ inline void BuildPackedResident(LocalTensor<int32_t> resident)
    {
        LocalTensor<int32_t> padded = residentPadded_.Get<int32_t>();
        LocalTensor<half> paddedHalf = padded.template ReinterpretCast<half>();
        LocalTensor<half> residentHalf = resident.template ReinterpretCast<half>();
        const uint32_t dstShape[2] = {packetRows_, tileElements_ * 2};
        const uint32_t srcShape[2] = {1, tileElements_ * 2};
        LocalTensor<uint8_t> tmp = broadcastTmp_.Get<uint8_t>();
        Broadcast<half, 2, 0>(paddedHalf, residentHalf, dstShape, srcShape, tmp);

        LocalTensor<half> packed = residentPacked_.Get<int32_t>().template ReinterpretCast<half>();
        const GatherMaskParams gatherParams{
            1, static_cast<uint16_t>(packetRows_), static_cast<uint16_t>(tileElements_ / 8), 0};
        uint64_t remainCount = 0;
        // pattern 7 是 DAV2201 高阶 Broadcast 对非对齐尾轴使用的紧凑化模式。
        GatherMask(packed, paddedHalf, static_cast<uint8_t>(7), true,
            static_cast<uint32_t>(tiling_->inner * 2), gatherParams, remainCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyFullPacket(uint64_t group, uint64_t repeatIndex)
    {
        LocalTensor<int32_t> full = fullQueue_.template AllocTensor<int32_t>();
        const uint64_t offset =
            (group * tiling_->repeat + repeatIndex) * tiling_->inner;
        if (packetRows_ == phaseRows_) {
            if constexpr (SELF_BROADCAST) {
                DataCopy(full, otherGm_[offset], packetElements_);
            } else {
                DataCopy(full, selfGm_[offset], packetElements_);
            }
        } else {
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(packetElements_ * sizeof(int32_t)), 0, 0, 0};
            const DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            if constexpr (SELF_BROADCAST) {
                DataCopyPad(full, otherGm_[offset], copyParams, padParams);
            } else {
                DataCopyPad(full, selfGm_[offset], copyParams, padParams);
            }
        }
        fullQueue_.EnQue(full);
    }

    __aicore__ inline void ComputePacked(LocalTensor<int8_t> output,
        LocalTensor<int32_t> resident, LocalTensor<int32_t> full)
    {
        LocalTensor<uint8_t> mask = output.template ReinterpretCast<uint8_t>();
        if constexpr (SELF_BROADCAST) {
            Min(full, resident, full, computeElements_);
            Compare(mask, resident, full, CMPMODE::EQ, computeElements_);
        } else {
            Max(full, full, resident, computeElements_);
            Compare(mask, resident, full, CMPMODE::EQ, computeElements_);
        }
        LocalTensor<half> selected = full.template ReinterpretCast<half>();
        Duplicate(selected, static_cast<half>(0), computeElements_);
        Select(selected, mask, selected, static_cast<half>(1),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElements_);
        Cast(output, selected, RoundMode::CAST_NONE, computeElements_);
    }

    __aicore__ inline void CopyOutPacket(uint64_t group, uint64_t repeatIndex,
        LocalTensor<int8_t> output)
    {
        const uint64_t offset =
            (group * tiling_->repeat + repeatIndex) * tiling_->inner;
        const DataCopyExtParams copyParams{1, packetElements_, 0, 0, 0};
        DataCopyPad(outputGm_[offset], output, copyParams);
    }

    const GreaterTilingData* tiling_{nullptr};
    uint32_t tileElements_{0};
    uint32_t packetRows_{1};
    uint32_t phaseRows_{1};
    uint32_t packetElements_{0};
    uint32_t computeElements_{0};
    uint32_t paddedPacketElements_{0};
    GlobalTensor<int32_t> selfGm_;
    GlobalTensor<int32_t> otherGm_;
    GlobalTensor<int8_t> outputGm_;
    TQue<TPosition::VECIN, 1> residentQueue_;
    TQue<TPosition::VECIN, 2> fullQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> residentPadded_;
    TBuf<TPosition::VECCALC> residentPacked_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    TPipe* pipe_{nullptr};
};

}  // namespace GreaterKernel

extern "C" __global__ __aicore__ void greater(
    GM_ADDR self, GM_ADDR other, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(GREATER_KEY_V2_WHOLE_LINEAR)) {
        KERNEL_TASK_TYPE(GREATER_KEY_V2_WHOLE_LINEAR, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_MEMBER(GreaterTilingData, totalElements, totalElements, tiling);
        AscendC::InitSocState();
        GreaterKernel::KernelWholeLinear<DTYPE_SELF> op;
        op.Process(self, other, output, totalElements);
        return;
    }
    if (!TILING_KEY_IS(GREATER_KEY_V2)) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    // 紧凑常驻行仅由 INT32 host 分支产生。用编译期分流避免把该大路径实例化进
    // 其他 dtype 的 kernel object，保持未命中该调度的二进制布局与基线一致。
    if constexpr (std::is_same_v<DTYPE_SELF, int32_t>) {
        if (tilingData.scheduleMode == GreaterKernel::SCHEDULE_RESIDENT_PITCHED) {
            KERNEL_TASK_TYPE(GREATER_KEY_V2, KERNEL_TYPE_AIV_ONLY);
            AscendC::TPipe pipe;
            if (tilingData.broadcastMask == GreaterKernel::BROADCAST_SELF) {
                GreaterKernel::KernelResidentPitchedRow<true> op;
                op.Init(self, other, output, &tilingData, &pipe);
                op.Process();
            } else {
                GreaterKernel::KernelResidentPitchedRow<false> op;
                op.Init(self, other, output, &tilingData, &pipe);
                op.Process();
            }
            return;
        }
        if (tilingData.scheduleMode == GreaterKernel::SCHEDULE_RESIDENT_PACKED) {
            KERNEL_TASK_TYPE(GREATER_KEY_V2, KERNEL_TYPE_AIV_ONLY);
            AscendC::TPipe pipe;
            if (tilingData.broadcastMask == GreaterKernel::BROADCAST_SELF) {
                GreaterKernel::KernelResidentPackedRow<true> op;
                op.Init(self, other, output, &tilingData, &pipe);
                op.Process();
            } else {
                GreaterKernel::KernelResidentPackedRow<false> op;
                op.Init(self, other, output, &tilingData, &pipe);
                op.Process();
            }
            return;
        }
    }

    if (tilingData.scheduleMode == GreaterKernel::SCHEDULE_LINEAR_SPAN) {
        KERNEL_TASK_TYPE(GREATER_KEY_V2, KERNEL_TYPE_AIV_ONLY);
        if (tilingData.activeCores == 1 &&
            tilingData.totalElements <= tilingData.tileElements) {
            AscendC::InitSocState();
            GreaterKernel::KernelWholeLinear<DTYPE_SELF> op;
            op.Process(self, other, output, tilingData.totalElements);
        } else {
            AscendC::TPipe pipe;
            if (tilingData.linearBufferNum == 2) {
                GreaterKernel::KernelLinearSpan<DTYPE_SELF, 2> op;
                op.Init(self, other, output, tilingData.tileElements,
                    tilingData.outputCacheEnabled != 0, &pipe);
                op.Process(tilingData.totalElements, tilingData.activeCores);
            } else {
                GreaterKernel::KernelLinearSpan<DTYPE_SELF, 1> op;
                op.Init(self, other, output, tilingData.tileElements,
                    tilingData.outputCacheEnabled != 0, &pipe);
                op.Process(tilingData.totalElements, tilingData.activeCores);
            }
        }
    } else if (tilingData.scheduleMode == GreaterKernel::SCHEDULE_RESIDENT_ROW) {
        KERNEL_TASK_TYPE(GREATER_KEY_V2, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        if (tilingData.broadcastMask == GreaterKernel::BROADCAST_SELF) {
            GreaterKernel::KernelResidentRow<DTYPE_SELF, true> op;
            op.Init(self, other, output, &tilingData, &pipe);
            op.Process();
        } else {
            GreaterKernel::KernelResidentRow<DTYPE_SELF, false> op;
            op.Init(self, other, output, &tilingData, &pipe);
            op.Process();
        }
    } else if (tilingData.scheduleMode == GreaterKernel::SCHEDULE_AFFINE_ROW) {
        KERNEL_TASK_TYPE(GREATER_KEY_V2, KERNEL_TYPE_AIV_ONLY);
        AscendC::TPipe pipe;
        GreaterKernel::KernelAffineRow<DTYPE_SELF> op;
        op.Init(self, other, output, tilingData.dims, tilingData.selfStrides,
            tilingData.otherStrides, tilingData.collapsedRank, tilingData.inner,
            tilingData.tileElements, tilingData.outputCacheEnabled != 0, &pipe);
        op.Process(tilingData.rowCount, tilingData.activeCores);
    }
}
