#include "register/tilingdata_base.h"
#include "../op_kernel/square_sum_v1_tiling_key.h"

namespace optiling {
// CANN ACLNN 的通用 ND 张量能力上限为 8 维，覆盖规格中省略号表示的前导维。
constexpr uint32_t SQUARE_SUM_MAX_RANK = 8;

BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, inputSize);
  TILING_DATA_FIELD_DEF(uint64_t, outputSize);
  TILING_DATA_FIELD_DEF(uint64_t, dim0);
  TILING_DATA_FIELD_DEF(uint64_t, dim1);
  TILING_DATA_FIELD_DEF(uint64_t, dim2);
  TILING_DATA_FIELD_DEF(uint64_t, reduceSize);
  TILING_DATA_FIELD_DEF(uint64_t, workspaceStride);
  TILING_DATA_FIELD_DEF(uint32_t, topology);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, axisMask);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, tilePrimary);
  TILING_DATA_FIELD_DEF(uint32_t, alignedInner);
  TILING_DATA_FIELD_DEF(uint32_t, sharedTmpBytes);
  TILING_DATA_FIELD_DEF(uint32_t, elementSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputBufferNum);
  TILING_DATA_FIELD_DEF(uint32_t, raStaggerGroups);
  TILING_DATA_FIELD_DEF(uint32_t, compactAr);
  TILING_DATA_FIELD_DEF(uint32_t, compactRar);
  TILING_DATA_FIELD_DEF(uint32_t, compactAra);
  TILING_DATA_FIELD_DEF(uint32_t, bypassL2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, SQUARE_SUM_MAX_RANK, shape);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, SQUARE_SUM_MAX_RANK, stride);
  TILING_DATA_FIELD_DEF(uint32_t, rarScheduleMode);
  TILING_DATA_FIELD_DEF(uint32_t, compactRa);
  TILING_DATA_FIELD_DEF(uint32_t, compactRaPhaseRows);
  TILING_DATA_FIELD_DEF(uint32_t, compactRaPhaseAlignedSplit);
  TILING_DATA_FIELD_DEF(uint32_t, compactRaPhaseSegments);
  TILING_DATA_FIELD_DEF(uint32_t, compactRaPhaseCheckpointGroups);
  TILING_DATA_FIELD_DEF(uint32_t, compactRaPhaseScheduleMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}
