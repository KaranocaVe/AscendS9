
#include "register/tilingdata_base.h"
#include "../op_kernel/index_add_tiling_key.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, outerSize);
  TILING_DATA_FIELD_DEF(uint64_t, axisSize);
  TILING_DATA_FIELD_DEF(uint64_t, innerSize);
  TILING_DATA_FIELD_DEF(uint64_t, indexSize);
  TILING_DATA_FIELD_DEF(uint64_t, innerTile);
  TILING_DATA_FIELD_DEF(uint64_t, outerTile);
  TILING_DATA_FIELD_DEF(uint64_t, outerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, strategy);
  TILING_DATA_FIELD_DEF(uint32_t, innerTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, outerTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, taskNum);
  TILING_DATA_FIELD_DEF(uint32_t, sourceChunk);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, elementSize);
  TILING_DATA_FIELD_DEF(uint32_t, coopLayerCap);
  // resident 混合切分仅由形状、UB 和核数派生；为 0 时保持原有等宽切分。
  TILING_DATA_FIELD_DEF(uint32_t, mixedInnerTile);
  TILING_DATA_FIELD_DEF(uint32_t, mixedInnerTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, mixedOuterCount);
  TILING_DATA_FIELD_DEF(uint32_t, residentBufferBytes);
  // resident 双槽源流水：为每个 source panel 预留两个可轮换槽位。
  TILING_DATA_FIELD_DEF(uint32_t, residentDoubleBuffer);
  // 窄 Gather 的 payload 仅在 UB 中扩展到完整 Vector repeat；0 表示保持逻辑轴宽。
  TILING_DATA_FIELD_DEF(uint32_t, gatherUbPaddedAxis);
  // 由 host 按当前 UB 预算确认可容纳的 16 行转置块 Gather 路径。
  TILING_DATA_FIELD_DEF(uint32_t, transposeBlockGather);
  // FP32 转置块路径每次暂存的 source 列数；0 表示未启用该路径。
  TILING_DATA_FIELD_DEF(uint32_t, transposeSourceChunk);
  // FP32 转置块路径每个 source chunk 可容纳的最大 occurrence layer 数。
  TILING_DATA_FIELD_DEF(uint32_t, transposeLayerCap);
  // FP32 payload 单次从 workspace 搬入的 occurrence 层数；小于上限时分组读取，
  // 以释放 UB 给 source chunk，同时仍保留完整 metadata 表与严格回退。
  TILING_DATA_FIELD_DEF(uint32_t, transposePayloadLayerCap);
  // FP32 转置块路径的 workspace offset 槽宽；仅允许 2B compact 或 4B 原表。
  TILING_DATA_FIELD_DEF(uint32_t, transposeOffsetBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
