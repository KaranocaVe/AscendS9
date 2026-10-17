#ifndef GREATER_TILING_H
#define GREATER_TILING_H

#include "register/tilingdata_base.h"

#include <cstdint>

namespace optiling {

constexpr uint32_t GREATER_MAX_RANK = 25;
constexpr uint64_t GREATER_KEY_V2 = 6;
constexpr uint64_t GREATER_KEY_V2_WHOLE_LINEAR = 7;

// 调度模式由 host 依据广播拓扑和资源约束统一选择。
constexpr uint32_t GREATER_SCHEDULE_LINEAR_SPAN = 0;
constexpr uint32_t GREATER_SCHEDULE_RESIDENT_ROW = 1;
constexpr uint32_t GREATER_SCHEDULE_AFFINE_ROW = 2;
// INT32 行广播在整相位包可连续搬入且 UB 足够时使用紧凑包路径。
constexpr uint32_t GREATER_SCHEDULE_RESIDENT_PACKED = 3;
// INT32 行广播按 32B 行距搬入，比较阶段用 BinaryRepeatParams 复用常驻行。
constexpr uint32_t GREATER_SCHEDULE_RESIDENT_PITCHED = 4;

// 行广播的核间分区方式。
constexpr uint32_t GREATER_PARTITION_GROUP_MAJOR = 0;
constexpr uint32_t GREATER_PARTITION_ROW_MAJOR = 1;

BEGIN_TILING_DATA_DEF(GreaterTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, version);
  TILING_DATA_FIELD_DEF(uint32_t, scheduleMode);
  TILING_DATA_FIELD_DEF(uint32_t, partitionMode);
  TILING_DATA_FIELD_DEF(uint32_t, broadcastMask);
  TILING_DATA_FIELD_DEF(uint32_t, tileElements);
  TILING_DATA_FIELD_DEF(uint32_t, rowBatch);
  TILING_DATA_FIELD_DEF(uint32_t, linearBufferNum);
  TILING_DATA_FIELD_DEF(uint32_t, activeCores);
  TILING_DATA_FIELD_DEF(uint32_t, collapsedRank);
  TILING_DATA_FIELD_DEF(uint64_t, totalElements);
  TILING_DATA_FIELD_DEF(uint64_t, rowCount);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, repeat);
  TILING_DATA_FIELD_DEF(uint64_t, inner);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, GREATER_MAX_RANK, dims);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, GREATER_MAX_RANK, selfStrides);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, GREATER_MAX_RANK, otherStrides);
  // 输出仅在完整逻辑工作集可容纳于运行时 L2 时保留默认缓存策略。
  TILING_DATA_FIELD_DEF(uint32_t, outputCacheEnabled);
  // INT32 行路径使用的 full 包行数，由 UB 账本、相位周期和 repeat 整除关系决定。
  TILING_DATA_FIELD_DEF(uint32_t, packedPacketRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
}

#endif
