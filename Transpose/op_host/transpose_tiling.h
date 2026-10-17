#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"
#include "../op_kernel/transpose_tiling_key.h"

#include <cstdint>

namespace optiling {
constexpr uint32_t TRANSPOSE_MAX_RANK = 8;

constexpr uint64_t TRANSPOSE_KEY_COPY = 1;
constexpr uint64_t TRANSPOSE_KEY_BLOCK_SWAP = 2;
constexpr uint64_t TRANSPOSE_KEY_GENERIC_AFFINE = 3;
// 单矩阵 Strided 有独立的热循环；使用单独 tiling key 让设备侧在编译期选择该路径。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_SINGLE_MATRIX = 4;
// Dense 矩阵热循环不依赖其他 BlockSwap 微内核；使用独立 key 去除设备侧五路分派。
constexpr uint64_t TRANSPOSE_KEY_DENSE_MATRIX = 5;
// 多矩阵 Strided 热循环使用独立 key，避免进入 BlockSwap 后再作微内核分派。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_MATRIX = 6;
// 非退化 Strided 使用独立 key，使外层模板分派也在编译期完成。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_NONFIXED_AXIS = 9;
// 满足完整条带容量约束的单矩阵路径使用独立 TBuf kernel，避免携带队列微内核布局。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_SINGLE_TBUF = 10;
// 固定轴多矩阵 Strided 使用独立入口，隔离其长 MTE2 热循环的设备代码布局。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_FIXED_MATRIX = 11;
// 固定轴多矩阵的紧凑专用 kernel，保持队列数据流但隔离无关微内核代码。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_FIXED_COMPACT = 12;
// FP32 相位对齐搬入后在 UB 内压缩，再执行 VA 转置的专用入口。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_FIXED_PHASE_COMPACT = 13;
// FP32 完整相位周期规整后，以逐行 mask 进行一次串流 GatherMask 的专用入口。
constexpr uint64_t TRANSPOSE_KEY_STRIDED_FIXED_PHASE_MAJOR_COMPACT = 14;

enum class TransposePlanKind : uint32_t {
    COPY = 1,
    BLOCK_SWAP = 2,
    GENERIC_AFFINE = 3,
};

enum class TransposeBlockSwapKernel : uint32_t {
    DENSE_MATRIX = 1,
    STRIDED_MATRIX = 2,
    RECORD_TRANS_DATA = 3,
    RECORD_GATHER = 4,
    STRIDED_SINGLE_MATRIX = 5,
};

BEGIN_TILING_DATA_DEF(TransposeTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalElements);
  TILING_DATA_FIELD_DEF(uint64_t, batch);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, rows);
  TILING_DATA_FIELD_DEF(uint64_t, cols);
  TILING_DATA_FIELD_DEF(uint64_t, inner);
  TILING_DATA_FIELD_DEF(uint64_t, record);
  TILING_DATA_FIELD_DEF(uint64_t, tileRows);
  TILING_DATA_FIELD_DEF(uint64_t, tileCols);
  TILING_DATA_FIELD_DEF(uint64_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, panelRows);
  TILING_DATA_FIELD_DEF(uint64_t, ubSize);
  TILING_DATA_FIELD_DEF(uint32_t, planKind);
  TILING_DATA_FIELD_DEF(uint32_t, microKernel);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, mergedRank);
  TILING_DATA_FIELD_DEF(uint32_t, elementSize);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, bufferNum);
  // 每个逻辑任务连续处理的列 tile 数；1 表示保持单列 tile 任务。
  TILING_DATA_FIELD_DEF(uint32_t, colTilesPerTask);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, TRANSPOSE_MAX_RANK, shape);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, TRANSPOSE_MAX_RANK, inputStride);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, TRANSPOSE_MAX_RANK, outputStride);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, TRANSPOSE_MAX_RANK, perm);
  TILING_DATA_FIELD_DEF(uint64_t, panelCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}

#endif
