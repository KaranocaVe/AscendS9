#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

#include <cstdint>

namespace optiling {
constexpr uint64_t CONCAT_KEY_FLAT = 1;
constexpr uint64_t CONCAT_KEY_ROW = 2;
constexpr uint64_t CONCAT_KEY_GRID = 3;
constexpr uint64_t CONCAT_KEY_INPUT_LPT = 4;
constexpr uint64_t CONCAT_KEY_GRID_SPAN = 5;
constexpr uint64_t CONCAT_KEY_PREPARED_FLAT = 6;
constexpr uint64_t CONCAT_KEY_ROW_GATHER = 7;
constexpr uint64_t CONCAT_KEY_GRID_SPAN_SINGLE_BUFFER = 8;
constexpr uint64_t CONCAT_KEY_PREPARED_FLAT_TWO_COMMANDS = 9;
constexpr uint64_t CONCAT_KEY_PREPARED_FLAT_TWO_COMMANDS_STATIC = 10;
constexpr uint64_t CONCAT_KEY_PREPARED_RECT = 11;
constexpr uint64_t CONCAT_KEY_PREPARED_RECT_STATIC = 12;
// 行源数据常驻 UB，再按行重排写回的 Gather 路径。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_RESIDENT = 13;
// 将各输入段直接落到输出 UB 行布局的实验路径。
constexpr uint64_t CONCAT_KEY_ROW_DIRECT = 14;
// 将一个 Gather 批拆为两个输出槽，以重叠 Vector 重排与前一槽 MTE3 写回。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_OUTPUT_PINGPONG = 15;
// 使用双 source 槽预取下一批输入，使 MTE2 与当前批 Gather 重叠。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_SOURCE_PINGPONG = 16;
// 合并多行 source 搬入，减少高输入数窄段场景中的 MTE2 指令发射。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_MERGED_BATCH = 17;
// source 搬入批次与输出/Gather 批次解耦，避免扩大跨行 MTE3 写回。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_STAGED_SOURCE = 18;
// 将完整 32B 输出块映射到相位对齐 source stripe，边界块经 UB stash 补齐。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_BLOCK = 19;
// source 按行主序写入 UB；较大 source stage 内复用同一张 Gather offset 表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_ROW_MAJOR_STAGED = 20;
// 以较大批次连续搬入并完成 Gather，再分小批写回，避免长跨行 MTE3 写回。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_SPLIT_OUTPUT = 21;
// 连续 source stage 常驻两张子批 offset 表，避免每个 source 子批重复建表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_CACHED_OFFSETS = 22;
// 连续 source stage 使用一张完整行数的 offset 表，供全部输出子批按行切片复用。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_FULL_STAGE_OFFSETS = 23;
// source stage 的尾输出子批仅生成实际行数的 offset，避免无效行的 Vector 建表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_COMPACT_STAGE_OFFSETS = 24;
// 以 MicroAPI 寄存器索引和逐行 stride 生成 RowGather 地址，避免常驻整张 offset 表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_MICRO = 25;
// 各输入连续搬入 UB 后直接经二维 MTE3 写到目标行，避免 Gather 和整张 offset 表。
constexpr uint64_t CONCAT_KEY_ROW_DIRECT_MTE = 26;
// 连续 source stage 复用一个输出批 offset 表，并按子批行偏移滚动更新。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_ROLLING_STAGE_OFFSETS = 27;
// 连续 source stage 使用一行增量表批量平移/恢复 offset，避免逐行逐段标量更新。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_DELTA_STAGE_OFFSETS = 28;
// 行并行的大工作集流式输入不写入 L2；与普通 ROW 共用同一份紧凑 TilingData。
constexpr uint64_t CONCAT_KEY_ROW_STREAM_INPUT_L2_BYPASS = 29;
// 连续搬入配合相位 padding 后，以 masked Copy 按行重排，避免整张 Gather offset 表。
constexpr uint64_t CONCAT_KEY_ROW_MASKED_COPY = 30;
// 将一个连续 source tile 分为两个互补分区，在不复制整块 UB 的前提下交错 MTE2、Gather 与 MTE3。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_PARTITION_PIPELINE = 31;
// 各核复用 workspace 中按动态分组生成的 Gather offset seed，避免重复建首行表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_SHARED_SEED = 33;
// 保持每核连续行块，仅交错物理核与逻辑行块的映射，用于改善高扇入搬运的 bank 分布。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_CORE_SWIZZLE = 34;
// 各物理核循环错开独立 Gather group 的执行起点，打散高扇入短段的跨核锁步访问。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_GROUP_PHASE = 35;
// 输入连续带和行块构成二维任务；每核只重排一个窄带，降低高扇入场景的连续 MTE2 发射量。
constexpr uint64_t CONCAT_KEY_ROW_BAND_GATHER = 36;
// 两段对齐输入 wavefront：dense Gather + UB Copy 汇入完整 output，保持单次整行 MTE3。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_WAVEFRONT = 38;
// 两段连续输入分别用跨行距 Gather 直接写入完整 output，去掉 partial output 的 UB Copy。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_DIRECT_WAVEFRONT = 39;
// 两个同尺寸 source 批平面经单条二维 DMA 搬入，并复用一张 Gather offset 表。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_TWO_PLANE = 40;
// 每个 Gather group 在既有 UB 最大容量内独立选择批行数，避免窄尾组空置资源。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_GROUP_BATCH = 42;
// Gather 的 offset 表与 output 复用 VECOUT，仅常驻 base/stride 两行 VECCALC seed。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_OFFSET_OUTPUT_ALIAS = 43;
// 混合 AIC:AIV=1:2 的启动语义探针：AIV 仍使用 RowGather，AIC 暂不领取输出行。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_MIX_AIV_PROBE = 44;
// 双 source 槽预取与 output/offset 别名组合，保持原 RowGather 批行数并重叠 MTE2。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_SOURCE_PINGPONG_OFFSET_OUTPUT_ALIAS = 45;
// 相邻输入组共享一张跨行 Gather 输出矩阵，减少短段的输出写回与 offset 表常驻。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_PAIR = 46;
// 使用已有 source 尾部交替预取下一批后缀，隐藏高扇入短段的部分 MTE2 延迟。
constexpr uint64_t CONCAT_KEY_ROW_GATHER_SUFFIX_PREFETCH = 47;
// 静态 prepared-rect 的一次性输入流版本；命令头最高位携带 L2 提示。
constexpr uint64_t CONCAT_KEY_PREPARED_RECT_STATIC_STREAM_INPUT_L2_BYPASS = 48;
// Host 预计算动态输入行宽，避免每个 AIV 重复解析相同的 ListTensorDesc。
constexpr uint64_t CONCAT_KEY_ROW_PRECOMPUTED_BYTES = 49;
// INT32 ROW 在完整 UB tile 能严格减少二维 DMA 批次时使用单槽搬运。
constexpr uint64_t CONCAT_KEY_ROW_PRECOMPUTED_SINGLE_BUFFER = 55;
// 小型对齐 ROW 将输入段线性搬入 UB，再以 Gather 组装完整输出行后一次写回。
constexpr uint64_t CONCAT_KEY_ROW_PACKED_UB = 65;
constexpr uint32_t CONCAT_MAX_FLAT_CORES = 64;
// 任务表超出容量时由 Host 回退到原始 FLAT 路径。
constexpr uint32_t CONCAT_MAX_PREPARED_FLAT_TASKS = 320;
constexpr uint32_t CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE = 2;
constexpr uint32_t CONCAT_MAX_PREPARED_FLAT_COMMANDS =
    CONCAT_MAX_FLAT_CORES * CONCAT_PREPARED_FLAT_COMMANDS_PER_CORE;
constexpr uint32_t CONCAT_MAX_GRID_COLS = 64;
constexpr uint32_t CONCAT_PREPARED_RECT_COMMANDS_PER_CORE = 4;
constexpr uint32_t CONCAT_MAX_PREPARED_RECT_COMMANDS =
    CONCAT_MAX_GRID_COLS * CONCAT_PREPARED_RECT_COMMANDS_PER_CORE;
constexpr uint32_t CONCAT_PREPARED_RECT_WORDS_PER_COMMAND = 4;
constexpr uint32_t CONCAT_MAX_PREPARED_RECT_WORDS =
    CONCAT_MAX_PREPARED_RECT_COMMANDS * CONCAT_PREPARED_RECT_WORDS_PER_COMMAND;
constexpr uint32_t CONCAT_MAX_GRID_INPUTS = 256;
constexpr uint32_t CONCAT_GATHER_PLAN_COUNT = 2;
constexpr uint32_t CONCAT_MAX_GATHER_GROUPS =
    CONCAT_GATHER_PLAN_COUNT * CONCAT_MAX_GRID_INPUTS;
constexpr uint32_t CONCAT_MAX_BAND_GATHER_TASKS = CONCAT_MAX_FLAT_CORES;
constexpr uint32_t CONCAT_MAX_PAIR_GATHER_TASKS = CONCAT_MAX_FLAT_CORES;
constexpr uint32_t CONCAT_MAX_LPT_CORES = 64;
constexpr uint32_t CONCAT_MAX_LPT_TASKS = 256;

BEGIN_TILING_DATA_DEF(ConcatFlatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
  TILING_DATA_FIELD_DEF(uint64_t, coreChunkBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_FLAT_CORES, coreStartInput);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_FLAT_CORES, coreStartInputOffset);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatBandGatherTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GRID_INPUTS, inputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_BAND_GATHER_TASKS, taskInputStart);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_BAND_GATHER_TASKS, taskInputEnd);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskOutputStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskGroupElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskSourceElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskBatchRows);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskSourceBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskOutputBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_BAND_GATHER_TASKS, taskOffsetBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_BAND_GATHER_TASKS, taskRowStart);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_BAND_GATHER_TASKS, taskRowCount);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatPairGatherTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GRID_INPUTS, inputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskInputStart);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskInputSplit);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskInputEnd);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskOutputStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskGroup0Elements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskGroup1Elements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskPairElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskSourceElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskBatchRows);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskSourceBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskOutputBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskOffsetBufferBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskRowStart);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PAIR_GATHER_TASKS, taskRowCount);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatPreparedRectTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PREPARED_RECT_WORDS, commandData);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatPreparedFlatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint64_t, coreChunkBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_FLAT_CORES, coreOutputStart);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_FLAT_CORES, coreTaskEnd);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_PREPARED_FLAT_TASKS, taskInput);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PREPARED_FLAT_TASKS, taskInputOffset);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PREPARED_FLAT_TASKS, taskBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatPreparedFlatTwoCommandsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_FLAT_CORES, coreOutputStart);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS, commandInput);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS, commandInputOffset);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS, commandOutputOffset);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_PREPARED_FLAT_COMMANDS, commandBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatRowGatherTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, highCoreCount);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore0);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore1);
  TILING_DATA_FIELD_DEF(uint32_t, batchRows0);
  TILING_DATA_FIELD_DEF(uint32_t, batchRows1);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBatchRows0);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBatchRows1);
  TILING_DATA_FIELD_DEF(uint32_t, groupCount0);
  TILING_DATA_FIELD_DEF(uint32_t, groupCount1);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, outputBufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, outputBufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, offsetBufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, offsetBufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, sourcePart0BufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, sourcePart0BufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, sourcePart1BufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, sourcePart1BufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, outputPart0BufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, outputPart0BufferBytes1);
  TILING_DATA_FIELD_DEF(uint32_t, outputPart1BufferBytes0);
  TILING_DATA_FIELD_DEF(uint32_t, outputPart1BufferBytes1);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GRID_INPUTS, inputRowElements);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_GATHER_GROUPS, groupInputEnd);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_GATHER_GROUPS, groupInputSplit);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_GATHER_GROUPS, groupInputSplit2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupSourceElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupBatchRows);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_GATHER_GROUPS, groupSuffixInputStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupSuffixOutputStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupSuffixElements);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GATHER_GROUPS, groupSuffixSourceElements);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatGridSpanTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, rowGroups);
  TILING_DATA_FIELD_DEF(uint32_t, colGroups);
  TILING_DATA_FIELD_DEF(uint32_t, reserved);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF(uint64_t, colChunkBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_GRID_COLS, coreSpan);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_GRID_INPUTS, inputRowBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatRowTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatRowPrecomputedTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GRID_INPUTS, inputRowBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatPackedRowTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, outputRowElements);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBufferBytes);
  TILING_DATA_FIELD_DEF(uint32_t, outputBufferBytes);
  TILING_DATA_FIELD_DEF(uint32_t, offsetBufferBytes);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_GRID_INPUTS, inputRowElements);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatGridTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, rowGroups);
  TILING_DATA_FIELD_DEF(uint32_t, colGroups);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF(uint64_t, colChunkBytes);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatInputLptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_LPT_CORES, coreTaskStart);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_LPT_CORES, coreTaskCount);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_LPT_TASKS, taskInput);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_LPT_TASKS, taskCore);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_LPT_TASKS, taskRowStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_LPT_TASKS, taskRowCount);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_LPT_TASKS, taskRowBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, CONCAT_MAX_LPT_TASKS, taskOutputCol);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ConcatDefaultTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat_1, ConcatFlatTilingData)
REGISTER_TILING_DATA_CLASS(Concat_2, ConcatRowTilingData)
REGISTER_TILING_DATA_CLASS(Concat_3, ConcatGridTilingData)
REGISTER_TILING_DATA_CLASS(Concat_4, ConcatInputLptTilingData)
REGISTER_TILING_DATA_CLASS(Concat_5, ConcatGridSpanTilingData)
REGISTER_TILING_DATA_CLASS(Concat_6, ConcatPreparedFlatTilingData)
REGISTER_TILING_DATA_CLASS(Concat_7, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_8, ConcatGridSpanTilingData)
REGISTER_TILING_DATA_CLASS(Concat_9, ConcatPreparedFlatTwoCommandsTilingData)
REGISTER_TILING_DATA_CLASS(Concat_10, ConcatPreparedFlatTwoCommandsTilingData)
REGISTER_TILING_DATA_CLASS(Concat_11, ConcatPreparedRectTilingData)
REGISTER_TILING_DATA_CLASS(Concat_12, ConcatPreparedRectTilingData)
REGISTER_TILING_DATA_CLASS(Concat_13, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_14, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_15, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_16, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_17, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_18, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_19, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_20, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_21, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_22, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_23, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_24, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_25, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_26, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_27, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_28, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_29, ConcatRowTilingData)
REGISTER_TILING_DATA_CLASS(Concat_30, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_31, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_33, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_34, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_35, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_36, ConcatBandGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_38, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_39, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_40, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_42, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_43, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_44, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_45, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_46, ConcatPairGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_47, ConcatRowGatherTilingData)
REGISTER_TILING_DATA_CLASS(Concat_49, ConcatRowPrecomputedTilingData)
REGISTER_TILING_DATA_CLASS(Concat_55, ConcatRowPrecomputedTilingData)
REGISTER_TILING_DATA_CLASS(Concat_65, ConcatPackedRowTilingData)
// OPC 仍会查询无 key 的基类注册；实际运行按对应的 Concat_<key> 解析紧凑结构。
REGISTER_TILING_DATA_CLASS(Concat, ConcatDefaultTilingData)
}

#endif
