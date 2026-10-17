# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

# 仅调整设备端局部指令选择，不改变 tiling、DMA 或算子接口。
add_ops_compile_options(Transpose OPTIONS --cce-peephole-opt)

add_kernels_compile()
