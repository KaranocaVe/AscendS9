# AscendS9

A source-only collection of five Ascend C custom operators assembled from the supplied ZIP archives.

## Operators

- `Concat`
- `Greater`
- `IndexAdd`
- `SquareSumV1`
- `Transpose`

Each operator keeps its own `op_host/` and `op_kernel/` source and CMake files.

## Build

The operator CMake files call build functions supplied by an Ascend CANN/AscendC custom-operator environment (including `opbuild` and `add_kernels_compile`). The archives did not include a root build project, SDK version information, or tests; integrate these operator directories into a compatible CANN custom-operator project.

## Included files

This repository contains the `.cpp`, `.h`, and `CMakeLists.txt` source/build files. The `custom_opp_euleros_aarch64.run` prebuilt installer bundled in each archive is omitted.

## License

The supplied files contained no license or copyright notice, so this repository adds no license. Public availability does not grant additional rights to redistribute, modify, or use the source; confirm the applicable permissions before reuse.
