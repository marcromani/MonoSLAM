# MonoSLAM

A concise and thoroughly documented C++ implementation of the seminal work by [Davison et *al.* 2007](https://www.doc.ic.ac.uk/~ajd/Publications/davison_etal_pami2007.pdf), with parallel feature integration. The project is mostly compliant with ISO C++17, and it compiles against OpenCV 5.

It runs at 30 Hz on commodity hardware, check the [sample videos](https://www.youtube.com/watch?v=zg3O0ZN1_rI&list=PLQjxkIPDI8hgJj4gUFwuYeaorkd8JWDpo).

## How to build

Configure the project with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

To enable debug symbols use:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Build with:

```bash
cmake --build build --parallel
```

To build only one target use:

`cmake --build build --target calibration --parallel` (Calibration)

`cmake --build build --target monoslam --parallel` (MonoSLAM)

## Developer dependencies

Source code formatting depends on:
```
clang-format 23.1.4
cmakefmt 2.3.1
```

Usage:

```bash
clang-format -i Calibration/*.cpp MonoSLAM/*.cpp MonoSLAM/*.h MonoSLAM/filter/*.cpp MonoSLAM/filter/*.h
cmakefmt -i .
```
