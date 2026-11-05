cmake_minimum_required(VERSION 4.2)

project(MonoSLAM LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV's installed configuration still uses the legacy FindCUDA module
cmake_policy(PUSH)
cmake_policy(SET CMP0146 OLD)

find_package(
    OpenCV
    5
    REQUIRED
    COMPONENTS
        calib
        core
        features
        geometry
        highgui
        imgcodecs
        imgproc
        objdetect
        video
        videoio)

cmake_policy(POP)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    add_compile_options(-Wall -Wextra -pedantic-errors -Wconversion -Wsign-conversion -Wshadow)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<CONFIG:Release>:-march=native> $<$<CONFIG:Release>:-funroll-loops>)
endif()

add_subdirectory(Calibration)
add_subdirectory(MonoSLAM)
