cmake_minimum_required(VERSION 4.2)

add_executable(calibration main.cpp)

set_target_properties(
    calibration
    PROPERTIES
        CXX_STANDARD
        17
        CXX_STANDARD_REQUIRED
        YES
        CXX_EXTENSIONS
        NO)

# OpenCV's installed configuration still uses the legacy FindCUDA module
cmake_policy(PUSH)
cmake_policy(SET CMP0146 OLD)

find_package(OpenCV 5 REQUIRED COMPONENTS core calib highgui imgproc objdetect videoio)

cmake_policy(POP)

target_include_directories(calibration SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(calibration PRIVATE ${OpenCV_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(
        calibration
        PRIVATE
            -Wall
            -Wextra
            -pedantic-errors
            -Wconversion
            -Wsign-conversion
            -Wshadow)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
        calibration
        PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-funroll-loops>)
endif()
