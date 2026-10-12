cmake_minimum_required(VERSION 4.2)

project(Calibration LANGUAGES CXX)

# OpenCV's installed configuration still uses the legacy FindCUDA module
cmake_policy(PUSH)
cmake_policy(SET CMP0146 OLD)

find_package(OpenCV 5 REQUIRED COMPONENTS core calib highgui imgproc objdetect videoio)

cmake_policy(POP)

add_executable(calibration calibration.cpp)

set_target_properties(calibration PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(calibration SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibration PRIVATE ${OpenCV_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(calibration PRIVATE
        -Wall
        -Wextra
        -pedantic-errors
        -Wconversion
        -Wsign-conversion
        -Wshadow
    )
endif()
