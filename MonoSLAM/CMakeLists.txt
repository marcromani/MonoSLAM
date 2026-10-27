cmake_minimum_required(VERSION 4.2)

add_executable(
    monoslam
    camera.cpp
    feature.cpp
    filter/ekf.cpp
    geometry.cpp
    main.cpp
    matcher.cpp
    quaternion.cpp
    system_observation_model.cpp
    system_process_model.cpp
    util.cpp)

set_target_properties(
    monoslam
    PROPERTIES
        CXX_STANDARD
        17
        CXX_STANDARD_REQUIRED
        YES
        CXX_EXTENSIONS
        NO)

target_include_directories(monoslam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# OpenCV's installed configuration still uses the legacy FindCUDA module
cmake_policy(PUSH)
cmake_policy(SET CMP0146 OLD)

find_package(OpenCV 5 REQUIRED COMPONENTS core calib highgui imgproc objdetect videoio)

cmake_policy(POP)

target_include_directories(monoslam SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(monoslam PRIVATE ${OpenCV_LIBS})

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(
        monoslam
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
        monoslam
        PRIVATE
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-funroll-loops>)
endif()
