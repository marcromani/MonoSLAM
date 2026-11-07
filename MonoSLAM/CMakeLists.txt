# The sanitizers, in debug builds. The calibration program goes without: its window, which OpenCV
# never frees, would be reported as a leak, and only monoslam suppresses that
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    add_compile_definitions($<$<CONFIG:Debug,RelWithDebInfo>:MONOSLAM_SANITIZERS_ENABLED>)
    add_compile_options($<$<CONFIG:Debug,RelWithDebInfo>:-fsanitize=address,undefined>)
    add_link_options($<$<CONFIG:Debug,RelWithDebInfo>:-fsanitize=address,undefined>)
endif()

# Everything but the programs, which share it
add_library(
    monoslam_core
    STATIC
        camera.cpp
        ekf.cpp
        feature.cpp
        frame_source.cpp
        initialization.cpp
        inverse_depth.cpp
        mapper.cpp
        relocalization.cpp
        state_stream.cpp
        tracking.cpp)

target_include_directories(monoslam_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(monoslam_core PUBLIC ${OpenCV_LIBS})

add_executable(analyze_pattern tools/analyze_pattern.cpp)
target_link_libraries(analyze_pattern PRIVATE monoslam_core)

add_executable(monoslam main.cpp display.cpp)
target_link_libraries(monoslam PRIVATE monoslam_core)

add_executable(record_sequence tools/record_sequence.cpp)
target_link_libraries(record_sequence PRIVATE monoslam_core)

if(BUILD_TESTING)
    add_executable(monoslam_tests tests/system_tests.cpp)
    target_link_libraries(monoslam_tests PRIVATE monoslam_core)

    add_test(NAME monoslam_tests COMMAND monoslam_tests)
    set_tests_properties(monoslam_tests PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endif()
