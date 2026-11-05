add_executable(calibration main.cpp)

target_link_libraries(calibration PRIVATE ${OpenCV_LIBS})
