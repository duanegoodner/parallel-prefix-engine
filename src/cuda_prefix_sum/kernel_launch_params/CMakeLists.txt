add_library(kernel_launch_params kernel_launch_params.cpp)
target_include_directories(kernel_launch_params PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_include_directories(kernel_launch_params
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(kernel_launch_params PRIVATE kernel_array my_default_flags)

