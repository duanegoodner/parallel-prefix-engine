add_library(kernel_array kernel_array.cpp)

target_include_directories(kernel_array
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    kernel_array PRIVATE
    my_default_flags)


