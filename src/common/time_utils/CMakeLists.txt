add_library(time_utils time_utils.cpp)
target_include_directories(time_utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(time_utils PRIVATE my_default_flags)