add_library(arg_parser arg_parser.cpp)
target_include_directories(arg_parser PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(arg_parser PUBLIC CLI11::CLI11)
target_link_libraries(arg_parser PRIVATE my_default_flags)