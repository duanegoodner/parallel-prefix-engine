add_executable(
    test_program_args
    test_program_args.cpp
    )
target_include_directories(
    test_program_args PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include

    )
target_link_libraries(
    test_program_args
    PRIVATE
    GTest::gtest_main
    program_args
    )
target_link_libraries(test_program_args PRIVATE my_default_flags)

gtest_discover_tests(test_program_args)
