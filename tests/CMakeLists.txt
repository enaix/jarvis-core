file(GLOB_RECURSE TESTS_SRC ./*.h ./*.cpp) # relative to tests/

add_executable(tests ${TESTS_SRC})

target_link_libraries(tests INTERFACE common) # include lib/
