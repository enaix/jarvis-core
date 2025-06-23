file(GLOB_RECURSE TESTS_SRC ./*.h ./*.cpp) # relative to tests/

add_executable(tests ${TESTS_SRC})

target_link_libraries(tests common) # include lib/