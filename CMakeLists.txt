cmake_minimum_required(VERSION 3.27)
project(jsc) # jarvis-core

set(CMAKE_CXX_STANDARD 20)

# Set debug options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -Werror=return-type")
endif()


# Common library

file(GLOB_RECURSE COMMON_FILES lib/*.h lib/*.cpp)

add_library(common STATIC ${COMMON_FILES})
target_include_directories(common PUBLIC lib) # Include common headers
set_property(TARGET common PROPERTY LINKER_LANGUAGE CXX)


# Subdirectories

add_subdirectory(tests build/tests/)