cmake_minimum_required(VERSION 3.27)
project(jsc) # jarvis-core

set(CMAKE_CXX_STANDARD 17)

# Set debug options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -Werror=return-type -DALWAYS_THROW_ON_ERROR")
endif()

option(FORCE_OPTIMAL_STRUCTS "Force the compiler to produce optimial struct sizes" OFF)
option(ALWAYS_THROW_ON_ERROR "Always throw on error instead of returning false" ON) # Should be always enabled for alpha release
if (FORCE_OPTIMAL_STRUCTS)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPTIMAL_STRUCTS")
endif()

if (ALWAYS_THROW_ON_ERROR)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DALWAYS_THROW_ON_ERROR")
endif()

# Python bindings

set(DEV_MODULE Development.Module)
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind)

# Common library

set(COMMON_FILES lib/graph.h)

add_library(common INTERFACE ${COMMON_FILES})
target_include_directories(common INTERFACE lib) # Include common headers
set_property(TARGET common PROPERTY LINKER_LANGUAGE CXX)
nanobind_add_module(jsc_common ${COMMON_FILES} extra/binds.cpp)


# Subdirectories

include_directories(lib)
add_subdirectory(tests build/tests/)
