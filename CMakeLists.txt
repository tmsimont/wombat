cmake_minimum_required (VERSION 3.0)
project (wombat)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
 
include_directories ("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)
add_subdirectory(tests)

