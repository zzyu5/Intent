cmake_minimum_required(VERSION 3.20)
project(IntentDSL LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("${MLIR_INCLUDE_DIRS}")
include_directories("${LLVM_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")
add_definitions("${LLVM_DEFINITIONS}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
