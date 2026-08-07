################################################################################
# LAKSA Compiler
#
# MLIR dialects implementing the MYRTUS LAKSA compiler.
################################################################################

cmake_minimum_required(VERSION 3.25)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(laksa
    VERSION     0.1.0
    LANGUAGES   C CXX
)

# Set global language standard to C++20.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

# Detect if this is a stand-alone build.
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # Set shared output directories.
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

    # Documentation will be output here.
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fix for MLIR:
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find GUROBI
find_package(GUROBI REQUIRED)
message(STATUS "GUROBI include: ${GUROBI_INCLUDE_DIRS}")
message(STATUS "GUROBI core lib: ${GUROBI_LIBRARY}")
message(STATUS "GUROBI c++  lib: ${GUROBI_CXX_LIBRARY}")
if(NOT TARGET Gurobi::core)
  add_library(Gurobi::core SHARED IMPORTED GLOBAL)
  set_target_properties(Gurobi::core PROPERTIES
    IMPORTED_LOCATION "${GUROBI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIRS}"
  )
endif()
if(NOT TARGET Gurobi::cxx)
  add_library(Gurobi::cxx STATIC IMPORTED GLOBAL)
  set_target_properties(Gurobi::cxx PROPERTIES
    IMPORTED_LOCATION "${GUROBI_CXX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES Gurobi::core
  )
endif()

# Load LLVM and MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Preserve link-time library paths in installed binaries so extension modules
# can find LLVM/MLIR shared libraries without requiring LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Apply target properties for LLVM/MLIR globally.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
# And GUROBI
include_directories(SYSTEM ${GUROBI_INCLUDE_DIRS})

# Apply target properties for this project globally.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_compile_options(
    # NOTE: C++20 warning adjustments for this project.
    -Wall
    -Wextra
    -Wno-ambiguous-reversed-operator
    # NOTE: GDB < 10.x can't handle DWARFv5 correctly.
    $<$<CXX_COMPILER_ID:Clang>:-fdebug-default-version=4>
)

add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(lib/CAPI)
add_subdirectory(test)
add_subdirectory(tools)

if (MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  # Note that both upstream and downstreams often call this macro. It gates
  # internally on the MLIR_CONFIGURE_PYTHON_DEV_PACKAGES option.
  mlir_configure_python_dev_packages()
  add_subdirectory(python)
endif()
