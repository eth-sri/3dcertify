project(DeepG)

cmake_minimum_required(VERSION 3.3)

option(CXX "enable C++ compilation" ON)
if (CXX)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O3 -march=native -mavx2 -mfma")
endif ()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Visual Studio compiler with static runtime libraries
if (MSVC AND MT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif ()

set(GUROBI_CXX_LIBRARY "$ENV{GUROBI_HOME}/lib/libgurobi_c++.a")
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(sources)
include_directories(.)
file(GLOB ABSTRACTION abstraction/*.cpp)
file(GLOB DOMAINS domains/*.cpp)
file(GLOB TRANSFORMS transforms/*.cpp)
file(GLOB UTILS utils/*.cpp)

add_executable(deepg_constraints deepg_constraints.cpp ${ABSTRACTION} ${DOMAINS} ${TRANSFORMS} ${UTILS})
add_executable(deepg_pointclouds deepg_pointclouds.cpp ${ABSTRACTION} ${DOMAINS} ${TRANSFORMS} ${UTILS})

if (CXX)
    target_link_libraries(deepg_constraints optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(deepg_pointclouds ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif ()

target_link_libraries(deepg_constraints ${GUROBI_LIBRARY} Threads::Threads)
target_link_libraries(deepg_pointclouds ${GUROBI_LIBRARY} Threads::Threads)

if (${CMAKE_SOURCE_DIR} STREQUAL ..)
    include(FeatureSummary)
    feature_summary(WHAT ALL)
endif ()
