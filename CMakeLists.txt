cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(lstm)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 11)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

if (NOT CONFIGURED_ONCE)
    set(CMAKE_CXX_FLAGS "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_C_FLAGS   "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()

include_directories(include)

find_package(Boost COMPONENTS system filesystem timer REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(Eigen3_DIR /usr/local/share/eigen3/cmake/)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable (lstm main.cpp)
target_sources(lstm PUBLIC "lstm.cpp")
target_link_libraries (lstm Eigen3::Eigen)
target_link_libraries(lstm 
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_TIMER_LIBRARY}
)

set_target_properties(lstm PROPERTIES LINKER_LANGUAGE CXX)

set(CONFIGURED_ONCE TRUE CACHE INTERNAL
    "A flag showing that CMake has configured at least once.")