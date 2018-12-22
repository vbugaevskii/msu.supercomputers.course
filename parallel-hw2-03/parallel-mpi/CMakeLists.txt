cmake_minimum_required(VERSION 3.10)
project(parallel_mpi)

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(main main.cpp process_split.h border_conditions.h utils.h send_recv_mpi.h timer.h)

target_link_libraries(main ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})