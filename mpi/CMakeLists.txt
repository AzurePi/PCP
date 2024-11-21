cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(mpi)

# Diretório de saída para os executáveis
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Define o diretório de código-fonte
set(SOURCE_DIR "./src")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi
        ${SOURCE_DIR}/integral_dupla.c
        ${SOURCE_DIR}/integral_dupla.h)

target_link_libraries(mpi ${MPI_LIBRARIES} m)


