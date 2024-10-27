cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(OpenMP)

# Diretório de saída para os executáveis
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Define o diretório de código-fonte
set(SOURCE_DIR "./src")

find_package(OpenMP)

add_executable(OpenMP
        ${SOURCE_DIR}/integral_dupla.c ${SOURCE_DIR}/integral_dupla.h)

target_link_libraries(OpenMP PRIVATE OpenMP::OpenMP_C m)


