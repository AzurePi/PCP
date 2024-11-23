cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cuda LANGUAGES CXX CUDA)

# Diretório de saída para os executáveis
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Define o diretório de código-fonte
set(SOURCE_DIR "./src")

set(SOURCE_FILES
        ${SOURCE_DIR}/integral_dupla.cu
        ${SOURCE_DIR}/integral_dupla.h
)

add_executable(cuda ${SOURCE_FILES})

set_target_properties(cuda PROPERTIES CUDA_ARCHITECTURES "native")

target_link_libraries(cuda m)
