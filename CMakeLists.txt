cmake_minimum_required(VERSION 3.10)
project(PCP)

# Adiciona os subprojetos
add_subdirectory(borboleta)
add_subdirectory(openMP)
add_subdirectory(mpi)
add_subdirectory(cuda)

