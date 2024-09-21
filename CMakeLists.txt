cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

project(PCC)

add_executable(borboleta
        borboleta/server.c borboleta/server.h
        borboleta/worker.c borboleta/worker.h
        borboleta/my_sockets.c borboleta/my_sockets.h)
target_link_libraries(borboleta m)
