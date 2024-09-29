cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

project(PCP)

add_executable(borboleta_worker
        borboleta/worker.c borboleta/worker.h
        borboleta/my_sockets.c borboleta/my_sockets.h
        borboleta/messages.c borboleta/messages.h
)
target_link_libraries(borboleta_worker m)

add_executable(borboleta_manager
        borboleta/manager.c borboleta/manager.h
        borboleta/my_sockets.c borboleta/my_sockets.h
)
target_link_libraries(borboleta_manager m)
