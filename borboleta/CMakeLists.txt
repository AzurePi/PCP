cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(Borboleta)

# Diretório de saída para os executáveis
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Define o diretório de código-fonte
set(SOURCE_DIR "./src")

# Executável de um worker
add_executable(borboleta_worker
        ${SOURCE_DIR}/worker.c ${SOURCE_DIR}/worker.h
        ${SOURCE_DIR}/my_sockets.c ${SOURCE_DIR}/my_sockets.h
        ${SOURCE_DIR}/messages.c ${SOURCE_DIR}/messages.h
)
target_link_libraries(borboleta_worker m)

# Executável do manager
add_executable(borboleta_manager
        ${SOURCE_DIR}/manager.c ${SOURCE_DIR}/manager.h
        ${SOURCE_DIR}/my_sockets.c ${SOURCE_DIR}/my_sockets.h
)
target_link_libraries(borboleta_manager m)

