#include "my_sockets.h"

int create_manager_input(int manager_port) {
    struct sockaddr_in address;
    int opt = 1;

    int socketfd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketfd < 0) {
        perror("Socket creation failed");
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(manager_port);

    setsockopt(socketfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(socketfd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        printf("Bind on manager\n");
        fprintf(stderr, "Error code: %d\n", errno);
        close(socketfd);
        return -1;
    }

    if (listen(socketfd, BACKLOG) < 0) {
        printf("Error on listen on manager\n");
        close(socketfd);
        return -1;
    }

    printf("Socket criado na porta %d\n\n", manager_port);
    printf("Aguardando conexão...\n");

    int addrlen = sizeof(address);
    int connected_socket = accept(socketfd, (struct sockaddr *)&address, (socklen_t *)&addrlen);

    printf("Conexão estabelecida\n");

    return connected_socket;
}

int create_socket_input(int worker_id, int worker_port) {
    struct sockaddr_in address;
    int opt = 1;

    int socketfd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketfd < 0) {
        perror("Socket creation failed");
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(worker_port);

    setsockopt(socketfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(socketfd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        printf("Bind on worker %d failed\n", worker_id);
        fprintf(stderr, "Error code: %d\n", errno);
        close(socketfd);
        return -1;
    }

    if (listen(socketfd, BACKLOG) < 0) {
        printf("Error on listen on worker %d\n", worker_id);
        close(socketfd);
        return -1;
    }
    printf("Worker %d: Socket criado na porta %d\n\n", worker_id, worker_port);

    return socketfd;
}

int accept_connection(int worker_id, int input_socket) {
    struct sockaddr_in address;

    int attempts = 5;

    while (attempts > 0) {
        int addrlen = sizeof(address);
        int connected_socket = accept(input_socket, (struct sockaddr *)&address, (socklen_t *)&addrlen);
        if (connected_socket >= 0)
            return connected_socket;

        perror("Accept falhou, tentando novamente...");
        attempts--;
        usleep(50000000); //(50 milissegundos)
    }

    printf("Worker %d: Número máximo de tentativas de accept atingido.\n", worker_id);
    return -1; // Falhou após o número máximo de tentativas
}


int create_socket_output(int partner_port) {
    struct sockaddr_in address;

    int socketfd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketfd < 0) {
        perror("Socket creation failed");
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_port = htons(partner_port); // Convertendo o número da porta para a ordem de bytes da rede
    address.sin_addr.s_addr = inet_addr("127.0.0.1"); // Conectar-se ao localhost

    // Tentar se conectar ao socket no caminho especificado
    int attempts = 5;
    while (attempts-- > 0) {
        if (connect(socketfd, (struct sockaddr *)&address, sizeof(address)))
            return socketfd;
        usleep(100000);
    }
    close(socketfd);
    return -1;
}
