#include "my_sockets.h"

int create_socket_input(int port_num) {
    struct sockaddr_in address;
    int opt = 1;

    int socketfd = socket(AF_LOCAL, SOCK_STREAM, 0);
    if (socketfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    setsockopt(socketfd, SOL_SOCKET,SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_num);

    if (bind(socketfd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    int backlog = 3;
    listen(socketfd, backlog);

    return socketfd;
}

int create_socket_output(int send_to_port) {
    struct sockaddr_in serv_addr;

    int socketfd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(send_to_port);

    int status = connect(socketfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    if (status < 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }

    return socketfd;
}
