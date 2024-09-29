#ifndef MY_SOCKETS_H
#define MY_SOCKETS_H

#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define BACKLOG 3

int create_manager_input(int manager_port);

int create_socket_input(int worker_id, int worker_port);
int accept_connection(int worker_id, int input_socket);
int create_socket_output(int partner_port);

#endif //MY_SOCKETS_H
