#ifndef MY_SOCKETS_H
#define MY_SOCKETS_H

#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>

int create_socket_input(int port_num);

int create_socket_output(int send_to_port);

#endif //MY_SOCKETS_H
