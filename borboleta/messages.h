#ifndef MESSAGES_H
#define MESSAGES_H

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "my_sockets.h"

#define MAX_ATTEMPTS 10
#define RETRY_DELAY 100000

bool send_ready(int partner_id, int output_socket);
bool receive_ready(int partner_id, int connected_socket);

bool await_message(int partner_id, int input_socket, char message[6]);

bool receive_message(int input_socket, char message[]);
bool check_sender(char *str, int partner_id);
bool receive_and_confirm(int partner_id, int partner_port, int connected_socket, char message[]);

bool send_message(int output_socket, int partner_id, char message[6]);
bool send_and_confirm(int worker_id, int partner_id, int input_socket, char message[6], int output_socket);


#endif //MESSAGES_H
