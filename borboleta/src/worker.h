#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "my_sockets.h"
#include "messages.h"

#define PORT_MANAGER 8080
#define PORT_WORKER(worker_id) PORT_MANAGER + worker_id + 1


bool send_operand(int input_socket, int partner_id, int partner_port);
bool receive_operand(int input_socket, int partner_id, int partner_port, int *recebido);


#endif //MAIN_H
