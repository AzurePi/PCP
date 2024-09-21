#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "my_sockets.h"


#define PORT_MANAGER 8080
#define PORT_WORKER(X) PORT_MANAGER + X + 1


#endif //MAIN_H
