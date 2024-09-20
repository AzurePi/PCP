#include "worker.h"

int process_id;
int process_port;


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s[filename] X[process id] Y[operand]\n", argv[0]);
        return 0;
    }

    // inicializar os sockets

    srand((unsigned)time(NULL) * process_port / process_id);

    process_id = atoi(argv[1]);
    process_port = PORT_WORKER(process_id);

    int etapa = 0;
    int operand;

    while (etapa < 4) {
        switch (etapa) {
        case 0: operand = rand() * 100;

            // determinar para qual portão enviar
            // enviar o operando para o portão da próxima etapa

            etapa = 1;
            break;
        case 1: etapa = 2;
            break;
        case 2: etapa = 3;
            break;
        case 3: etapa = 4;
            break;
        default: break;
        }
    }


    return 0;
}
