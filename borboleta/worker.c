#include "worker.h"

#define N_WORKERS 8

int worker_id;
int worker_port;
int operando = 0;

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <process_num>\n", argv[0]);
        printf("Wrong number of arguments");
        return 1;
    }

    // pegamos o ID atribuído ao worker, e com base nisso achamos sua porta
    worker_id = atoi(argv[1]);
    worker_port = PORT_WORKER(worker_id);

    int input_socket = create_socket_input(worker_id, worker_port);
    if (input_socket <= 0) {
        printf("Error creating input socket for worker %d\n", worker_id);
        return 1;
    }

    srand(time(0) * worker_port + worker_id);

    // setamos a quantidade de etapas do processo, e iteramos por elas
    int etapa = 1;
    int max_etapas = log(N_WORKERS) / log(2) + 2;

    while (etapa < max_etapas) {
        // determinamos se esse worker será usado nessa etapa
        bool used = worker_id % etapa == 0;
        if (!used)
            break;

        printf("Etapa %.0f\n", log(etapa)/log(2));

        // se estamos na primeira etapa, atribuímos um valor aleatório ao operando, em [1,99]
        if (etapa == 1)
            operando = rand() % 99 + 1;

        // determinamos com qual outro worker vamos trocar mensagens (nosso parceiro nessa etapa)
        int partner_id = worker_id ^ etapa;
        int partner_port = PORT_WORKER(partner_id);

        // se somos o worker maior, devemos enviar nosso valor ao menor
        if (worker_id > partner_id) {
            if (!send_operand(input_socket, partner_id, partner_port))
                return 1;
        }
        else { // se somos o worker menor, devemos receber um valor e realizar uma operação
            int recebido;

            if (!receive_operand(input_socket, partner_id, partner_port, &recebido))
                return 1;

            operando += recebido; // somamos o valor obtido ao valor que já tínhamos
        }
        usleep(5000);
        etapa *= 2; // avançamos para a próxima etapa
    }
    close(input_socket);

    // o worker 0 deve enviar seu valor para o manager no final
    if (worker_id == 0) {
        printf("Worker 0 enviando resultado ao manager\n\n");

        int output_socket = create_socket_output(PORT_MANAGER);
        send(output_socket, &operando, sizeof(operando), 0);
        close(output_socket);
    }

    printf("Worker %d encerrou seu trabalho com sucesso\n", worker_id);
    return 0;
}

bool send_operand(int input_socket, int partner_id, int partner_port) {
    char message[6];

    sprintf(message, "%01d|%03d", worker_id, operando); // nossa mensagem terá o formato "porta|operando"

    printf("Sending \"%d\" to worker %d on port %d\n", operando, partner_id, partner_port);

    int output_socket = create_socket_output(partner_port);
    if (output_socket <= 0) {
        printf("Error creating output socket for worker %d\n", worker_id);
        return false;
    }

    if (!send_and_confirm(worker_id, partner_id, input_socket, message, output_socket)) {
        close(output_socket);
        return false;
    }
    close(output_socket);
    printf("\n");
    return true;
}

bool receive_operand(int input_socket, int partner_id, int partner_port, int *recebido) {
    char message[6];

    int connected_socket = accept_connection(worker_id, input_socket);
    if (connected_socket <= 0) {
        printf("Error creating connected socket for worker %d\n", worker_id);
        return false;
    }

    if (!receive_and_confirm(partner_id, partner_port, connected_socket, message))
        return false;
    close(connected_socket);

    sscanf(message, "%*d|%d", recebido);
    return true;
}
