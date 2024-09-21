#include "worker.h"

int process_id;
int process_port;

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s[filename] X[process id] Y[operand]\n", argv[0]);
        return 0;
    }

    srand((unsigned)time(NULL) * process_port / process_id);

    // pegamos o ID atribuído ao worker, e com base nisso achamos sua porta
    process_id = atoi(argv[1]);
    process_port = PORT_WORKER(process_id);

    // setamos a quantidade de etapas do processo, e iteramos por elas
    int etapa = 0;
    int max_etapas = 4;

    int operando;
    while (etapa < max_etapas) {
        // determinamos se esse worker será usado nessa etapa
        int factor = pow(2, etapa);
        bool used = process_id % factor == 0;

        // se essa porta não for usada nessa etapa, não será em nenhuma das posteriores, então saímos do laço
        if (!used)
            break;

        // determinamos para qual outro processo devemos enviar dados
        int out_id = -1;
        int socket_out = -1;

        if (etapa < max_etapas - 1) {
            out_id = process_id - etapa - 1;
            socket_out = PORT_WORKER(out_id);
        } else // se estamos na última etapa,
            socket_out = PORT_MANAGER;

        // se estamos na primeira etapa, atribuímos um valor aleatório ao operando
        if (etapa == 0) {
            operando = rand() * 100;
        } else {
            //TODO: senão, recebemos um valor de input, e somamos ao operando que já temos
            int recebido;

            operando += recebido;
        }

        //TODO: enviamos um valor para socket_out

        etapa++;
    }


    return 0;
}
