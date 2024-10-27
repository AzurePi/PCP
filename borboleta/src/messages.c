#include "messages.h"

bool send_ready(int partner_id, int output_socket) {
    usleep(1000);
    char ready_message[] = "READY";

    if (!send_message(output_socket, partner_id, ready_message)) {
        printf("Error sending ready response to worker %d\n", partner_id);
        return false;
    }
    return true;
}

bool receive_ready(int partner_id, int connected_socket) {
    char response[6];

    if (!receive_message(connected_socket, response)) {
        printf("Error receiving ready response from worker %d\n", partner_id);
        return false;
    }

    if (strcmp(response, "READY") != 0) {
        printf("Unexpected response from worker %d: %s\n", partner_id, response);
        return false;
    }
    return true;
}

bool await_message(int partner_id, int input_socket, char message[6]) {
    int attempts = MAX_ATTEMPTS;
    bool right_sender;

    // esperamos até que recebamos a mensagem do parceiro esperado
    do {
        if (attempts == 0) { // verificamos se ainda temos chances para tentar
            printf("Too many tries. Finishing process\n");
            return false;
        }

        if (!receive_message(input_socket, message)) { // se não conseguimos receber uma mensagem
            return false;
        }
        right_sender = check_sender(message, partner_id);

        if (!right_sender) {
            printf("Awaiting message from %d...\n", partner_id);
            usleep(RETRY_DELAY); // usamos um delay antes de tentarmos receber outra mensagem
        }

        attempts--;
    }
    while (!right_sender && attempts >= 0);
    return right_sender;
}

bool receive_message(int input_socket, char message[]) {
    int attempts = MAX_ATTEMPTS;
    int has_received;

    do {
        if (attempts == 0) { // verificamos se ainda temos chances para tentar
            printf("Too many tries. Finishing process\n");
            return false;
        }

        // tentamos receber uma mensagem
        has_received = recv(input_socket, message, sizeof(message), 0);

        // se não recebemos algo,
        if (has_received <= 0) {
            printf("No message received\n");
            printf("Retrying...\n");
            usleep(RETRY_DELAY);
        }
        attempts--;
    }
    while (has_received <= 0 && attempts >= 0);

    return has_received > 0;
}

bool check_sender(char message[6], int partner_id) {
    int received_id;

    sscanf(message, "%d|", &received_id);

    if (partner_id != received_id) {
        printf("Received message from worker %d, but expected worker %d\n", received_id, partner_id);
        return false;
    }

    return true;
}

bool send_message(int output_socket, int partner_id, char message[6]) {
    int attempts = MAX_ATTEMPTS;
    while (send(output_socket, message, strlen(message) + 1, 0) < 0 && attempts >= 0) {
        printf("Error sending message to worker %d\n", partner_id);
        printf("Retrying...\n");

        if (attempts == 0) {
            printf("Too many tries. Finishing process\n");
            return false;
        }

        usleep(RETRY_DELAY);
        attempts--;
    }
    return true;
}

bool send_and_confirm(int worker_id, int partner_id, int input_socket, char message[6], int output_socket) {
    bool confirmed;
    int attempts = MAX_ATTEMPTS;
    do {
        if (attempts == 0) {
            printf("Too many tries. Finishing process\n");
            return false;
        }

        if (!send_message(output_socket, partner_id, message))
            return false; // se o envio da mensagem falhou, abortamos

        int connected_socket = accept_connection(worker_id, input_socket);
        confirmed = receive_ready(partner_id, connected_socket);
        close(connected_socket);

        // se não recebemos algo,
        if (!confirmed) {
            printf("No confirmation received\n");
            printf("Retrying...\n");
            usleep(RETRY_DELAY);
        }

        attempts--;
    }
    while (!confirmed && attempts >= 0);
    return confirmed;
}

bool receive_and_confirm(int partner_id, int partner_port, int connected_socket, char message[]) {
    if (!await_message(partner_id, connected_socket, message)) {
        printf("Error receiving message from worker %d\n", partner_id);
        close(connected_socket);
        return false;
    }
    close(connected_socket);

    int output_socket = create_socket_output(partner_port);
    if (output_socket < 0) {
        close(output_socket);
        return false;
    }

    if (!send_ready(partner_id, output_socket)) {
        close(output_socket);
        return false;
    }
    close(output_socket);

    return true;
}
