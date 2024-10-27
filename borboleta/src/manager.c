#include "manager.h"

int main() {
    int resultado;

    printf("Manager: ");

    int input_socket = create_manager_input(PORT_MANAGER);

    recv(input_socket, &resultado, sizeof(int), 0);

    printf("\nResultado: %d\n\n", resultado);
    printf("Barreira borboleta finalizada\n");

    return 0;
}
