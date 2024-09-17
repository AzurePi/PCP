#include "main.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s[filename] X[process id] Y[operand]\n", argv[0]);
        return 0;
    }
    const int process_id = atoi(argv[1]);

    srand((unsigned)time(NULL) * process_id);
    int operand = rand() * 100;



    return 0;
}
