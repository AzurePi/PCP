#include "integral_dupla.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <X_INTERVALOS> <Y_INTERVALOS> <NÚCLEOS>\n", argv[0]);
        return 1;
    }

    // Lê os argumentos da linha de comando
    int x_intervalos = atoi(argv[1]);
    int y_intervalos = atoi(argv[2]);
    int n_cores = atoi(argv[3]);

    // Inicializa mpi
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (n_cores != size) {
        if (rank == 0) {
            fprintf(stderr, "Erro: número de núcleos (%d) não coincide com o tamanho do comunicador mpi (%d).\n",
                    n_cores, size);
        }
        MPI_Finalize();
        return 1;
    }

    // Número de execuções para calcular a média
    const int NUM_EXECUTIONS = 10;
    double total_time = 0.0;

    for (int run = 0; run < NUM_EXECUTIONS; run++) {
        if (rank == 0) {
            printf("Execução %d/%d com %d núcleo(s), X=%d, Y=%d\n", run + 1, NUM_EXECUTIONS, n_cores, x_intervalos,
                   y_intervalos);
        }

        // Mede o tempo de início
        double start_time = MPI_Wtime();

        // Calcula a integral
        double result = integral_dupla_omp(x_intervalos, y_intervalos);

        // Mede o tempo de fim
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        total_time += elapsed_time;

        // Apenas o processo 0 imprime os resultados
        if (rank == 0)
            printf("Resultado da integral: %f | Tempo: %f segundos\n", result, elapsed_time);
    }

    // Calcula o tempo médio
    double avg_time = total_time / NUM_EXECUTIONS;

    // Apenas o processo 0 salva os resultados
    if (rank == 0) {
        printf("Tempo médio com %d núcleo(s), X=%d, Y=%d: %f segundos\n", n_cores, x_intervalos, y_intervalos,
               avg_time);
        salvar_tempos(n_cores, x_intervalos, y_intervalos, avg_time);
    }

    MPI_Finalize();
    return 0;
}


double integral_dupla_omp(int x_intervalos, int y_intervalos) {
    const double limite_sup = 1.5; // limite superior da integração
    const double limite_inf = 0; // limite inferior da integração

    const double h_x = (limite_sup - limite_inf) / x_intervalos; // tamanho de cada intervalo no eixo X
    const double h_y = (limite_sup - limite_inf) / y_intervalos; // tamanho de cada intervalo no eixo Y

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate workload for each process
    int x_per_proc = x_intervalos / size; // Number of x intervals per process
    int start = rank * x_per_proc;
    int end = rank == size - 1 ? x_intervalos : start + x_per_proc;

    double local_integral = 0.0; // valor final da integral

    for (int i = start; i < end; i++) {
        // para cada intervalo no eixo Y
        for (int j = 0; j < y_intervalos; j++) {
            const double x = limite_inf + i * h_x;
            const double y = limite_inf + j * h_y;

            if (i == 0 || i == x_intervalos - 1 || j == 0 || j == y_intervalos - 1)
                local_integral += 0.5 * F(x, y); // nas extremidades do trapézio, fazemos um tratamento diferente
            else
                local_integral += F(x, y); // calculamos o valor da função no ponto
        }
    }

    double global_integral = 0.0;
    MPI_Reduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        global_integral *= h_x * h_y;
    return global_integral;
}

void salvar_tempos(int cores, int x_intervalos, int y_intervalos, double tempo_medio) {
    const char *output_file = "tempos_MPI.txt";

    // Verifica se o arquivo já existe
    int arquivo_existe = access(output_file, F_OK) == 0;

    FILE *resultados = fopen(output_file, "a");
    if (resultados == NULL) {
        perror("Erro ao abrir o arquivo para salvar resultados");
        return;
    }

    // Escreve o cabeçalho apenas na primeira vez
    if (!arquivo_existe) {
        fprintf(resultados, "Cores\tIntervalo_X\tIntervalo_Y\tTempo_Medio(s)\n");
        fprintf(resultados, "---------------------------------------------\n");
    }

    // Salva os dados no arquivo
    fprintf(resultados, "%d\t\t%d\t\t%d\t\t%f\n", cores, x_intervalos, y_intervalos, tempo_medio);

    fclose(resultados);
    printf("Resultados salvos em %s\n", output_file);
}
