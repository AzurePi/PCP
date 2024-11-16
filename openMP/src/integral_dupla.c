#include "integral_dupla.h"

int main() {
    const int threads[] = {1, 2, 4, 8};
    const int threads_count = sizeof(threads) / sizeof(int);

    const int intervalos[] = {10e2, 10e3, 10e4}; // 1.000, 10.000, 100.000
    const int intervalos_count = sizeof(intervalos) / sizeof(int);

    double tempos[threads_count][intervalos_count][intervalos_count];

    for (int t = 0; t < threads_count; t++) {
        const int n_threads = threads[t];

        for (int x = 0; x < intervalos_count; x++) {
            const int x_intervalos = intervalos[x];

            for (int y = 0; y < intervalos_count; y++) {
                const int y_intervalos = intervalos[y];

                // realizamos cada uma das 36 combinações 10 vezes, para estabelecermos uma média dos tempos
                double total_time = 0.0; // Inicializa o tempo total para a média

                for (int run = 0; run < 10; run++) { // Executa 10 vezes
                    printf("%d %d %d (execução %d): ", n_threads, x_intervalos, y_intervalos, run + 1);
                    const double begin = omp_get_wtime();
                    const double val = integral_dupla_omp(n_threads, x_intervalos, y_intervalos);
                    const double end = omp_get_wtime();

                    const double time_taken = end - begin;
                    total_time += time_taken; // Acumula o tempo
                    printf("%fs (integral = %f)\n", time_taken, val);
                }

                // Calcula a média dos tempos
                tempos[t][x][y] = total_time / 10.0;
            }
        }
    }

    salvar_tempos(tempos, threads_count, intervalos_count);
}


double integral_dupla_omp(int n_threads, int x_intervalos, int y_intervalos) {
    const double limite_sup = 1.5; // limite superior da integração
    const double limite_inf = 0; // limite inferior da integração

    const double h_x = (limite_sup - limite_inf) / x_intervalos; // tamanho de cada intervalo no eixo X
    const double h_y = (limite_sup - limite_inf) / y_intervalos; // tamanho de cada intervalo no eixo Y

    double integral = 0.0; // valor final da integral

    // reduction(+integral) define a variável "integral" como acumuladora da soma, de forma segura
    // collapse(2) faz com que o OpenMP trate os laços aninhados como um único laço para propósitos de distribuição
#pragma omp parallel for num_threads(n_threads) reduction(+:integral) collapse(2)
    // para cada intervalo no eixo X
    for (int i = 0; i < x_intervalos; i++) {
        // para cada intervalo no eixo Y
        for (int j = 0; j < y_intervalos; j++) {
            const double x = limite_inf + i * h_x;
            const double y = limite_inf + j * h_y;

            if (i == 0 || i == x_intervalos - 1 || j == 0 || j == y_intervalos - 1)
                integral += 0.5 * F(x, y); // nas extremidades do trapézio, fazemos um tratamento diferente
            else
                integral += F(x, y); // calculamos o valor da função no ponto
        }
    }

    integral *= h_x * h_y; // multiplicamos pelo tamanho de cada intervalo do trapézio
    return integral;
}

void salvar_tempos(double (*tempos)[3][3], int cores_count, int intervalos_count) {
    printf("Salvando tempos...\n");

    FILE *resultados = fopen("./tempos_openMP.txt", "w");

    for (int i = 0; i < cores_count; i++) {
        for (int j = 0; j < intervalos_count; j++) {
            for (int k = 0; k < intervalos_count; k++) {
                fprintf(resultados, "%.0f threads, %dº tamanho de intervalo X, %dº tamanho de intervalo Y: ", pow(2,i), j + 1,
                        k + 1);
                fprintf(resultados, "%f s\n", tempos[i][j][k]);
            }
        }
    }
    fclose(resultados);
}
