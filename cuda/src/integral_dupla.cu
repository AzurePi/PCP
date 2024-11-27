#include "integral_dupla.h"

float cpuSecond() {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (float)tp.tv_sec + tp.tv_nsec * 1.e-9f;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <X_INTERVALOS> <Y_INTERVALOS> <BLOCOS>\n", argv[0]);
        return 1;
    }

    // Intervalo da integral
    float limite_inf = 0.0f, limite_sup = 1.5f;

    // Parâmetros de entrada
    int x_intervalos = atoi(argv[1]); // Número de intervalos em x
    int y_intervalos = atoi(argv[2]); // Número de intervalos em y
    int n_blocos = atoi(argv[3]); // Número de blocos CUDA

    // Passo dos intervalos
    float h_x = (limite_sup - limite_inf) / x_intervalos;
    float h_y = (limite_sup - limite_inf) / y_intervalos;

    double total_time = 0.0f;

    float *resultado_d, resultado_h = 0.0f; // Variável para armazenar o resultado final

    // Realizar 10 execuções para calcular o tempo médio
    for (int i = 0; i < 10; ++i) {
        cudaMalloc((void **)&resultado_d, sizeof(float));
        cudaMemset(resultado_d, 0, sizeof(float));

        const double start = cpuSecond(); // Medir o tempo de execução
        integral_dupla_cuda<<<n_blocos, THREADS_POR_BLOCO>>>(resultado_d, h_x, h_y, x_intervalos, y_intervalos,
                                                             limite_inf);
        cudaDeviceSynchronize();
        const double end = cpuSecond(); // Medir o tempo de execução
        const double run_time = end - start;
        total_time += run_time;

        // Recuperar o resultado
        cudaMemcpy(&resultado_h, resultado_d, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(resultado_d);

        resultado_h *= h_x * h_y; // Ajustar o resultado pelo tamanho dos passos
        printf("Resultado da integral: %f | Tempo: %lf\n", resultado_h, run_time);
    }

    double avg_time = total_time / 10.0;

    printf("Tempo médio com %d bloco(s), X=%d, Y=%d: %f segundos\n", n_blocos, x_intervalos, y_intervalos, avg_time);
    salvar_tempos(n_blocos, x_intervalos, y_intervalos, avg_time);

    return 0;
}

// Kernel CUDA para calcular a integral dupla
__global__ void integral_dupla_cuda(float *resultado, float h_x, float h_d, int x_intervalos, int y_intervalos,
                                    float limite_inf) {
    __shared__ float soma_bloco[THREADS_POR_BLOCO]; //acumulador para o bloco
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float soma_thread = 0.0f; //acumulador para um dado thread

    // Cada thread processa múltiplos intervalos
    for (int i = idx; i < x_intervalos * y_intervalos; i += totalThreads) {
        int ix = i % x_intervalos;
        int iy = i / x_intervalos;

        float x = limite_inf + ix * h_x;
        float y = limite_inf + iy * h_d;

        float f = F(x, y);
        if (ix == 0 || ix == x_intervalos - 1 || iy == 0 || iy == y_intervalos - 1)
            soma_thread += f * 0.5f; // borda
        else
            soma_thread += f; // interior
    }

    soma_bloco[threadIdx.x] = soma_thread;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            soma_bloco[threadIdx.x] += soma_bloco[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(resultado, soma_bloco[0]);
}

void salvar_tempos(int blocos, int x_intervalos, int y_intervalos, double tempo_medio) {
    const char *output_file = "tempos_CUDA.txt";

    // Verifica se o arquivo já existe
    int arquivo_existe = access(output_file, F_OK) == 0;

    FILE *resultados = fopen(output_file, "a");
    if (resultados == NULL) {
        perror("Erro ao abrir o arquivo para salvar resultados");
        return;
    }

    // Escreve o cabeçalho apenas na primeira vez
    if (!arquivo_existe) {
        fprintf(resultados, "Blocos\tIntervalo_X\tIntervalo_Y\tTempo_Medio(s)\n");
        fprintf(resultados, "---------------------------------------------\n");
    }

    // Salva os dados no arquivo
    fprintf(resultados, "%d\t\t%d\t\t%d\t\t%lf\n", blocos, x_intervalos, y_intervalos, tempo_medio);

    fclose(resultados);
    printf("Resultados salvos em %s\n", output_file);
}
