#include "integral_dupla.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <X_INTERVALOS> <Y_INTERVALOS> <BLOCOS>\n", argv[0]);
        return 1;
    }

    // Lê os argumentos da linha de comando
    int x_intervalos = atoi(argv[1]);
    int y_intervalos = atoi(argv[2]);
    int n_blocos = atoi(argv[3]);

    // Número de execuções para calcular a média
    const int NUM_EXECUTIONS = 10;
    double total_time = 0.0;

    for (int run = 0; run < NUM_EXECUTIONS; run++) {
        // Mede o tempo de início
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Calcula a integral
        double result = integral_dupla_cuda(n_blocos, x_intervalos, y_intervalos);

        // Mede o tempo de fim
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time += elapsed_time;

        cudaEventDestroy(start); cudaEventDestroy(stop);

        printf("Resultado da integral: %f | Tempo: %f segundos\n", result, elapsed_time);
    }

    // Calcula o tempo médio
    double avg_time = total_time / NUM_EXECUTIONS;

    printf("Tempo médio com %d bloco(s), X=%d, Y=%d: %f segundos\n", n_blocos, x_intervalos, y_intervalos, avg_time);
    salvar_tempos(n_blocos, x_intervalos, y_intervalos, avg_time);


    return 0;
}

__global__ void integral_kernel(int x_intervalos, int y_intervalos, double h_x, double h_y, double *d_resultado) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x_intervalos && j < y_intervalos) {
        const double limite_inf = 0.0;
        const double x = limite_inf + i * h_x;
        const double y = limite_inf + j * h_y;

        double valor = F(x, y);
        if ((i == 0 || i == x_intervalos - 1) && (j == 0 || j == y_intervalos - 1))
            valor *= 0.25; // Vértices
        else if (i == 0 || i == x_intervalos - 1 || j == 0 || j == y_intervalos - 1)
            valor *= 0.5; // Arestas

        atomicAdd(d_resultado, valor); // Soma atômica para evitar condições de corrida
    }
}


double integral_dupla_cuda(int n_blocos, int x_intervalos, int y_intervalos) {
    const double limite_sup = 1.5; // limite superior da integração
    const double limite_inf = 0; // limite inferior da integração

    const double h_x = (limite_sup - limite_inf) / x_intervalos; // tamanho de cada intervalo no eixo X
    const double h_y = (limite_sup - limite_inf) / y_intervalos; // tamanho de cada intervalo no eixo Y

    double *d_resultado;
    cudaMalloc(&d_resultado, sizeof(double)); // Aloca memória na GPU para o resultado
    cudaMemset(d_resultado, 0, sizeof(double)); // Inicializa o valor com 0

    dim3 threadsPerBlock(32, 16); // Define o número de threads por bloco de forma bidimensional, totalizando 512
    dim3 numBlocks(n_blocos, n_blocos);

    integral_kernel<<<numBlocks, threadsPerBlock>>>(x_intervalos, y_intervalos, h_x, h_y, d_resultado);

    cudaDeviceSynchronize(); // Espera até que todos os cálculos sejam concluídos

    double h_resultado;
    cudaMemcpy(&h_resultado, d_resultado, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_resultado);

    h_resultado *= h_x * h_y;
    return h_resultado;
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
    fprintf(resultados, "%d\t\t%d\t\t%d\t\t%f\n", blocos, x_intervalos, y_intervalos, tempo_medio);

    fclose(resultados);
    printf("Resultados salvos em %s\n", output_file);
}
