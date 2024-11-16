#!/bin/bash

# Define variáveis
BUILD_DIR="build"
EXECUTABLE="./mpi"  # Caminho para o executável gerado

# Remove diretório de build anterior (se existir) e cria um novo
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# Compila o código com CMake
cmake ..
make

# Volta ao diretório raiz
cd ..

# Define os parâmetros de teste
CORES_LIST=(1 2 4 8)          # Lista de números de núcleos
INTERVALS_LIST=(1000 10000 100000) # Lista de intervalos

# Loop para testar todas as combinações de parâmetros
for CORES in "${CORES_LIST[@]}"; do
  for X_INTERVALS in "${INTERVALS_LIST[@]}"; do
    for Y_INTERVALS in "${INTERVALS_LIST[@]}"; do
      echo "Executando com $CORES núcleo(s), X=$X_INTERVALS, Y=$Y_INTERVALS"
      mpirun -np "$CORES" $EXECUTABLE "$X_INTERVALS" "$Y_INTERVALS" "$CORES"
    done
  done
done

echo "Todas as execuções foram concluídas. Resultados salvos em tempos_MPI.txt"
