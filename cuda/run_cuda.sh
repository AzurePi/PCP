#!/bin/bash

# Define variáveis
BUILD_DIR="build"
EXECUTABLE="./cuda"  # Caminho para o executável gerado

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
BLOCKS_LIST=(10 100 1000)          # Lista de números de blocos
INTERVALS_LIST=(1000 10000 100000) # Lista de intervalos

# Loop para testar todas as combinações de parâmetros
for BLOCKS in "${BLOCKS_LIST[@]}"; do
  for X_INTERVALS in "${INTERVALS_LIST[@]}"; do
    for Y_INTERVALS in "${INTERVALS_LIST[@]}"; do
      echo "Executando com $BLOCKS bloco(s), $X_INTERVALS intervalos em X, $Y_INTERVALS intervalos em Y"
      $EXECUTABLE "$X_INTERVALS" "$Y_INTERVALS" "$BLOCKS"
    done
  done
done

echo "Todas as execuções foram concluídas. Resultados salvos em tempos_CUDA.txt"
