#!/bin/bash

BORBOLETA_PATH="./borboleta"

# Compilação do código utilizando CMake
cd $BORBOLETA_PATH
cmake .
make


# Número de workers na barreira
NUM_WORKERS=8

SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

COLUMNS=4  # quantidade de colunas desejada
ROWS=$(( (NUM_WORKERS + COLUMNS - 1) / COLUMNS ))  # número de linhas necessárias

# calculamos o tamanho das janelas de acordo com o tamanho da tela
TERMINAL_WIDTH=$((SCREEN_WIDTH / COLUMNS))
TERMINAL_HEIGHT=$((SCREEN_HEIGHT / ROWS))

# abrimos o manager
gnome-terminal --geometry=80x24+0+0 -- bash -c "./borboleta_manager; exec bash"

# um loop para criar os workers
for i in $(seq 0 $((NUM_WORKERS - 1)))
do
    # Calcula em qual coluna e qual linha a janela está
    col=$((i % COLUMNS))
    row=$((i / COLUMNS))

    # Calcula a posição da janela com base na linha e coluna
    pos_x=$((col * TERMINAL_WIDTH))
    pos_y=$((row * TERMINAL_HEIGHT))

    # abrimos o worker
    gnome-terminal --geometry=80x24+$pos_x+$pos_y -- bash -c "./borboleta_worker $i; exec bash"

    sleep 0.6 # delay para facilitar a visualização
done
