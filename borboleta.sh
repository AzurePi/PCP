#!/bin/bash

BORBOLETA_PATH="./borboleta"

# Compilação do código utilizando CMake
mkdir -p $BORBOLETA_PATH
cd $BORBOLETA_PATH
cmake ..
make

# Voltar ao diretório original
cd ..

# Number of windows
NUM_PROCESSES=8

# Screen dimensions (adjust to your screen resolution)
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

# Terminal window size
TERMINAL_WIDTH=300
TERMINAL_HEIGHT=300

# Calculate the number of columns and rows for tiling
COLUMNS=$((SCREEN_WIDTH / TERMINAL_WIDTH))
ROWS=$((SCREEN_HEIGHT / TERMINAL_HEIGHT))

# Open the
gnome-terminal -- bash -c "$BORBOLETA_PATH/borboleta_manager; exec bash"

# Loop to open terminals in a tiled arrangement
for i in $(seq 0 $((NUM_PROCESSES - 1)))
do
    # Calculate the column and row positions for tiling
    col=$((i % COLUMNS))
    row=$((i / COLUMNS))

    # Calculate the x and y position for the window
    pos_x=$((col * TERMINAL_WIDTH))
    pos_y=$((row * TERMINAL_HEIGHT))

    # Open the terminal with the specified size and position
    gnome-terminal --geometry=80x24+$pos_x+$pos_y -- bash -c "$BORBOLETA_PATH/borboleta_worker $i; exec bash"

    sleep 0.1
done
