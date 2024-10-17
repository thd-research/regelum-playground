#!/usr/bin/env bash

gpu=$(lspci | grep -i '.* vga .* nvidia .*')

if [[ $gpu == *' NVIDIA '* ]]; then
    docker exec -ti ros-regelum-nvi bash
else
    printf 'Nvidia GPU is not present: %s\n' "$gpu"
    docker exec -ti ros-regelum bash
fi
    