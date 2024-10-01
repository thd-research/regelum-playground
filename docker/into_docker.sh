#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
gpu=$(lspci | grep -i '.* vga .* nvidia .*')

if [[ $gpu == *' NVIDIA '* ]]; then
    docker exec -ti ros-regelum-nvi bash
else
    printf 'Nvidia GPU is not present: %s\n' "$gpu"
    docker exec -ti ros-regelum bash
fi
