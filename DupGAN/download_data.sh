#!/bin/bash
fileid="1esGliOE1uNvn2K6zAL5edoKqyQI7yIta"
filename="generator_discriminator_99.tar"

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ./ckpts/${filename}
rm -f cookie
