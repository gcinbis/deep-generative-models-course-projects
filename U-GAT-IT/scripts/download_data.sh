#!/bin/sh
if [ -f selfie2anime.zip ]; then
	echo "Data is already available."
else
	echo "Downloading..."
	wget --no-check-certificate -r 'https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view' -O selfie2anime.zip
	echo "Download successfull."
fi

dir="../data/selfie2anime"

if [ -d "$dir" ]; then
    echo "$dir exists."
else 
    mkdir $dir
fi
echo "Extracting..."
unzip selfie2anime.zip -d $dir
echo "Extraction completed."