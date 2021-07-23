#!/bin/sh
if [ -f chkpt.zip ]; then
	echo "Data is already available."
else [alt text]
	echo "Downloading..."
	wget --no-check-certificate -r 'https://drive.google.com/u/0/uc?id=12rsi3jNxflYBiyctn6ipNu2Tdh2WA_SZ&export=download' -O chkpt.zip
	echo "Download successfull."
fi

dir="../saved"

if [ -d "$dir" ]; then
    echo "$dir exists."
else 
    mkdir $dir
fi
echo "Extracting..."
unzip chkpt.zip -d $dir
echo "Extraction completed."