#!/bin/sh
if [ -f img_align_celeba.zip ]; then
	echo "Aldready downloaded!"
else
	echo "Downloading"
	wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1JKAluJEagidnUYin77yjoiN_FW63zuZj' -O img_align_celeba.zip
	echo "Download completed!"
fi
PATH="../dataset/celebA"
if [ -d "$PATH" ]; then
    echo "$PATH exists!"
else 
    mkdir ../dataset
    mkdir ../dataset/celebA
fi
echo "Extracting..."
unzip img_align_celeba.zip -d ../dataset/celebA
echo "Extraction completed!."

