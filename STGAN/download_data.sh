#!/usr/bin/bash

# Dowload code is obtained from:
# https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99 

DATAFILEID=1-QLU_-u5RErsMppPq6d74waBLnreehAM
DATAFILENAME=ceng796_project_data.tar.xz

MODELFILEID=1FCmLtTJrtfmzau6jtdiD22rAnyayuoKt
MODELFILENAME=ceng796_project_models.tar.xz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-QLU_-u5RErsMppPq6d74waBLnreehAM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-QLU_-u5RErsMppPq6d74waBLnreehAM" -O $DATAFILENAME && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FCmLtTJrtfmzau6jtdiD22rAnyayuoKt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FCmLtTJrtfmzau6jtdiD22rAnyayuoKt" -O $MODELFILENAME && rm -rf /tmp/cookies.txt

tar -xzf ceng796_project_data.tar.xz
tar -xzf ceng796_project_models.tar.xz

rm -rf ceng796_project_data.tar.xz
rm -rf ceng796_project_models.tar.xz