#!/bin/bash
fileid="1doCIMjlUuNS8HxSHWCM1Frf66WnGbYQE"
filename="dataset.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf dataset.tar.xz



fileid="1x6W1O1Y7pk6VbClY2oLP-T_gVi5CyfCt"
filename="ground_truth_dataset.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf ground_truth_dataset.tar.xz



fileid="1FWmrE9I-Nq7SyVOYPnaWa6ShPcdeG-xb"
filename="mask.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf mask.tar.xz


fileid="1o6EeKMgkhscHAUMOWfmbQp1hVWdTJ-le"
filename="goal_dataset.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf goal_dataset.tar.xz

fileid="197LLup_7WhEnr2hHv0epvgv3wUPovw-t"
filename="test_data.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf test_data.tar.xz


fileid="1ciIEF-T_riQaqC3EisevlpQWaHHEZA9l"
filename="models.tar.xz"

CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=${fileid}" -O ${filename}
rm -f /tmp/cookies.txt
tar -xf models.tar.xz


