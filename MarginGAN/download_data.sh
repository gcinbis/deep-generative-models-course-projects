#!/bin/bash

## Uncomment if the download of pretrained classifiers are necessary
#FILE="$PWD/pretrained_classifiers"
#
#if [ ! -e $FILE ]; then
#  mkdir pretrained_classifiers
#fi
#
#cd pretrained_classifiers
#
#wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/pretrained_classifiers/pre_cls_label_model_100.pt 
#wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/pretrained_classifiers/pre_cls_label_model_600.pt 
#wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/pretrained_classifiers/pre_cls_label_model_1000.pt
#wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/pretrained_classifiers/pre_cls_label_model_3000.pt
#
#cd ..

FILE="$PWD/best_models"

if [ ! -e $FILE ]; then
  mkdir best_models
fi

cd best_models
if [ ! -e "$FILE/60" ]; then
  mkdir 60
fi
if [ ! -e "$FILE/61" ]; then
  mkdir 61
fi
if [ ! -e "$FILE/62" ]; then
  mkdir 62
fi
if [ ! -e "$FILE/63" ]; then
  mkdir 63
fi
if [ ! -e "$FILE/64" ]; then
  mkdir 64
fi

cd 60
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/C_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/D_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/G_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/C_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/D_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/G_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/C_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/D_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/G_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/C_epoch_21_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/D_epoch_21_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/60/G_epoch_21_label_3000.pt
cd ..

cd 61
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/C_epoch_15_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/D_epoch_15_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/G_epoch_15_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/C_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/D_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/G_epoch_7_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/C_epoch_10_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/D_epoch_10_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/G_epoch_10_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/C_epoch_39_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/D_epoch_39_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/61/G_epoch_39_label_3000.pt
cd ..

cd 62
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/C_epoch_29_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/D_epoch_29_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/G_epoch_29_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/C_epoch_21_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/D_epoch_21_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/G_epoch_21_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/C_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/D_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/G_epoch_15_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/C_epoch_5_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/D_epoch_5_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/62/G_epoch_5_label_3000.pt
cd ..

cd 63
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/C_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/D_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/G_epoch_23_label_100.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/C_epoch_10_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/D_epoch_10_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/G_epoch_10_label_600.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/C_epoch_31_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/D_epoch_31_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/G_epoch_31_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/C_epoch_15_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/D_epoch_15_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/63/G_epoch_15_label_3000.pt
cd ..

cd 64
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/C_epoch_24_label_100.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/D_epoch_24_label_100.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/G_epoch_24_label_100.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/C_epoch_40_label_600.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/D_epoch_40_label_600.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/G_epoch_40_label_600.pt 
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/C_epoch_11_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/D_epoch_11_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/G_epoch_11_label_1000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/C_epoch_1_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/D_epoch_1_label_3000.pt
wget -c https://github.com/sonatbaltaci/marginGAN/raw/master/best_models/64/G_epoch_1_label_3000.pt
cd ..
