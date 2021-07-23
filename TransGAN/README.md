# [TransGAN: Two Transformers Can Make One Strong GAN, and That Can Scale Up, CVPR 2021](https://arxiv.org/pdf/2102.07074v2.pdf)


Paper Authors: Yifan Jiang, Shiyu Chang, Zhangyang Wang

*CVPR 2021*

<table>
<tr>
<td style="text-align: center">0 Epoch</td>
<td style="text-align: center">40 Epoch</td> 
<td style="text-align: center">100 Epoch</td>
<td style="text-align: center">200 Epoch</td> 
</tr>
<trt>
<p align="center"><img width="30%" src="https://raw.githubusercontent.com/asarigun/TransGAN/main/images/atransgan_cifar.gif"></p>
</tr>
<tr>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/0.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/40.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/100.jpg" style="width: 400px;"/> </td>
<td> <img src="https://raw.githubusercontent.com/asarigun/TransGAN/main/results/200.jpg" style="width: 400px;"/> </td>
</tr>
</table>

This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models. The re-implementation is provided by:

* Ahmet Sarıgün, ahmet.sarigun@metu.edu.tr

* Dursun Bekci, bekci.dursun@metu.edu.tr

Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.

## Installation

Before running ```train.py```, check whether you have libraries in ```requirements.txt```! Also, create ```./fid_stat``` folder and download the [fid_stat](bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) file in this folder. To save your model during training, create ```./checkpoint``` folder using ```mkdir checkpoint```.

## Training 

```python train.py```


## Checkpoint

You can find the checkpoint [here](https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view?usp=sharing) that we saved during training for this project.


## Citation
```
@article{jiang2021transgan,
  title={TransGAN: Two Transformers Can Make One Strong GAN},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
```
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
```
@inproceedings{zhao2020diffaugment,
  title={Differentiable Augmentation for Data-Efficient GAN Training},
  author={Zhao, Shengyu and Liu, Zhijian and Lin, Ji and Zhu, Jun-Yan and Han, Song},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```