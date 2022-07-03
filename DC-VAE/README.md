# [Dual Contradistinctive Generative Autoencoder](https://arxiv.org/abs/2011.10063)

Gaurav Parmar, Dacheng Li, Kwonjoon Lee, Zhuowen Tu

*CVPR 2021*

<p align="center"><img src="https://raw.githubusercontent.com/hcagri/Dual-Contradistinctive-Generative-Autoencoder/master/figures/Sampling.png" alt="Model" style="height: 300px; width:300px;"/></p>
<p align="center">Figure: Sampling Results of Re-implementation.</p>


This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models.  
The re-implementation is provided by:
* Aybora Köksal, aybora@metu.edu.tr
* Halil Çağrı Bilgi, cagri.bilgi@metu.edu.tr

The re-implemetation is reviewed by
* Sezai Artun Ozyegin – Merve Tapli

Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.



## Installation
First [anaconda](https://www.anaconda.com/products/distribution#Downloads) package manager has to be installed on your system. \
Then, to create the correct dependecies, run the below command. 
```
conda env create --file requirements.txt
```
`Note:` This requirements txt is only for cpu use \
Activate the conda environment
```
conda activate DC-VAE-env
```
To train the model use the below command. This command will start training, and creates a runs folder on the main directory where the metrics and logs of each experiment are easily tracktable. 
```
python run.py
```
Produce qualitative and quantitative results with pre-trained model, look for the RESULTS part in [main.ipynb](main.ipynb) 
