# [Wasserstein Auto Encoders](https://arxiv.org/pdf/1711.01558v4.pdf)

Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly and Bernhard Scholkopf

*ICLR 2018*

## Implementation Results
### Qualitative Results
There are three qualitative results as proposed in the paper: image reconstrution, interpolation and random-sampling. Image reconstruction and interpolation are applied to test images.  

#### Reconstruction
Image reconstruction performs the reconstruction of an image using an encoder and a decoder.
![Alt text](image/this_implementation_results/test-reconstruction.png?raw=true)

#### Interpolation
Interpolation steps of two images (*x1*, *x2*) on latent space *Z* are generated.
![Alt text](image/this_implementation_results/test-interpolation.png?raw=true)

#### Random-Sampling
A latent code *z* is sampled from a fixed prior distribution on a latent space *Z*. Then, *z* is mapped to the image *x* on input space *X*.
![Alt text](image/this_implementation_results/random-sampling.png?raw=true)

### Quantitative Results
FID is calculated using 1K samples.  
Fréchet Inception Distance (FID) =  99.75676458019791


This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models. The re-implementation is provided by:

* Ali Abbasi, ali.abbasi@metu.edu.tr
* Samet Cetin, cetin.samet@metu.edu.tr


Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.

### Installation Instructions 
Execute following command to install requirements:
```
$ pip install -r requirements.txt
```
Execute following command to download pretrained encoder and decoder weights into *checkpoint/wae-mmd/* directory:
```
$ bash download_data.sh
```


