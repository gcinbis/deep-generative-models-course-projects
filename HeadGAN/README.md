# HeadGAN


The paper that was implemented is "HeadGAN: One-shot Neural Head Synthesis and Editing".

It could be found at: https://arxiv.org/pdf/2012.08261.pdf

Code Authors: Ahmet Taha ALBAYRAK, Abdüllatif AĞCA

Implementation details: In the original work, a GAN consisting of AdaIN and SPADE layers with multiple
discriminators was used to produce high quality facial transformation between two images. This being
applied to a sequence of frames, a face, its orientation and mimics in a video is transferred to a static
image. To achieve this, the original work used 4 separate helper libraries (3 of them is neural) to extract features.

These are as follows:

1) Retinaface: To extract the facial landmarks and dense point cloud, the authors used this as a feature extractor.
   Although it successfully extracts landmarks, the version that is shared online cannot extract the dense point cloud.
   We failed to find that version just like many other researcher failed to do so, according to the forums discussions at
   github.
2) LSFM: This library is used to align one face dense point cloud to another face's point cloud. This is required for
   facial transfer but since we failed to find Retinaface's related version, we used another library instead of these two combined.

3) 3DDFA V2: Because of the lack of Retinaface point cloud extraction, we decided to do it with the help of another library,
which is 3DDFA V2. With the help of this library, facial alignment and shape feature extraction (transformation, expression, shape)
is done too. After extracting the related features, related point clouds were rendered as PNCC (Projected Normalized Coordinate Code),
which is essentially transforming all the vertices to fit inside [0, 1] range in all x, y and z axes. Then, these coordinates are
rendered as R, G and B. The final RGB image contains information about a face's 3D features.

4) pyAudioAnalysis: A library to extract features of audio files, just like energy, entropy, MFCC etc.
5) Deep Speech 2.0: A library to extract the words from an audio file.

## Requirements

The libraries can be installed via pip or conda

`pip install -r requirements.txt`

## Download Data

The dataset can be downloaded from the following link: [VoxCeleb2](https://mm.kaist.ac.kr/datasets/voxceleb/#downloads) 

In order to access the files you must fill the request form in the link above. After getting username and password, change the username and password in the `download_data.sh` file.

This script downloads only test data. To access the train data, you must download the train data from the link above.

## Pretrained Model

Pretrained model can be downloaded by using the below
[model.zip](https://drive.google.com/file/d/14N6A2KACdzHIgCoD0Zj7vVI88VlmnxAd/view?usp=sharing)
