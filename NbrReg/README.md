# [Deep Semantic Text Hashing with Weak Supervision](https://dl.acm.org/doi/pdf/10.1145/3209978.3210090)


Suthee Chaidaroon, Travis Ebesu and Yi Fang


*SIGIR2018*


This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models. The re-implementation is provided by:

* Erman Yafay, erman.yafay@ceng.metu.edu.tr


Please see the jupyter notebook file [main.ipynb](main.ipynb) for a summary of paper, the implementation notes and our experimental results.
**Optional**
*Download raw data (this includes the data and trained models for the dataset provided by the original authors), process and obtain the BM25 weighted document vectors, then train and test the model*

```bash
# Download data
./download_data.sh

# Create NewsGroups20 data into ng20_test.mat
./prepare_data.py -i 20ng-all-stemmed.txt -o ng20_test.mat

# Train an 32 bit model with early stopping and save the model into ng20_test.pt
./nbrreg.py -b 32 train -i 3 ng20_test.mat ng20_test.pt

# Test the model on test split and output average precision
./nbrreg.py -b 32 ng20_test.pt
```
