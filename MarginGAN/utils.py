import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize(image):
    """ Creates a (batch_size//10)x10 sample image grid. 
    """
    image = make_grid(image,nrow=10)
    npimg = image.detach().cpu().numpy()
    npimg = (np.transpose(npimg, (1,2,0)) + 1)/2
    mpl.rcParams['savefig.pad_inches'] = 0
    frame = plt.gca()
    plt.axis('off')
    plt.imshow(npimg, interpolation='nearest')

def create_dirs(job_id):
    """ Creates necessary directories.
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    if not os.path.exists('logs/'+str(job_id)):
        os.mkdir('logs/'+str(job_id))
        
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
        
    if not os.path.exists('imgs/'+str(job_id)):
        os.mkdir('imgs/'+str(job_id))
        
    if not os.path.exists('models'):
        os.mkdir('models')
        
    if not os.path.exists('models/'+str(job_id)):
        os.mkdir('models/'+str(job_id))

def param_log(params):
    """ Logs the hyperparameters of the model.
    """
    f = open('logs/'+str(params['job_id'])+'/params.txt','w')
    for k, v in params.items():
        f.write(str(k) + ':'+ str(v) + '\n')
    f.close()

