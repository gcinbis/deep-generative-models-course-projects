import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import random

class ScoreSamples():
    """
    Keeps the scores in lists of a predefined size to take inner expectations in the 
    Discriminator and generator adverserial losses.
    """    
    def __init__(self, maxSize):
        self.score_b_List = list()
        self.score_s_List = list()
        self.maxSize = maxSize
        self.numsamples = 0
    def addSamp(self, scores_b_, scores_s_ ):      
        scores_b = torch.unsqueeze(scores_b_, 0)
        scores_s = torch.unsqueeze(scores_s_, 0)
        if self.numsamples < self.maxSize:
            self.score_b_List.append(scores_b)
            self.score_s_List.append(scores_s)
            self.numsamples = self.numsamples + 1
        else: 
            self.score_b_List.pop(0)
            self.score_b_List.append(scores_b)
            
            self.score_s_List.pop(0)
            self.score_s_List.append(scores_s)
    def getSamp(self):
      if self.numsamples == self.maxSize:
          samples_b = random.sample(self.score_b_List, self.maxSize)
          samples_s = random.sample(self.score_s_List, self.maxSize)
      else:    
          samples_b = self.score_b_List
          samples_s = self.score_s_List
      samples_s = torch.cat(samples_s).detach()  
      samples_b = torch.cat(samples_b).detach()
      return samples_s, samples_b

class VGGContent(nn.Module):
    """
    Returns the layers of pretrained VGG19 until the relu4_4 layer
    This network outputs will be used to  compare the content between 
    deblurred and sharp (real)  images as proposed in :

    Johnson et. al. "Perceptual losses for real-time style transfer and super-resolution." (ECCV 2016)
    """
    def __init__(self):
        super(VGGContent, self).__init__()
        
    def get_net(self):
    
        # Get the pretrained VGG19 model  
        vgg19 = models.vgg19(pretrained=True)
        vgg19 = vgg19.cuda()

        # We need only the first 11 layers for the output of relu4_3 
        cont_net = nn.Sequential(*vgg19.features[0:25])
        cont_net = cont_net.cuda()
        cont_net = cont_net.eval()
        return cont_net 

def discriminator_loss(sharp_scores, deblur_scores, SampleScores):    
    """
    Computes the DoubleScale RaGANLS Discriminator Loss.
    LRaLSGAN definition= Ex[(D(x)- Ez[ D(G(z))]-1 )^2] + Ez[(D(G(z))- Ex[ D(x)]+1 )^2]
    """     
    # Pass the fake(deblurred) images generated by Generator to the discriminator to fake it
    #deblur_scores = GANmodel.forward(deblur_images.detach())

    # Pass the real(sharp) images to the discriminator
    #sharp_scores = GANmodel.forward(sharp_images)
    
    # Compute the discriminator loss using LRaLSGAN formulation 
    
    SampleScores.addSamp(deblur_scores, sharp_scores)
    sharp_samples, deblur_samples = SampleScores.getSamp()
    
    lossDisc = torch.mean((sharp_scores - torch.mean(deblur_samples) - 1).pow(2) ) + torch.mean((deblur_scores - torch.mean(sharp_samples) + 1).pow(2))
    
    return lossDisc


def perceptual_loss(sharp_images, deblur_images):
    """
    Computes the Perceptual Loss to compare the 
    reconstructed (deblurred) and the original(sharp) images
    """     
    #Measures the L2 distance between generated and original
    loss = nn.MSELoss()
    lossP = loss(sharp_images, deblur_images)
    return lossP
     
    
    
def content_loss(sharp_images, deblur_images, cont_net):
    """
    Computes the Content Loss to compare the 
    reconstructed (deblurred) and the original(sharp) images

    Takes the output feature maps of the relu4_3 layer of pretrained VGG19 to compare the content between 
    images as proposed in :
    Johnson et. al. "Perceptual losses for real-time style transfer and super-resolution." (ECCV 2016) 
    """    
    
    # Torchvision models documentation:
    # All pre-trained models expect input images normalized in the same way, The images have to be loaded in           
    # to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
    deblur_images = (deblur_images + 1) * 0.5
    sharp_images = (sharp_images + 1) * 0.5
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    deblur_images = normalize(deblur_images)
    sharp_images= normalize(sharp_images)
    
    content_deblur = cont_net(deblur_images)
    content_sharp = cont_net(sharp_images)
    content_sharp = content_sharp.detach()
    
    loss = nn.MSELoss()
    lossC = torch.mean(loss(content_deblur,content_sharp))
    
    return lossC


def generator_loss_adv(sharp_scores, deblur_scores, SampleScores):
    """
    Computes the DoubleScale RaGANLS Generator Adverseraial Loss
    """      
    
    # Compute the generator loss using
    # Paper: 
    # The relativistic discriminator: a key element missing from standard gan. arXiv preprint arXiv:1807.00734, 2018.
    # LRaLSGAN definition for generator:
    # Ez[(D(G(z))- Ex[ D(x)]-1 )^2] + Ex[(D(x)- Ez[ D(G(z))]+1 )^2]
    
    
    sharp_samples, deblur_samples = SampleScores.getSamp()
    
    Ladv = torch.mean((deblur_scores - torch.mean(sharp_samples) - 1).pow(2))+ torch.mean((sharp_scores - torch.mean(deblur_samples) + 1).pow(2))
                
    lossGenAdv = 0.01* Ladv
    
    return lossGenAdv

def generator_loss_cont(sharp_images, deblur_images, cont_net):
    """
    Computes the 
    Content Loss + Perceptual Loss for generator 
    """      
 
                
    # Perceptual Loss at the output of the generator
    # pixel-space loss LP , e.g.,the simplest L2 distance
    Lp = perceptual_loss(sharp_images, deblur_images)
    
    # Content Loss at the output of the generator
    # In contrast to the L2, it computes the
    # Euclidean loss on the VGG19 conv3 3 feature maps.
    Lx = content_loss(sharp_images, deblur_images, cont_net)
       
    # Overall Gen Loss= 0.5*Lp + 0.006*Lx + 0.01* Ladv
    lossGenCont = 0.5*Lp + 0.006*Lx 
    
    return lossGenCont