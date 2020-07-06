import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from utils import *

class Generator(nn.Module):
    """ Generator class.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(62,1024)
        self.fc2 = nn.Linear(1024,7*7*128)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128*7*7)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x): 
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = x.view(-1,128,7,7)
        x = self.upconv1(x)
        x = F.relu(self.bn3(x))
        x = self.upconv2(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    """ Discriminator class.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(5*5*128,1024)
        self.fc2 = nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),0.1)
        x = F.leaky_relu(self.bn1(self.conv2(x)),0.1)
        x = x.view(-1,5*5*128)
        x = F.leaky_relu(self.bn2(self.fc1(x)),0.1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Classifier(nn.Module):
    """ Classifier class.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(5*5*64,512)
        self.fc2 = nn.Linear(512,10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1,5*5*64)
        x = self.fc2(F.relu(self.fc1(x)))
        x_soft = F.softmax(x,dim=1)
        return x, x_soft

def inverted_cross_entropy_loss(pred,target,device,eps=1e-6):
    """ Inverted cross-entropy loss defined in the paper.
    Args:
        pred (torch.Tensor): Predicted softmax probabilities for each class (batch_size, num_classes).
        target (torch.Tensor): Target classes.
    Returns:
        loss (float): Loss value calculated over one-hot predicted classes and target classes.
    """
    one_hot = torch.zeros(pred.shape)
    one_hot = one_hot.to(device)
    for i in range (pred.shape[0]):
        one_hot[i,target[i]] = 1
        
    loss = torch.mean(-torch.sum(torch.mul(one_hot,(torch.log(1-pred+eps))),dim=1))
    return loss

class MarginGAN(object):
    """ MarginGAN class.
    Args:
        label_size (int, optional): Number of labeled examples in semi-supervised learning.
        batch_size (int, optional): Number of batches in 1 epoch.
        device (str, optional): Device name.
        lrs (list or tuple, optional): Learning rates of discriminator, generator and classifier, respectively.
        beta_1 (float, optional): beta_1 parameter of Adam optimizer of discriminator and generator.
        beta_2 (float, optional): beta_2 parameter of Adam optimizer of discriminator and generator.
        momentum (float, optional): momentum parameter of SGD optimizer of classifier.
        pretrained (str, optional): File path of the pretrained classifier model.
    """
    def __init__(self, label_size=100, batch_size=50, device='cpu', lrs=[0.0002,0.0002,0.01], beta_1=0.5, beta_2=0.999, momentum=0.8, pretrained=None):
        super(MarginGAN, self).__init__()
        
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)
        self.C = Classifier().to(device)
        if pretrained:
            self.C.load_state_dict(torch.load(pretrained).state_dict())

        self.C.train()
        self.D.train()
        self.G.train()
        
        # Define real/fake labels
        self.real_img = 1
        self.fake_img = 0

        # optimizer for D
        self.optimD = optim.Adam(self.D.parameters(),lr=lrs[0],betas=(beta_1,beta_2))
        # optimizer for G
        self.optimG = optim.Adam(self.G.parameters(),lr=lrs[1],betas=(beta_1,beta_2))
        # optimizer for C
        self.optimC = optim.SGD(self.C.parameters(),lr=lrs[2],momentum=momentum)

        # losses
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ice = inverted_cross_entropy_loss

        self.device = device
        self.batch_size = batch_size
        self.label_size = label_size
    
    def train(self,data_loader,epoch,log_every):
        """ Training code of MarginGAN for 1 epoch.
        Args:
            data_loader (torch.utils.data.DataLoader): Dataloader containing unlabeled and labeled samples.
            epoch (int): Index of the current epoch.
            log_every (int): Log frequency.
        """
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_C = 0.0
        # Go over the dataset by cycling over the labeled examples along with the unlabeled examples
        for i, ((labeled_image,real_label), (unlabeled_image,_)) in enumerate(data_loader,0):
            # Sample noise from a uniform distribution
            noise = torch.rand(self.batch_size,62).to(self.device)
            
            # Generate an image
            gen_image = self.G(noise)
            
            # Create pseudolabels for the generated image
            pseudo_gen_out, _ = self.C(gen_image)
            pseudo_gen_label = torch.argmax(pseudo_gen_out.data, 1).to(self.device)
            
            # GPU support
            labeled_image = labeled_image.to(self.device)
            unlabeled_image = unlabeled_image.to(self.device)
            real_label = real_label.to(self.device)
            
            # Create pseudolabels for the unlabeled image
            pseudo_ul_out, _ = self.C(unlabeled_image)
            pseudo_ul_label = torch.argmax(pseudo_ul_out.data, 1).to(self.device)
            
            ##############
            # Classifier #
            ##############
            self.optimC.zero_grad()
            est_labeled, _ = self.C(labeled_image)
            _ ,est_gen_softmax = self.C(gen_image)
            est_gen_label = torch.max(est_gen_softmax,1)[1]
            est_unlabeled, _ = self.C(unlabeled_image)
            lossC = self.ce(est_unlabeled,pseudo_ul_label) + self.ce(est_labeled,real_label) + self.ice(est_gen_softmax,est_gen_label,self.device)
            lossC.backward()
            self.optimC.step()
            
            #################
            # Discriminator #
            #################
            self.optimD.zero_grad()
            gen_image = self.G(noise)
            r_label = torch.full((self.batch_size,),self.real_img,device=self.device)
            f_label = torch.full((self.batch_size,),self.fake_img,device=self.device)
            labeled_output = self.D(labeled_image)
            gen_output = self.D(gen_image)
            unlabeled_output = self.D(unlabeled_image)
            lossD = self.bce(unlabeled_output,r_label) + self.bce(gen_output,f_label) + self.bce(labeled_output,r_label)
            lossD.backward()
            self.optimD.step()
            
            #############
            # Generator #
            #############
            self.optimG.zero_grad()
            gen_image = self.G(noise)
            gen_output = self.D(gen_image)
            pseudo_gen_out, _ = self.C(gen_image)
            pseudo_gen_label = torch.argmax(pseudo_gen_out.data, 1).to(self.device)
            lossG = self.bce(gen_output,r_label) + self.ce(pseudo_gen_out,pseudo_gen_label)
            lossG.backward()
            self.optimG.step()

            running_loss_G += lossG.item()
            running_loss_D += lossD.item()
            running_loss_C += lossC.item()

            # Log at every log_every steps
            if (i+1) % log_every == 0:
                print('[%d, %5d] loss_G: %.3f' %
                    (epoch + 1, i + 1, running_loss_G / log_every))
                running_loss_G = 0.0
                print('[%d, %5d] loss_D: %.3f' %
                    (epoch + 1, i + 1, running_loss_D / log_every))
                running_loss_D = 0.0
                print('[%d, %5d] loss_C: %.3f' %
                    (epoch + 1, i + 1, running_loss_C / log_every))
                running_loss_C = 0.0

    def load(self,load_model):
        """ Load model.
        Args:
            load_model (list or tuple): Models to load, classifier, discriminator, and generator, respectively.
        """
        self.C = torch.load(load_model[0],map_location=self.device)
        self.D = torch.load(load_model[1],map_location=self.device)
        self.G = torch.load(load_model[2],map_location=self.device)

    def eval(self,dataloader):
        """ Evaluation code of MarginGAN.
        Args:
           dataloader (torch.utils.data.DataLoader): Test dataset.
        Returns:
            correct (int): Number of correctly classified samples.
        """
        correct=0
        total=0
        self.C.eval()
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.C(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.C.train()
        
        return correct


    def save(self,epoch,fixed_noise,job_id):
        """ Code to save the MarginGAN model.
        Args:
            epoch (int): Index of the current epoch.
            fixed_noise (torch.Tensor): Fixed noise to generate image.
            job_id (int): Index for the current experiment.
        """
        torch.save(self.C,"models/"+str(job_id)+"/C_epoch_"+str(epoch+1)+"_label_"+str(self.label_size)+".pt")
        torch.save(self.D,"models/"+str(job_id)+"/D_epoch_"+str(epoch+1)+"_label_"+str(self.label_size)+".pt")
        torch.save(self.G,"models/"+str(job_id)+"/G_epoch_"+str(epoch+1)+"_label_"+str(self.label_size)+".pt")
        
        gen_img = self.G(fixed_noise)
        save_image(gen_img,'imgs/'+str(job_id)+'/img_epoch_'+str(epoch+1)+'_label_'+str(self.label_size)+'.png',nrow=10)
        
    def print_model(self):
        print(self.C)
        print(self.D)
        print(self.G)
        
    def imshow(self,fixed_noise):
        """ Code to visualize one batch of images with MarginGAN model.
        Args:
            fixed_noise (torch.Tensor): Fixed noise to generate image.
        """
        gen_img = self.G(fixed_noise)
        visualize(gen_img)
