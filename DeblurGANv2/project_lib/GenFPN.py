import torch
import torch.nn as nn
import torch.nn.functional as F
import MobileNetV2_my 


#create global discriminator, output size 1x1
class Discriminator_Global(nn.Module):
   def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,128,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2)) #128x128x128
        self.block2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2)) #64x64x64
        self.block3 = nn.Sequential(nn.Conv2d(64,32,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2))   #32x3232
        self.block4 = nn.Sequential(nn.Conv2d(32,16,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2))  #16x16x16
        self.block5 = nn.Sequential(nn.Conv2d(16,8,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2))  #8x8x8
        self.block6 = nn.Sequential(nn.Conv2d(8,4,kernel_size=4, stride=2,padding =1), nn.LeakyReLU(0.2))  #4x4x4
        self.block7 = nn.Sequential(nn.Conv2d(4,1,kernel_size=4, stride=2,padding =0), nn.LeakyReLU(0.2))  #1x1x1
   def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        return F.sigmoid(out)

#create local discriminator, output size 70x70
class Discriminator_Local(nn.Module):
   def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,128,kernel_size=4 ,stride=2,padding=1), nn.LeakyReLU(0.2)) 
        self.block2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=4, stride=2,padding=1), nn.LeakyReLU(0.2)) 
        self.block3 = nn.Sequential(nn.Conv2d(64,32,kernel_size=4, stride=2,padding=1), nn.LeakyReLU(0.2))   
        self.block4 = nn.Sequential(nn.Conv2d(32,16,kernel_size=4, stride=1,padding =1), nn.LeakyReLU(0.2)) 
        self.block5 = nn.Sequential(nn.Conv2d(16,1,kernel_size=4, stride=1,padding =1), nn.LeakyReLU(0.2))  


   def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return F.sigmoid(out)


#create Generator
class FPNMobileNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        net = MobileNetV2_my.MobileNetV2()
        features = net.features

        self.enc0 = nn.Sequential(*features[0:2])  
        self.enc1 = nn.Sequential(*features[2:4])
        self.enc2 = nn.Sequential(*features[4:7])
        self.enc3 = nn.Sequential(*features[7:11])
        self.enc4 = nn.Sequential(*features[11:16])

        self.lateral0 = nn.Conv2d(16, 128, kernel_size=1)
        self.adition0 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.conv0 = nn.Conv2d(128, 32, kernel_size=3,padding=1)

        self.lateral1 = nn.Conv2d(24, 128, kernel_size=1)
        self.adition1 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(128, 32, kernel_size=3,padding=1)

        self.lateral2 = nn.Conv2d(32, 128, kernel_size=1)
        self.adition2 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3,padding=1)

        self.lateral3 = nn.Conv2d(64, 128, kernel_size=1)
        self.adition3 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3,padding=1)

        self.conv4_1 = nn.Conv2d(128, 32, kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(64, 32, kernel_size=2)

        self.lateral4 = nn.Conv2d(160, 128, kernel_size=1)

        self.LastConv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.LastConv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.LastConv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return torch.nn.functional.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):

        # get encoding in different scale
        enc0 = self.enc0(x) # 16

        enc1 = self.enc1(enc0) # 24

        enc2 = self.enc2(enc1) # 32
   
        enc3 = self.enc3(enc2) # 64

        enc4 = self.enc4(enc3) # 160

        # get lateral in different scale
        lat0 = self.lateral0(enc0) # 16

        lat1 = self.lateral1(enc1) # 24

        lat2 = self.lateral2(enc2) # 32
   
        lat3 = self.lateral3(enc3) # 64

        lat4 = self.lateral4(enc4) # 96

        #get top-down enc
        e4 = lat4
        e3 = F.relu(self.adition0(self._upsample_add(e4,lat3)))
        e2 = F.relu(self.adition1(self._upsample_add(e3,lat2)))
        e1 = F.relu(self.adition2(self._upsample_add(e2,lat1)))
        e0 = F.relu(self.adition3(self._upsample_add(e1,lat0)))

        #upsample & concat top-down encoding
        
        e4 = F.relu(self.conv4_1(e4))

        #e4 = F.relu(self.conv4_2(e4))
        #print(e4.size())
        e4 = torch.nn.functional.interpolate(e4, scale_factor=8)
        e3 = torch.nn.functional.interpolate(F.relu(self.conv3(e3)), scale_factor=4)
        e2 = torch.nn.functional.interpolate(F.relu(self.conv2(e2)), scale_factor=2)
        e1 = F.relu(self.conv1(e1))
        
        #concat layers
        concat = torch.cat([e4,e3,e2,e1],dim = 1)

        last_vec = torch.nn.functional.interpolate(F.relu(self.LastConv1(concat)), scale_factor = 2)

        last_vec = torch.nn.functional.interpolate(F.relu(self.LastConv2(last_vec + e0)), scale_factor = 2)
        
        # add input image
        gen_im = F.tanh(self.LastConv3(last_vec)) + x
        
        # set output range
        gen_im = torch.clamp(gen_im,min = -1, max = 1)


        return gen_im

# create whole GAN architecture
class GAN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.gen = FPNMobileNet(pretrained=True)
        self.localD =  Discriminator_Local()
        self.globD = Discriminator_Global()

