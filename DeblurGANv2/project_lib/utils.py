import numpy as np
import matplotlib.pyplot as plt
import torch
import project_lib.eval_metrics as eval_metrics
import project_lib.dataset.ImageDirectory as ImageDirectory

# Utility functions to plot loss curves, visualize the output images and
# Calculating the metrics for performance evaluation of the GAN model.
def loss_plot(disc_loss, gen_loss, name):
    # Loss plots
    fig = plt.figure(figsize=(8, 5))
    colors = "bgrcmykbgrcmyk"
    c_ind = 0 
    plt.plot(disc_loss, color=colors[c_ind], label= name + ' Disc loss')
    plt.xlabel('Epochs')
    plt.ylabel('Disc_Losses')
    plt.grid()
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(gen_loss, color=colors[c_ind+1],  label= name +' Gen loss')
    plt.xlabel('Epochs')
    plt.ylabel('Gen_Losses')
    plt.grid()
    plt.legend()
    plt.show()

def tensor_to_np_img(tens_img):
    """Takes in tensor image and converts to np image"""
    np_img = tens_img.squeeze().detach().cpu().numpy()
    np_img = (np.transpose(np_img, (1, 2, 0)) + 1)* 127.5
    np_img = np_img.astype('uint8')
    return np_img  
    
def visOutputs(model,cpt_path, test_loader, num_vis):
    model.load_state_dict(torch.load(cpt_path))
    counter = 0
    for image_s, image_b in test_loader:
        gen_im = model.gen(image_b.cuda())
        counter += 1

        f, axarr = plt.subplots(1,3,figsize=(30,30))

        axarr[0].imshow(tensor_to_np_img(image_s.cuda()))
        axarr[0].set_xlabel("GT sharp")
        axarr[1].imshow(tensor_to_np_img(image_b.cuda()))
        axarr[1].set_xlabel("GT blurred")
        axarr[2].imshow(tensor_to_np_img(gen_im))
        axarr[2].set_xlabel("Reconstructed sharp")
        plt.show()
        if counter == num_vis:
          break
        
def calculateMetrics(model,model_path,test_loader):

    model.load_state_dict(torch.load(model_path))

    PSNR = 0
    SSIM = 0

    for image_s, image_b in test_loader:
        gen_im = model.gen(image_b.cuda())
        
        gen_im_np = tensor_to_np_img(gen_im)
        image_s_np = tensor_to_np_img(image_s)

        PSNR += eval_metrics.PSNR(gen_im_np,image_s_np)
        SSIM += eval_metrics.SSIM(gen_im_np,image_s_np)
    print(PSNR / len(test_loader))
    print(SSIM / len(test_loader))        