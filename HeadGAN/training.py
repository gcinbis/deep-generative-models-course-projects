import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

import numpy as np
from headgan import HeadGAN
from loss import *
from misc.scores import EvaluationMetrics
import os 
import tqdm


def evaluate_model(model, test_loader, device, n_test_samples=None):
    """
    Evaluate the model on the test set
    """
    model.eval()
    with torch.no_grad():
        if n_test_samples is None:
            n_test_samples = len(test_loader)
        
        eval_scores = {'psnr': 0, 'fid': 0, 'l1': 0}
        for i, (data, labels, unnormalized) in enumerate(test_loader):
            if i >= n_test_samples:
                break
            ref_image = data[0].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            ref_3d_face = data[1].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            face_3d = data[2].reshape(-1, 3, 256, 256, 3).permute(0, 1, 4, 2, 3).to(device)
            audio_features = data[3].reshape(-1,1,300).to(device)
            labels = labels.reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)

            # Forward pass
            genearated_image, losses = model(ref_image, ref_3d_face, face_3d, audio_features, labels)

            eval_scores['psnr']  += EvaluationMetrics.psnr(genearated_image, labels)
            eval_scores['fid'] += EvaluationMetrics.fid_score(genearated_image, labels)
            eval_scores['l1'] += EvaluationMetrics.l1_distance(genearated_image, labels)

        eval_scores['psnr'] /= n_test_samples
        eval_scores['fid'] /= n_test_samples
        eval_scores['l1'] /= n_test_samples
    return eval_scores


def training_loop(train_dataset, test_dataset, batch_size, epochs, step_size, lr, beta1, beta2, use_cuda, log_dir, model_dir, model_name, num_workers=4, checkpoint_path=None, checkpoint_step=None, checkpoint_epoch = None): 
    """
    Training function for the HeadGAN model.
    """
    # Create SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)


    # Create the model
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = HeadGAN().to(device)
    model = nn.DataParallel(model)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load("{}_epoch{}_step{}.pth".format(checkpoint_path, checkpoint_epoch, checkpoint_step)))
    
    # Create the optimizer
    optimizer_F = optim.Adam(model.module.F.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_R = optim.Adam(model.module.R.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(model.module.D.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizer_Dm = optim.Adam(model.module.Dm.parameters(), lr=lr, betas=(beta1, beta2))

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    
    start_epoch = checkpoint_epoch +1 if checkpoint_epoch is not None else 0

    
    print("Start training")
    # Start training
    for epoch in range(start_epoch, epochs):

        # Train the model
        model.train()
        for i, (data, labels, original) in enumerate(train_loader):
            
            # Move the data to the GPU
            ref_image = data[0].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            ref_3d_face = data[1].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            face_3d = data[2].reshape(-1, 3, 256, 256, 3).permute(0, 1, 4, 2, 3).to(device)
            audio_features = data[3].reshape(-1,1,300).to(device)
            labels = labels.reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)

            # Clear the gradients
            optimizer_F.zero_grad()
            optimizer_R.zero_grad()
            optimizer_D.zero_grad()
            # optimizer_Dm.zero_grad()
            # Forward pass

            genearated_image, losses = model(ref_image, ref_3d_face, face_3d, audio_features, labels)
            # Compute the loss
            loss_g = losses['L_G']
            loss_d = losses['L_D']
            # loss_d_m = losses['L_D_m']
            # Backward pass
            loss_g.sum().backward(retain_graph=True)
            loss_d.sum().backward(retain_graph=True)
            # loss_d_m.sum().backward(retain_graph=True)
            # Update the weights
            optimizer_F.step()
            optimizer_R.step()
            optimizer_D.step()
            # optimizer_Dm.step()
            
            # Print the losses
            writer.add_scalar('Training/Loss/L_G', loss_g.sum().item(), epoch*step_size+i)
            writer.add_scalar('Training/Loss/L_D', loss_d.sum().item(), epoch*step_size+i)
            # writer.add_scalar('Loss/L_D_m', loss_d_m.sum().item(), epoch*len(train_loader)+i)
            
            

            # Print the loss
            if i % 10 == 0:
                print("Training Epoch: {}, Batch: {}, Loss_G: {}, Loss_D: {}, Loss_D_m: {}".format(epoch, i, loss_g, loss_d, 0))
            if i % 100 == 0:        
                writer.add_images('Image/Generated', genearated_image, epoch*step_size+i)
                writer.add_images('Image/3D Face', ref_3d_face, epoch*step_size+i)
                writer.add_images('Image/Reference', ref_image, epoch*step_size+i)
                writer.add_images('Image/Ground Truth', labels, epoch*step_size+i)
                torch.save(model.state_dict(), os.path.join(model_dir, model_name+'_epoch{}_step{}.pth'.format(epoch, epoch*step_size+i)))   

            if i == step_size:
                break

                # Test the model
        
        eval_loss = {'L_G': 0, 'L_D': 0, 'L_D_m': 0}
        eval_scores = {'psnr': 0, 'fid': 0, 'l1': 0}
        model.eval()
        for i, (data, labels, original) in enumerate(test_loader):
            # Move the data to the GPU
            if i == step_size/20:
                break
            ref_image = data[0].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            ref_3d_face = data[1].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)
            face_3d = data[2].reshape(-1, 3, 256, 256, 3).permute(0, 1, 4, 2, 3).to(device)
            audio_features = data[3].reshape(-1,1,300).to(device)
            labels = labels.reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).to(device)

            # Forward pass
            genearated_image, losses = model(ref_image, ref_3d_face, face_3d, audio_features, labels)

            # Compute the loss
            loss_g = losses['L_G']
            loss_d = losses['L_D']
            #loss_d_m = losses['L_D_m']
            
            eval_loss['L_G'] += loss_g.sum().detach().cpu().numpy()
            eval_loss['L_D'] += loss_d.sum().detach().cpu().numpy()
            #eval_loss['L_D_m'] += loss_d_m.sum().item()

            del loss_g
            del loss_d
            
            eval_scores['psnr']  += EvaluationMetrics.psnr(genearated_image, labels)
            eval_scores['fid'] += EvaluationMetrics.fid_score(genearated_image, labels)
            eval_scores['l1'] += EvaluationMetrics.l1_distance(genearated_image, labels)



        eval_loss['L_G'] /= step_size
        eval_loss['L_D'] /= step_size
        #eval_loss['L_D_m'] /= step_size
        eval_scores['psnr'] /= step_size
        eval_scores['fid'] /= step_size
        eval_scores['l1'] /= step_size

        writer.add_scalar('Eval/Loss/L_G_eval', eval_loss['L_G'], epoch)
        writer.add_scalar('Eval/Loss/L_D_eval', eval_loss['L_D'], epoch)
        writer.add_scalar('Eval/Loss/L_D_m_eval', eval_loss['L_D_m'], epoch)
        writer.add_scalar('Scores/PSNR_eval', eval_scores['psnr'], epoch)
        writer.add_scalar('Scores/FID_eval', eval_scores['fid'], epoch)
        writer.add_scalar('Scores/L1', eval_scores['l1'], epoch)

        # Print the loss
        print("Evaluation : Epoch: {}, Loss_G: {}, Loss_D: {}, Loss_D_m: {}".format(epoch, eval_loss['L_G'], eval_loss['L_D'], 0))
    
                
       