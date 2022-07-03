import torch
import torchvision
"""
Implementation of the Evaluation Metrics from the paper:
"""

class EvaluationMetrics():

    def l1_distance(x,y):
        """
        Calculates the L1 distance between two tensors.
        """
        return torch.mean(torch.abs(x - y))
        
    def fid_score(x,y,device = 'cpu'):
        """
        Calculates the FID score
        """

        inception_v3 = torchvision.models.inception_v3(pretrained=True, transform_input=False).to(device)
        inception_v3.eval()
        x = x.to(device)
        y = y.to(device)

        
        activation_x = inception_v3(x)
        activation_y = inception_v3(y)
        
        mu_x = activation_x.mean(dim=0)
        mu_y = activation_y.mean(dim=0)
        
        sigma_x = activation_x.var(dim=0)
        sigma_y = activation_y.var(dim=0)

        ssdif = torch.sum((mu_x - mu_y) ** 2)
        covmean = torch.mean(sigma_x.view(1, -1) * sigma_y.view(1, -1))

        fid = ssdif + torch.trace(sigma_x.view(1, -1) + sigma_y.view(1, -1) - 2 * covmean)
        return fid



    def psnr(x,y):
        """
        Calculates the PSNR between generated image and .
        """
        mse = torch.mean((x - y) ** 2)
        score = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return score

if __name__ == '__main__':
    x = torch.randn(2,3,256,256)
    y = torch.randn(2,3,256,256)
    print(EvaluationMetrics.fid_score(x,y))
    print(EvaluationMetrics.psnr(x,y))
    print(EvaluationMetrics.l1_distance(x,y))