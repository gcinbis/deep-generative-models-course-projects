import torch 


def h_cossim(x, y, row_wise = False):
    ''' Calculates Cosine Similarity '''
    num = torch.matmul(x, y.t())
    denum = torch.max(torch.outer(torch.linalg.norm(x, dim=1), torch.linalg.norm(y, dim=1)), torch.tensor(1e-8))
    return num/denum


def contrastive_loss(z_latent, x, f_z, expectation = True, beta = 1e-6):
    ''' Contrastive Loss, InfoNCE + KLD

    Args:
        z_latent : [logvar, mu] output of encoder network, variance and mean: [batch_size, 2*latent_size]
        x        : original inputs, [batch_size, cont_dim]
        f_z      : reconstructed images, [batch_size, cont_dim]
    
    Return:
        contrastive loss + KLD
    '''

    logvar, mu = z_latent.chunk(2, dim=1)    
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2), dim=1)

    distances = h_cossim(x, f_z).exp()
    positive_samples = torch.diag(distances)        # diagonal elements of dist are positive pairs.
    negative_samples = torch.sum(distances, dim=0)  # sum of columns gives the union of positive and negative samples

    l_instance =  -torch.log(torch.div(positive_samples, negative_samples)) # torch.nn.MSELoss()(x, f_z )

    if expectation:
        l_instance = torch.mean(l_instance)
        KLD = KLD.mean()
    
    return l_instance + KLD*beta



