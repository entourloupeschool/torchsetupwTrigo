import torch

def make_data( nb_samples, width_samples, noise_amp ):
    # torch tensor with rows rows and 2 columns
    # filled with random numbers between -width_samples and width_samples
    x_gen = torch.rand(nb_samples, 2) * width_samples * 2 - width_samples
    xcos = torch.cos(torch.select(x_gen, 1, 0)).view(nb_samples, 1)
    xsin = torch.sin(torch.select(x_gen, 1, 1)).view(nb_samples, 1)
    xtan = torch.tanh(torch.select(x_gen, 1, 0)).view(nb_samples, 1)
    x = torch.cat((x_gen, xcos, xsin, xtan), dim=1)
    
    # xb torch tensor with cosinus of first column of xa and sinus of second column of xa
    y = torch.cat((xcos, xsin), dim=1)
    
    # product of the two columns of xb
    y = torch.prod(y, dim=1).view(nb_samples, 1) 

    # add tan
    y = torch.add(y, xtan).view(nb_samples, 1)
    
    # noise
    noise = torch.randn(nb_samples, 1) * noise_amp
    
    # sum of xb and noise
    y = torch.add(y, noise).view(nb_samples, 1)
    
    return x, y