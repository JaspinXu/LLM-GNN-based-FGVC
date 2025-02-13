import torch
import torch.nn as nn
from infonce import InfoNCE

class CATE(nn.Module):
    def __init__(self, input_dim=512, ib_dim=512, interv_dim=512):
        super(CATE, self).__init__()


        self.encoder_IB = nn.Sequential(
            nn.Linear(input_dim, ib_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ib_dim, ib_dim*2),
        ).to("cuda")


        self.infonce_loss = InfoNCE()
        self.mse_loss = torch.nn.MSELoss()

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean
    def ib(self, x,p,n):
        p_c = torch.mean(p, dim=1)
        n_c = n
        x_re = self.encoder_IB(x)
        x_re = x_re.squeeze(1)
        mu, logvar = x_re.chunk(2, dim=-1)
        kl_loss = self.kl_loss(mu, logvar)
        x_re = self.reparameterise(mu, logvar)
        x_re,p_c,n_c =x_re.float(),p_c.float(),n_c.float()
        info_loss = self.infonce_loss(x_re,p_c,n_c)

        return x_re,kl_loss,info_loss
