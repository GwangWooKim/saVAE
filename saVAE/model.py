import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class VAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        self.para = Parameter(torch.randn(self.x_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.h_dim),
            torch.nn.ReLU(),
        )

        self.encoder_mu = torch.nn.Linear(self.h_dim, self.z_dim)
        self.encoder_logvar = torch.nn.Linear(self.h_dim, self.z_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim + self.n_cov, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.x_dim),
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        if self.log_transform:
            x = torch.log(1+x)
        h = self.encoder(x)
        z_mu, z_logvar = self.encoder_mu(h), self.encoder_logvar(h).clamp(-10,10)
        z = self._reparameterize(z_mu, z_logvar)
        return z_mu, z_logvar, z
    
    def decode(self, z, covariate, library):
        z = torch.concat((z,covariate), dim=1)
        x_hat = self.decoder(z)
        if self.dist == 'NB':
            x_hat = F.softmax(x_hat, dim=1)
            x_hat = x_hat * library.unsqueeze(1) 
        elif self.dist == 'B':
            x_hat = F.sigmoid(x_hat)
        return x_hat

    def forward(self, x):
        covariate = x[:, self.x_dim:]
        z_mu, z_logvar, z = self.encode(x[:, :self.x_dim])
        x_hat = self.decode(z, covariate, x.sum(1))
        return z_mu, z_logvar, z, x_hat
    
class VAE_corr(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VAE_corr, self).__init__()
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        self.para = Parameter(torch.randn(self.x_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.x_dim, self.h_dim),
            torch.nn.BatchNorm1d(self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.encoder_mu = torch.nn.Linear(self.h_dim, self.z_dim)
        self.encoder_logvar = torch.nn.Linear(self.h_dim, self.z_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim + self.n_cov, self.h_dim),
            torch.nn.BatchNorm1d(self.h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.x_dim)
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        if self.log_transform:
            x = torch.log(1+x)
        h = self.encoder(x)
        z_mu, z_logvar = self.encoder_mu(h), self.encoder_logvar(h).clamp(-10,10)
        z = self._reparameterize(z_mu, z_logvar)
        return z_mu, z_logvar, z
    
    def decode(self, z, covariate, library):
        z = torch.concat((z,covariate), dim=1)
        x_hat = self.decoder(z)
        if self.dist == 'NB':
            x_hat = F.softmax(x_hat, dim=1)
            x_hat = x_hat * library.unsqueeze(1) 
        elif self.dist == 'B':
            x_hat = F.sigmoid(x_hat)
        return x_hat

    def forward(self, x):
        covariate = x[:, self.x_dim:]
        z_mu, z_logvar, z = self.encode(x[:, :self.x_dim])
        x_hat = self.decode(z, covariate, x.sum(1))
        return z_mu, z_logvar, z, x_hat