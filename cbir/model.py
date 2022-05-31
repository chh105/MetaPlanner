# code adapted from https://github.com/AntixK/PyTorch-VAE
from typing import List, Callable, Union, Any, TypeVar, Tuple
from abc import abstractmethod
import copy

from torch.nn import functional as F
import torch
from torch import nn

import numpy as np

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class InfoVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = -0.5,
                 beta: float = 100.0,
                 reg_weight: int = 10.0,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 **kwargs) -> None:
        super(InfoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        self.in_channels = in_channels

        assert alpha <= 0, 'alpha must be negative or zero.'

        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 64, 128, 256]

        self.hidden_dims = copy.deepcopy(hidden_dims)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.GroupNorm(h_dim//2,h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 8, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 8, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.GroupNorm(hidden_dims[i + 1]//2,hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose3d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.GroupNorm(hidden_dims[-1]//2,hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            )


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, z, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        bias_corr = np.maximum(batch_size *  (batch_size - 1),1)
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = self.beta * recons_loss + \
               (1. - self.alpha) * kld_weight * kld_loss + \
               (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss


        # loss = self.beta * recons_loss + \
        #        (self.alpha + self.reg_weight - 1.)/bias_corr * mmd_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss, 'KLD':-kld_loss}

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1: Tensor,
                               x2: Tensor,
                               eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class SimSiam(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 beta: float = 1e-2,
                 gamma: float = 1e-1,
                 alpha: float = 1,
                 sigma: List = [1,1],
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SimSiam, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.in_channels = in_channels


        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 64, 128, 256]

        self.hidden_dims = copy.deepcopy(hidden_dims)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    # nn.BatchNorm3d(h_dim),
                    nn.GroupNorm(h_dim//2,h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.projector = nn.Sequential(nn.Linear(hidden_dims[-1] * 8, latent_dim),
                                       )
        self.predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.BatchNorm1d(latent_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(latent_dim, latent_dim),
                                       )

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    # nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.GroupNorm(hidden_dims[i + 1]//2,hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # nn.BatchNorm3d(hidden_dims[-1]),
            nn.GroupNorm(hidden_dims[-1]//2,hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
        )
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input1, input2, pos_img, neg_img):
        result1 = self.encoder(input1)
        result1 = torch.flatten(result1, start_dim=1)
        z1 = self.projector(result1)
        z1 = F.normalize(z1,p=2,dim=1)

        result2 = self.encoder(input2)
        result2 = torch.flatten(result2, start_dim=1)
        z2 = self.projector(result2)
        z2 = F.normalize(z2,p=2,dim=1)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        recon = self.decode(z1)

        pos_img_result = self.encoder(pos_img)
        pos_img_result = torch.flatten(pos_img_result, start_dim=1)
        z_pos = self.projector(pos_img_result)
        z_pos = F.normalize(z_pos,p=2,dim=1)

        neg_img_result = self.encoder(neg_img)
        neg_img_result = torch.flatten(neg_img_result, start_dim=1)
        z_neg = self.projector(neg_img_result)
        z_neg = F.normalize(z_neg,p=2,dim=1)



        return p1, p2, z1, z2, input1, recon, z_pos, z_neg

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        p1 = args[0]
        p2 = args[1]
        z1 = args[2]
        z2 = args[3]
        target = args[4]
        recon = args[5]
        z_pos = args[6]
        z_neg = args[7]



        recon_losses = []
        for c in range(target.size(dim=1)): # channel dim
            recon_losses.append(F.mse_loss(recon[:,c,:,:,:], target[:,c,:,:,:]))

        recon_loss = torch.mul(recon_losses[0],self.sigma[0])
        for c,rl in enumerate(recon_losses[1:]):
            recon_loss = torch.add(recon_loss,torch.mul(rl,self.sigma[c]))
        recon_loss = torch.mul(recon_loss,1/target.size(dim=1))

        # recon_loss = F.mse_loss(recon, target)


        triplet_loss = nn.TripletMarginLoss(margin = 1)(z1,z_pos,z_neg)

        feature_loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + \
                 F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) * 0.5

        loss = self.alpha*recon_loss + self.beta*feature_loss + self.gamma*triplet_loss
        return {'loss': loss, 'Recon_loss': recon_loss, 'Feature_loss': feature_loss, 'Triplet_loss': triplet_loss}

