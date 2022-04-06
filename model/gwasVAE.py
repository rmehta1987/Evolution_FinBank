import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))


class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
    
        SiLU(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf

    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return swish(input) # simply apply already implemented SiLU

class GwasVAE(BaseVAE):

    def __init__(self,
                 ss_stats_dim: int,
                 true_beta: int,
                 gamma: int,
                 omega: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(GwasVAE, self).__init__()

        self.true_beta = true_beta
        self.gamma = gamma
        self.omega = omega

        modules = []
        if hidden_dims is None:
            hidden_dims = [self.true_beta]*3
        channels = ss_stats_dim
        
        # ========================================================================#
        # Build true beta Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(channels, h_dim),
                    SiLU())
            )
        self.encoder_true_beta = nn.Sequential(*modules)
        # ========================================================================#
        '''
        # ========================================================================#
        # Build omega Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(channels, h_dim),
                    SiLU())
            )

        self.encoder_omega = nn.Sequential(*modules)

        # ========================================================================#
        
        # ========================================================================#
        # Build true gamma Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(channels, h_dim),
                    SiLU())
            )

        self.encoder_gamma = nn.Sequential(*modules)
        '''
        # ========================================================================#
        # Build true beta decoder
        self.debed_z1_code = nn.Linear(latent1_dim, 1024)
        self.debed_z2_code = nn.Linear(latent2_dim, 1024)
        modules = []
        hidden_dims.reverse()

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

       
    def encode_evolutionary_parameters(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_z2_layers(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z2_mu = self.fc_z2_mu(result)
        z2_log_var = self.fc_z2_var(result)

        return [z2_mu, z2_log_var]

    def encode_true_betas(self, input: Tensor, z2: Tensor) -> List[Tensor]:
        x = self.embed_data(input)
        z2 = self.embed_z2_code(z2)
        z2 = z2.view(-1, self.img_size, self.img_size).unsqueeze(1)
        result = torch.cat([x, z2], dim=1)

        result = self.encoder_z1_layers(result)
        result = torch.flatten(result, start_dim=1)
        z1_mu = self.fc_z1_mu(result)
        z1_log_var = self.fc_z1_var(result)

        return [z1_mu, z1_log_var]

    def encode(self, input: Tensor) -> List[Tensor]:
        
        true_beta_mean, true_beta_logvar = self.model.encode(encoder_true_beta)
        
        
        return [z1_mu, z1_log_var, z2_mu, z2_log_var, z2]

    def decode(self, input: Tensor) -> Tensor:
        result = self.decoder(input)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        # Encode the input summary statistics beta-hat into conditional evolutionary parameters
        # true-beta, gamma (selection coefficient), omega (strength of stabalizing selection)
        # true-beta ~q(true-beta | beta-hat)
        # gamma ~ q(gamma | true-beta)
        # omega ~ q(omega | true-beta)
        
        true_beta_mu, true_beta_log_var, omega_mu, omega_var, gamma_mu, gamma_var = self.encode(input)
       

        # Reconstruct the image using both the latent codes
        # x ~ p(x|z1, z2)
        debedded_z1 = self.debed_z1_code(z1)
        debedded_z2 = self.debed_z2_code(z2)
        result = torch.cat([debedded_z1, debedded_z2], dim=1)
        result = result.view(-1, 512, 2, 2)
        recons = self.decode(result)

        return  [recons,
                 input,
                 z1_mu, z1_log_var,
                 z2_mu, z2_log_var,
                 z1, z2]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        z1_mu = args[2]
        z1_log_var = args[3]

        z2_mu = args[4]
        z2_log_var = args[5]

        z1= args[6]
        z2 = args[7]

        # Reconstruct (decode) z2 into z1
        # z1 ~ p(z1|z2) [This for the loss calculation]
        z1_p_mu = self.recons_z1_mu(z2)
        z1_p_log_var = self.recons_z1_log_var(z2)


        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        z1_kld = torch.mean(-0.5 * torch.sum(1 + z1_log_var - z1_mu ** 2 - z1_log_var.exp(), dim = 1),
                            dim = 0)
        z2_kld = torch.mean(-0.5 * torch.sum(1 + z2_log_var - z2_mu ** 2 - z2_log_var.exp(), dim = 1),
                            dim = 0)

        z1_p_kld = torch.mean(-0.5 * torch.sum(1 + z1_p_log_var - (z1 - z1_p_mu) ** 2 - z1_p_log_var.exp(),
                                               dim = 1),
                            dim = 0)

        z2_p_kld = torch.mean(-0.5*(z2**2), dim = 0)

        kld_loss = -(z1_p_kld - z1_kld - z2_kld)
        loss = recons_loss + kld_weight * kld_loss
        # print(z2_p_kld)

        return {'loss': loss, 'Reconstruction Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        z2 = torch.randn(batch_size,
                         self.latent2_dim)

        z2 = z2.cuda(current_device)

        z1_mu = self.recons_z1_mu(z2)
        z1_log_var = self.recons_z1_log_var(z2)
        z1 = self.reparameterize(z1_mu, z1_log_var)

        debedded_z1 = self.debed_z1_code(z1)
        debedded_z2 = self.debed_z2_code(z2)

        result = torch.cat([debedded_z1, debedded_z2], dim=1)
        result = result.view(-1, 512, 2, 2)
        samples = self.decode(result)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
