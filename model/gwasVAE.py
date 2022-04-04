import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class GwasVAE(BaseVAE):

    def __init__(self,
                 ss_stats_dim: int,
                 true_beta: int,
                 gamma: int,
                 omega: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
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
        self.embed_z2_code = nn.Linear(latent2_dim, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        channels = in_channels + 1 # One more channel for the latent code
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channels = h_dim

        self.encoder_z1_layers = nn.Sequential(*modules)
        self.fc_z1_mu = nn.Linear(hidden_dims[-1]*4, latent1_dim)
        self.fc_z1_var = nn.Linear(hidden_dims[-1]*4, latent1_dim)
        # ========================================================================#

        # ========================================================================#
        # Build omega Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channels = h_dim

        self.encoder_z2_layers = nn.Sequential(*modules)
        self.fc_z2_mu = nn.Linear(hidden_dims[-1]*4, latent2_dim)
        self.fc_z2_var = nn.Linear(hidden_dims[-1]*4, latent2_dim)
        # ========================================================================#
        
        # ========================================================================#
        # Build true gamma Encoder
        self.embed_z2_code = nn.Linear(latent2_dim, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        channels = in_channels + 1 # One more channel for the latent code
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channels = h_dim

        self.encoder_z1_layers = nn.Sequential(*modules)
        self.fc_z1_mu = nn.Linear(hidden_dims[-1]*4, latent1_dim)
        self.fc_z1_var = nn.Linear(hidden_dims[-1]*4, latent1_dim)

        # ========================================================================#
        # Build true beta Decoder
        self.debed_z1_code = nn.Linear(latent1_dim, 1024)
        self.debed_z2_code = nn.Linear(latent2_dim, 1024)
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



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

       
    def encode_z2(self, input: Tensor) -> List[Tensor]:
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

    def encode_z1(self, input: Tensor, z2: Tensor) -> List[Tensor]:
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
        z2_mu, z2_log_var = self.encode_z2(input)
        z2 = self.reparameterize(z2_mu, z2_log_var)

        # z1 ~ q(z1|x, z2)
        z1_mu, z1_log_var = self.encode_z1(input, z2)
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

        # Encode the input into the latent codes z1 and z2
        # z2 ~q(z2 | x)
        # z1 ~ q(z1|x, z2)
        z1_mu, z1_log_var, z2_mu, z2_log_var, z2 = self.encode(input)
        z1 = self.reparameterize(z1_mu, z1_log_var)

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
