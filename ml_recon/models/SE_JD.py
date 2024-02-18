import torch
import torch.nn as nn
from ml_recon.models.unet import Unet_down, Unet_up, double_conv, up


class SingleEncoderJointDecoder(torch.nn.Module):
    def __init__(
            self, 
            in_chan,
            encoder_chan, 
            encoder_depth, 
            decoder_chan, 
            decoder_depth, 
            drop_prob=0
            ):
        super().__init__()

        print(encoder_chan)
        self.down_sample_layers = nn.ModuleList(
                [nn.Sequential(double_conv(2, encoder_chan, drop_prob)) for _ in range(in_chan//2)]
                )

        cur_chan = 0
        for i in range(len(self.down_sample_layers)):
            cur_chan = encoder_chan
            for _ in range(encoder_depth):
                self.down_sample_layers[i].append(Unet_down(cur_chan, cur_chan*2, drop_prob))
                cur_chan *= 2

        self.up_sample_layers = nn.ModuleList()

        latent_chan = cur_chan*(in_chan//2)
        self.up_sample_layers.append(
                nn.Sequential(
                    up(latent_chan, decoder_chan),
                    double_conv(decoder_chan, decoder_chan, drop_prob)
                    )
                )
        cur_chan = decoder_chan
        for _ in range(decoder_depth-1):
            self.up_sample_layers.append(
                    nn.Sequential(
                        up(cur_chan, cur_chan//2),
                        double_conv(cur_chan//2, cur_chan//2, drop_prob)
                        )
                    )
            cur_chan //= 2
        self.conv2d = nn.Conv2d(cur_chan, in_chan, 1, bias=False)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--encoder_chan', default=8, type=int)
        parser.add_argument('--encoder_depth', default=4, type=int)
        parser.add_argument('--decoder_chan', default=8, type=int)
        parser.add_argument('--decoder_depth', default=4, type=int)
        return parser

    def forward(self, x):
        latent_vector = []
        for i, contrast in enumerate(torch.split(x, 2, 1)):
            latent_vector.append(self.down_sample_layers[i](contrast))
        
        latent_vector = torch.cat(latent_vector, dim=1)
        
        x = latent_vector
        for i, layer in enumerate(self.up_sample_layers):
            x = layer(x)

        x = self.conv2d(x)
        return x


        
