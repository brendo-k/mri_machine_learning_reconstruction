from torch import nn
import torch
from ml_recon.models.UNet import concat, double_conv, Unet_down, Unet_up, down

class XNet(nn.Module):
    def __init__(self, contrast_order: list, channels: int, depth: int = 4, drop_prob: float = 0.0):
        super().__init__()
        self.contrast_order = contrast_order
        self.channels = channels
        self.depth = depth

        n_contrasts = len(self.contrast_order)
        self.down_sample_layers = nn.ModuleList([nn.ModuleList([
            double_conv(2, channels, drop_prob=0)
            ]) for _ in range(n_contrasts)])
        cur_chan = channels
        for _ in range(depth - 1):
            for c in range(n_contrasts):
                self.down_sample_layers[c].append(
                    Unet_down(cur_chan, cur_chan*2, drop_prob=drop_prob)
                    )
            cur_chan *= 2

        self.final_downsampling = down() 
        self.mixing_layer = double_conv(cur_chan*n_contrasts, cur_chan * 2, drop_prob=0)

        self.upsampling_layer = nn.ModuleList([nn.ModuleList([]) for _ in range(n_contrasts)])

        cur_chan = cur_chan * 2
        for _ in range(depth):
            for c in range(n_contrasts):
                self.upsampling_layer[c].append(Unet_up(cur_chan, cur_chan//2, drop_prob))
            cur_chan //= 2

        self.final_conv = nn.ModuleList(
            [nn.Conv2d(cur_chan, 2, 1, bias=False) for _ in range(n_contrasts)]
        )
        
    def forward(self, x:torch.Tensor):
        stack = [[] for _ in range(len(self.contrast_order))]
        for contrast_index, contrast in enumerate(x.split(2, dim=1)):
            sub_network = self.down_sample_layers[contrast_index]
            input = contrast
            for layer in sub_network: # type: ignore
                output = layer(input)
                input = output
                stack[contrast_index].append(output)

        latent_space = [self.final_downsampling(contrast_output[-1]) for contrast_output in stack]
        
        latent_space = torch.concatenate(latent_space, dim=1)

        latent_space = self.mixing_layer(latent_space)


        decoder_outputs = []
        for contrast_index in range(len(self.contrast_order)):
            x = latent_space
            sub_network = self.upsampling_layer[contrast_index]
            for i, layer in enumerate(sub_network):
                x = layer(x, stack[contrast_index][-i - 1])
            decoder_outputs.append(x)

        final = []
        for output, conv in zip(decoder_outputs, self.final_conv):
            final.append(conv(output))

        return torch.cat(final, dim=1)
        
