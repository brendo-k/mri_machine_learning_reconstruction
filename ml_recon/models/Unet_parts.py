import einops
import torch.nn as nn
import torch 

class double_conv(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
      
    def forward(self, x):
        x = self.layers(x)
        return x

class down(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.AvgPool2d(2, stride=(2, 2))

    def forward(self, x):
        x = self.max_pool(x)
        return x


class up(nn.Module):
  def __init__(self, in_chan, out_chan):
    super().__init__()
    self.layers = nn.Sequential(
      nn.ConvTranspose2d(in_chan, out_chan, stride=2, kernel_size=2, bias=False),
      nn.InstanceNorm2d(out_chan),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )
  def forward(self, x):
    x = self.layers(x)
    return x


class concat(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x_encode: torch.Tensor, x_decode: torch.Tensor):
    x_enc_shape = x_encode.shape[-2:]
    x_dec_shape = x_decode.shape[-2:]
    diff_x = x_enc_shape[0] - x_dec_shape[0]
    diff_y = x_enc_shape[1] - x_dec_shape[1]
    x_enc_trimmed = x_encode
    if diff_x != 0:
      x_enc_trimmed = x_enc_trimmed[:, :, diff_x//2:-diff_x//2, :]
    if diff_y != 0:
      x_enc_trimmed = x_enc_trimmed[:, :, :, diff_y//2:-diff_y//2]
    concated_data = torch.cat((x_decode, x_enc_trimmed), dim=1)
    return concated_data