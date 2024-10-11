import pytest 
import torch 
import ml_recon.pl_modules.pl_loupe as pl_loupe
from itertools import repeat

def test_norm_prob():
    contrasts = 4
    image_size = (320, 320)
    # image size
    input = [torch.rand((image_size)) for _ in range(contrasts)]

    # target mean probability 
    R = 6

    loupe = pl_loupe.LOUPE((contrasts,) + image_size, R, 2, ['t1', 't2', 'flair'])
    normed_input = loupe.norm_prob(input, torch.full((4, 1), R))

    normed_input = torch.stack(normed_input, dim=0) 

    actual_R_values = normed_input.mean((-1, -2))
    
    torch.testing.assert_close(actual_R_values, torch.full(actual_R_values.shape, 1/R))
    

if __name__ == '__main__':
    pytest.main()