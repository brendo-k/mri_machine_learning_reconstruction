from dataclasses import dataclass
import torch 

@dataclass
class TrainingSample:
    """Class for standardizing dataset outputs."""
    input: torch.Tensor 
    target: torch.Tensor 
    fs_k_space: torch.Tensor
    mask: torch.Tensor
    loss_mask: torch.Tensor