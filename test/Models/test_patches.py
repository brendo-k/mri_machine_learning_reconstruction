import torch
import pytest
import numpy as np
from ml_recon.models.patches import Patches
import matplotlib.pyplot as plt

# Unit tests for double_conv module
def test_zero_tensor():

    patches = Patches(7, 10, 21, 128)

    data = torch.ones(10, 16, 320, 320, dtype=torch.float32)
    similarity, indecies = patches.get_n_similar_patches(data, [0, 0])

    torch.testing.assert_close(torch.ones(10, 16, 10, 49, dtype=torch.float32), similarity)
    assert indecies.shape == (10, 16, 10, 2)
    
# Unit tests for double_conv module
def test_line_tensor():
    n_patches = 4

    patches = Patches(3, n_patches, 9, 128)

    data = torch.zeros(2, 4, 9, 9, dtype=torch.float32)
    data[:, :, ::2, :] = torch.tensor([1.0])
    similarity, indecies = patches.get_n_similar_patches(data, [3, 3])

    similarity_tensor = torch.zeros(2, 4, 4, 3, 3, dtype=torch.float32)
    similarity_tensor[:, :, :, :, 1] = torch.tensor([1.0])
    similarity_tensor = similarity_tensor.reshape(2, 4, 4, 9)

    torch.testing.assert_close(similarity_tensor, similarity)

    batch, contrast, n = np.indices((data.shape[0], data.shape[1], n_patches))
    patches = data[batch.flatten(), contrast.flatten(), indecies[..., 1].flatten():3, indecies[..., 0].flatten():3]
    print(patches.shape)
    torch.testing.assert_close(patches, similarity_tensor)

def test_patches_shape():

    patches = Patches(7, 5, 21, 128)
    data = torch.zeros(2, 4, 128, 128, dtype=torch.float32)

    denoised = patches(data)

    assert data.shape == denoised.shape

#def test_():
#
#    patches = Patches(8, 21, 7, 128)
#
#    data = torch.zeros(10, 16, 320, 320, dtype=torch.complex64)
#    data[:, :, ::2, :] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
#    similarity = patches.get_n_similar_patches(data, [0, 0])
#
#    similarity_tensor = torch.zeros(10, 16, 8, 7, 7, dtype=torch.complex64)
#    similarity_tensor[:, :, :, ::2, :] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
#    similarity_tensor = similarity_tensor.reshape(10, 16, 8, 49)
#    print(similarity_tensor[0, 0, :, :])
#    print(similarity[0, 0, :, :])
#    print(similarity_tensor[0, 0, :, :] == similarity[0, 0, :, :])
#
#    torch.testing.assert_close(similarity_tensor, similarity)

