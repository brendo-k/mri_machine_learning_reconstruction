import torch
from ml_recon.models.patches import Patches
import matplotlib.pyplot as plt

# Unit tests for double_conv module
def test_zero_tensor():

    patches = Patches(7, 10, 21, 128)

    data = torch.zeros(10, 16, 320, 320, dtype=torch.complex64)
    similarity = patches.get_n_similar_patches(data, [0, 0])

    torch.testing.assert_close(torch.zeros(10, 16, 10, 49, dtype=torch.complex64), similarity)
    
# Unit tests for double_conv module
def test_line_tensor():

    patches = Patches(7, 10, 21, 128)

    data = torch.zeros(10, 16, 320, 320, dtype=torch.complex64)
    data[:, :, ::2, :] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
    similarity = patches.get_n_similar_patches(data, [10, 10])

    similarity_tensor = torch.zeros(10, 16, 10, 7, 7, dtype=torch.complex64)
    similarity_tensor[:, :, :, ::2, :] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
    similarity_tensor = similarity_tensor.reshape(10, 16, 10, 49)
    print(similarity_tensor[0, 0, :, :])
    print(similarity[0, 0, :, :])
    print(similarity_tensor[0, 0, :, :] == similarity[0, 0, :, :])

    torch.testing.assert_close(similarity_tensor, similarity)


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
