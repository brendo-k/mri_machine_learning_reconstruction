import torch.nn as nn
import torch

class Patches(nn.Module):

    def __init__(self, patch_size, n_patches, window_size, embedding_dimension):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.window_size = window_size

        self.projection = nn.Linear(patch_size*patch_size, embedding_dimension, bias=True)


    def loop_through_image(self, data, n, window_size, patch_size):
        for x in range(data.shape[2] - window_size + 1):
            for y in range(data.shape[3] - window_size + 1):
                similar_patches = self.get_n_similar_patches(
                        data, 
                        [x, y], 
                        )
                
    def project_patches(self, patches):
        return self.projection(patches)
        
        
    def get_n_similar_patches(self, data, reference_index):
        # refernce index is the top left corner ie.) 0, 0 is the patch [:, :, 0:patch_size, 0:patch_size]
        # data shape [b contrast height width]
        n = self.n_patches
        patch_size = self.patch_size
        
        patch_bounds_x = torch.tensor([0, self.patch_size]) + reference_index[0]
        patch_bounds_y = torch.tensor([0, self.patch_size]) + reference_index[1]

        # [batch contrast p p]
        reference_patch = data[:, :, patch_bounds_x[0]:patch_bounds_x[1], patch_bounds_y[0]:patch_bounds_y[1]]

        window_bounds_x = [reference_index[0] - self.window_size//2, reference_index[0] + self.window_size//2]
        window_bounds_y = [reference_index[1] - self.window_size//2, reference_index[1] + self.window_size//2]

        window_bounds_x[0] = max(window_bounds_x[0], 0)
        window_bounds_y[0] = max(window_bounds_y[0], 0)
        
        window_bounds_x[1] = min(window_bounds_x[1], data.shape[-2])
        window_bounds_y[1] = min(window_bounds_y[1], data.shape[-1])


        window = data[:, :, window_bounds_x[0]:window_bounds_x[1], window_bounds_y[0]:window_bounds_y[1]]
        print(window.shape)

        similarity_tensor = torch.zeros(data.shape[0], data.shape[1], n, patch_size*patch_size, dtype=torch.complex64)

        for batch in range(window.shape[0]):
            for contrast in range(window.shape[1]):
                n_similar_tensors = torch.zeros(n, patch_size*patch_size, dtype=torch.complex64)
                n_distance = torch.tensor([10e32]).repeat(n)
                n_index = torch.tensor([0]).repeat(n)
                for x in range(window.shape[2] - patch_size + 1):
                    for y in range(window.shape[3] - patch_size + 1):
                        current_patch_tensor = window[batch, contrast, x:x+patch_size, y:y+patch_size]

                        distance = nn.functional.mse_loss(torch.view_as_real(current_patch_tensor), torch.view_as_real(reference_patch[batch, contrast, :, :]))

                        indecies = torch.nonzero(n_distance > distance)
                        if indecies.nelement() > 0:
                            n_distance[indecies[0]] = distance
                            n_similar_tensors[indecies[0], :] = current_patch_tensor.flatten()

                similarity_tensor[batch, contrast, :, :] = n_similar_tensors
        return similarity_tensor

        
        
        

        
        
        

