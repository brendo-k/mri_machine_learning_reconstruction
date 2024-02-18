import numpy as np
import torch.nn as nn
import torch
import einops

class Patches(nn.Module):

    def __init__(
            self, 
            patch_size=7,
            n_patches=10,
            window_size=20,
            embedding_dimension=128,
            nhead=4,
            n_encoders=4
            ):

        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension
        self.nhead = nhead
        self.n_encoders = n_encoders

        self.projection = nn.Linear(patch_size*patch_size, embedding_dimension, bias=True)
        self.encoder = torch.nn.Sequential()
        
        for _ in range(n_encoders): 
            self.encoder.append(
                    torch.nn.TransformerEncoderLayer(embedding_dimension, nhead=4)
                    )

        self.back_projection = nn.Linear(embedding_dimension, patch_size*patch_size, bias=True)
        # 1D positional embedding... Will try 2d later
        self.position_embedding = nn.Parameter(torch.randn(1, n_patches, embedding_dimension))


    def loop_through_image(self, data):
        index_counter = torch.ones(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
        new_image = data.clone()
        for x in range(0, data.shape[2] - self.window_size + 1, 4):
            for y in range(0, data.shape[3] - self.window_size + 1, 4):
                similar_patches, similar_indecies = self.get_n_similar_patches(
                        data, 
                        [x, y], 
                        )


                projected_patches = self.project_patches(similar_patches)
                projected_patches = projected_patches + self.position_embedding

                projected_patches = einops.rearrange(projected_patches, 'batch chan n_sim patch -> batch (chan n_sim) patch')
                denoised_patches = self.encoder(projected_patches)
                denoised_patches = self.back_projection(projected_patches)
                denoised_patches = einops.rearrange(denoised_patches, 'batch (chan n_sim) (patch1 patch2) -> batch chan n_sim patch1 patch2', n_sim=self.n_patches, patch1=self.patch_size) 

                for n in range(denoised_patches.shape[2]):
                    for b in range(denoised_patches.shape[0]):
                        for c in range(denoised_patches.shape[1]):
                            x_index, y_index = similar_indecies[b, c, n, :]


                            patch = new_image[b, c, y_index:y_index+self.patch_size, x_index:x_index+self.patch_size]
                            patch *= index_counter[b, c, y_index:y_index+self.patch_size, x_index:x_index+self.patch_size]
                            patch += denoised_patches[b, c, n, :].reshape((self.patch_size, self.patch_size))
                            patch /= (index_counter[b, c, y_index:y_index+self.patch_size, x_index:x_index+self.patch_size] + 1)
                            new_image[b, c, y_index:y_index+self.patch_size, x_index:x_index+self.patch_size] = patch

                            index_counter[b, c, x_index:x_index+self.patch_size, y_index:y_index+self.patch_size] += 1
        
        return new_image

    def forward(self, images):
        return self.loop_through_image(images) 
                                

                
    def project_patches(self, patches):
        return self.projection(patches)
        
        
    def get_n_similar_patches(self, data, reference_index):
        # refernce index is the top left corner ie.) 0, 0 is the patch [:, :, 0:patch_size, 0:patch_size]
        # data shape [b contrast height width]
        assert torch.is_floating_point(data), 'Should be floating point!'
        assert data.shape[-1] > reference_index[-1]
        assert data.shape[-2] > reference_index[-2]

        n = self.n_patches
        patch_size = self.patch_size
        batch_size = data.shape[0]
        contrast_size = data.shape[1]
        
        patch_bounds_x = torch.tensor([0, patch_size]) + reference_index[0]
        patch_bounds_y = torch.tensor([0, patch_size]) + reference_index[1]

        # [batch contrast p p]
        reference_patch = data[:, :, patch_bounds_y[0]:patch_bounds_y[1], patch_bounds_x[0]:patch_bounds_x[1]]

        window_bounds_x_start = reference_index[0] + patch_size//2 - self.window_size//2 
        window_bounds_y_start = reference_index[1] + patch_size//2 - self.window_size//2 
        window_bounds_x = [window_bounds_x_start, window_bounds_x_start+self.window_size]
        window_bounds_y = [window_bounds_y_start, window_bounds_y_start+self.window_size]

        window_bounds_x[0] = max(window_bounds_x[0], 0)
        window_bounds_y[0] = max(window_bounds_y[0], 0)
        
        window_bounds_x[1] = min(window_bounds_x[1], data.shape[-1])
        window_bounds_y[1] = min(window_bounds_y[1], data.shape[-2])

        window = data[:, :, window_bounds_y[0]:window_bounds_y[1], window_bounds_x[0]:window_bounds_x[1]]

        # unfolding = torch.nn.Unfold() POSSIBLE WAY TO SPEED UP
        # skimage.util.view_as_windows POSSIBLE EASY WAY TO DO

        vectorized_window = einops.rearrange(window, 'b c h w -> 1 (b c) h w')
        vectorized_patch = einops.rearrange(reference_patch, 'b c h w -> (b c) 1 h w')

        # output shape is [1 (batch*contrast) height-patch_size width-patch_size]
        similarity_metric = torch.nn.functional.conv2d(vectorized_window, vectorized_patch, groups=vectorized_window.shape[1])

        # shape now [(batch*contrast) (height widht)]
        similarity_metric = similarity_metric.flatten(2, 3).squeeze()

        # shape is now [(b c) n]
        _, similar_indecies = torch.topk(similarity_metric, self.n_patches, dim=1)

        indecies = np.unravel_index(similar_indecies, [window.shape[2]-self.patch_size+1, window.shape[3]-self.patch_size+1])

        patch_slice = np.stack([np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end)) 
                            for x_start, x_end, y_start, y_end in zip(indecies[0].flatten(), indecies[0].flatten() + self.patch_size, indecies[1].flatten(), indecies[1].flatten() + self.patch_size)]
                           )


        x_coords = indecies[0] + window_bounds_x[0]
        y_coords = indecies[1] + window_bounds_y[0]
        coords = torch.from_numpy(np.stack((x_coords, y_coords), axis=-1))

        patch_slice = patch_slice.reshape(patch_slice.shape[0], patch_slice.shape[1], -1)

        batch_indecies, contrast_indecies, _ = np.indices((data.shape[0], data.shape[1], self.n_patches))
        similarity_tensor = window[batch_indecies.flatten()[:, None], contrast_indecies.flatten()[:, None], patch_slice[:, 0, :], patch_slice[:, 1, :]]
        similarity_tensor = similarity_tensor.reshape((batch_size, contrast_size, n, patch_size*patch_size))

        coords = einops.rearrange(coords, '(b c) n coords -> b c n coords', b=batch_size)
        return similarity_tensor, coords
