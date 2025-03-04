from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.ndimage import gaussian_filter


from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares

def simulate_k_space(image, seed, noise_std=0.001, coil_file='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed_10.npy'):
    #simulate some random motion
    rng = np.random.default_rng(seed)
    x_shift, y_shift = rng.integers(-5, 5), rng.integers(-5, 5)
    #image [Contrast height width]
    image = np.roll(np.roll(image, x_shift, axis=-1), y_shift, axis=-2)
    #image_w_sense [Contrast coil height width]
    image_w_sense = apply_sensetivities(image, coil_file)
    image_w_phase = generate_and_apply_phase(image_w_sense, seed)

    gt_img = root_sum_of_squares(image_w_phase, coil_dim=1)
    k_space = fft_2d_img(image_w_phase)
    k_space = apply_noise(k_space, seed, noise_std)
    return k_space, gt_img

def apply_sensetivities(image, coil_file):

    if not coil_file:
        image = np.expand_dims(image, 1)
        return image

    sense_map = np.load(coil_file)

    sense_map = np.expand_dims(sense_map, 0)
    image_sense = sense_map * np.expand_dims(image, 1)
    return image_sense      


def generate_and_apply_phase(data, seed):
    nc = data.shape[0]
    phase = build_phase_from_same_dist(data, seed)
    #phase = build_phase(center_region, data.shape[-2], data.shape[-1], nc, seed)
    phase_data = apply_phase_map(data, phase)
    return phase_data


def build_phase_from_same_dist(data, seed): 
    rng = np.random.default_rng(seed=seed)
    temp = np.copy(data)
    smoothed_base = gaussian_filter(temp.mean(0, keepdims=True), 10, axes=(-1, -2))
    k_base = ifft_2d_img(root_sum_of_squares(smoothed_base, coil_dim=1))    
    coeffs = rng.uniform(-1, 1, size=(data.shape[2], data.shape[3])) + 1j*rng.uniform(-1, 1, size=(data.shape[2], data.shape[3]))
    base_phase_image = np.abs(fft_2d_img(np.abs(k_base) * coeffs))
    # center on zero
    base_phase_image -= base_phase_image.min()
    # [0, 2]
    base_phase_image/= (np.max(base_phase_image) / 2)
    # [-1, 1]
    base_phase_image-= -1
    #scale between -2pi and 2pi
    base_phase_image *= (np.pi)


    coeffs = rng.uniform(-1, 1, size=(data.shape[0], data.shape[2], data.shape[3])) + 1j*rng.uniform(-1, 1, size=(data.shape[0], data.shape[2], data.shape[3]))
    smoothed_data = gaussian_filter(temp, 10, axes=(-1, -2))
    k_space = ifft_2d_img(root_sum_of_squares(smoothed_data, coil_dim=1))    
    phase_images = np.abs(fft_2d_img(np.abs(k_space) * coeffs))
    
    # center on zero
    phase_images -= phase_images.min()
    # [0, 2]
    phase_images /= (np.max(phase_images) / 2)
    # [-1, 1]
    phase_images -= -1
    #scale between -2pi and 2pi
    phase_images *= (np.pi)

    return base_phase_image * 0.9 + phase_images * 0.1



def build_phase(center_region, nx, ny, nc, seed=None):
    rng = np.random.default_rng(seed)

    phase_frequency = np.zeros((nc, nx, ny), dtype=np.complex64)

    center = (nx//2, ny//2)
    center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
    center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
    # coeff between [-0.5, 0.5]
    real_coef = 0.5 - rng.random(size=(phase_frequency.shape[0], center_region, center_region))  
    image_coef = 0.5 - rng.random(size=(phase_frequency.shape[0], center_region, center_region))  
    coeff = real_coef + 1j*image_coef
    phase_frequency[:, center_box_x, center_box_y] = coeff

    phase = fft_2d_img(phase_frequency)
    phase = np.angle(phase)
    
    return phase


def apply_phase_map(data, phase):
    return data * np.exp(1j * np.expand_dims(phase, 1))


def apply_noise(k_space, seed, noise_std):
    rng = np.random.default_rng(seed)
    noise_scale = noise_std
    noise = rng.normal(scale=noise_scale, size=k_space.shape) + 1j * rng.normal(scale=noise_scale, size=k_space.shape)
    k_space += noise
    return k_space


def resample(data, resample_height, resample_width, method='linear'):
    contrasts, height, width = data.shape
    interp_data = np.zeros((contrasts, resample_height, resample_width))
    y = np.arange(0, height)
    x = np.arange(0, width)

    yi = np.linspace(0, height - 1, resample_height)
    xi = np.linspace(0, width - 1, resample_width)
    (yi, xi) = np.meshgrid(yi, xi, indexing='ij')
    for i in range(data.shape[0]):
        interpolator = RegularGridInterpolator((y, x), data[i, :, :], method=method)
        interp_data[i, :, :] = interpolator((yi, xi))
    
    return interp_data
