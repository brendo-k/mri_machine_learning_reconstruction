from scipy.interpolate import RegularGridInterpolator
import numpy as np

from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares

def simulate_k_space(image, seed, same_phase=False, center_region=20, noise_std=0.001, coil_size=12):
    #simulate some random motion
    rng = np.random.default_rng(seed)
    x_shift, y_shift = rng.integers(-5, 5), rng.integers(-5, 5)
    image = np.roll(np.roll(image, x_shift, axis=-1), y_shift, axis=-2)
    #image [Contrast height width]
    image_w_sense = apply_sensetivities(image, coil_size)
    #image_w_sense [Contrast coil height width]
    image_w_phase = generate_and_apply_phase(image_w_sense, seed, same_phase=same_phase, center_region=center_region)
    k_space = fft_2d_img(image_w_phase)
    k_space = apply_noise(k_space, seed, noise_std)
    return k_space


def apply_sensetivities(image, coil_size):
    coil_size = str(coil_size)

    if coil_size == '1':
        image = np.expand_dims(image, 1)
        return image

    #sense_map = np.load('/home/brenden/Documents/data/subset/coil_compressed_' + coil_size + '.npy')
    sense_map = np.load('/home/brenden/Documents/data/coil_compressed_10.npy')
    print(sense_map.shape)
    sense_map = np.transpose(sense_map, (2, 1, 0))

    #mag_sense_map = np.abs(sense_map)
    #mag_sense_phase = np.angle(sense_map)
    #print(sense_map.shape)

    #resampled_sense_mag = resample(mag_sense_map, image.shape[1], image.shape[2])
    #resampled_sense_phase = resample(mag_sense_phase, image.shape[1], image.shape[2], 'nearest')
    #
    #resampled_sense = resampled_sense_mag * np.exp(resampled_sense_phase * 1j)

    mag_sense_real = np.real(sense_map)
    mag_sense_imag = np.imag(sense_map)
    resampled_sense_real = resample(mag_sense_real, image.shape[1], image.shape[2], 'nearest')
    resampled_sense_imag = resample(mag_sense_imag, image.shape[1], image.shape[2], 'nearest')
    resampled_sense = resampled_sense_real + 1j * resampled_sense_imag 

    #kernels = fft_2d_img(sense_map, axes=[-1, -2])
    #kernels = zero_pad_or_crop(kernels, (sense_map.shape[0], image.shape[1], image.shape[2]))

    #plt.imshow(np.abs(kernels[0, :,:])**0.2)
    #plt.show()
    #resampled_sense = ifft_2d_img(kernels)
    #resampled_sense[resampled_sense < 0.005] = 0
    #resampled_sense = resampled_sense / np.sqrt(np.sum(resampled_sense * resampled_sense.conj() + 1e-20, 0, keepdims=True))

    #plt.imshow(np.abs(np.sum(resampled_sense * resampled_sense.conj(), axis=0)))
    #plt.show()

    sense_map = np.expand_dims(resampled_sense, 0)
    image_sense = sense_map * np.expand_dims(image, 1)
    return image_sense      


def zero_pad_or_crop(arr, target_shape):
    original_shape = arr.shape
    padding = []
    for original_s, target_s in zip(original_shape, target_shape):
        if original_s < target_s:
            padding.append(((target_s - original_s)//2, np.ceil(target_s - original_s).astype(int)))
        else:
            padding.append((0, 0))

    # Zero pad if the target shape is larger
    padded = np.pad(arr, padding, mode='constant')

    slices = []
    # Crop if the target shape is smaller
    for i in range(len(target_shape)):
        if original_shape[i] > target_shape[i]:
            # Calculate the start and end indices for center crop
            start = (original_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, target_shape[i])) 

    return padded[tuple(slices)]


def generate_and_apply_phase(data, seed, center_region=20, same_phase=False):
    if same_phase: 
        nc = 1
    else:
        nc = data.shape[0]

    #phase = build_phase(center_region, data.shape[-2], data.shape[-1], nc, same_phase=same_phase, seed=seed)
    phase = build_phase_from_same_dist(data, seed)
    data = apply_phase_map(data, phase)
    return data


def build_phase_from_same_dist(data, seed): 
    rng = np.random.default_rng(seed=seed)
    coeffs = rng.uniform(-1, 1, size=(data.shape[0], data.shape[2], data.shape[3])) + 1j*rng.uniform(-1, 1, size=(data.shape[0], data.shape[2], data.shape[3]))
    k_space = ifft_2d_img(root_sum_of_squares(data, coil_dim=1)) 
    phase_images = fft_2d_img(np.abs(k_space) * coeffs)
    phase = np.angle(phase_images)

    return phase



def build_phase(center_region, nx, ny, nc, same_phase=False, seed=None):
    rng = np.random.default_rng(seed)

    phase_frequency = np.zeros((1, nx, ny), dtype=np.complex64)
    if not same_phase:
        phase_frequency = np.tile(phase_frequency, (nc, 1, 1))

    center = (nx//2, ny//2)
    center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
    center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
    coeff = rng.random(size=(phase_frequency.shape[0], center_region, center_region)) + 1j * rng.random(size=(phase_frequency.shape[0], center_region, center_region))
    coeff -= 0.5 + 1j * 0.5
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