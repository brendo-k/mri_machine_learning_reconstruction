
import pytest
import numpy as np
from ml_recon.utils.simulated_k_space_from_brats import simulate_k_space
import tempfile
import h5py


@pytest.fixture
def sample_image():
    # Generate a sample image [contrast, height, width] for testing
    return np.random.rand(3, 128, 128)  # 3 contrasts, 64x64 image

@pytest.fixture(scope="session")
def sensetivity_file(tmp_path_factory):
    shape = (10, 128, 128)
    sensetivity_f = np.zeros(shape, dtype=np.complex64)
    sensetivity_f[:, shape[1]//2-3:shape[1]+3, shape[2]//2-4:shape[2]+3] = np.random.randn(10, 6, 6) + 1j*np.random.randn(10, 6, 6)
    sensetivities = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sensetivity_f, axes=(-1, -2)), axes=(1, -2)), axes=(-1, -2))

    xx, yy = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2), np.arange(-shape[2]//2, shape[2]//2))

    circle_index = (np.power(xx, 2) + np.power(yy, 2)) > 96**2
    for i in range(sensetivities.shape[0]):
        sensetivities[i][circle_index] = 0

    file_path = tmp_path_factory.mktemp("data") / "compressed_10_norm_coils.npy"

    np.save(file_path, sensetivities)
    return file_path
    

def test_simulate_k_space_output_shape(sample_image):
    seed = 42
    k_space = simulate_k_space(sample_image, seed,  center_region=20, noise_std=0.001, coil_size=10)
    
    # The output should have the shape [contrast, channel, height, width]
    assert k_space.shape == (3, 10, 64, 64), "Output shape mismatch"

def test_simulate_k_space_non_zero_phase(sample_image):
    seed = 42
    k_space = simulate_k_space(sample_image, seed,  center_region=20, noise_std=0.001, coil_size=10)
    
    # Perform inverse FFT to get the image back in the image domain
    img_domain = np.fft.ifft2(k_space, axes=(-2, -1))
    
    # Check if the phase of the resulting image is non-zero
    phase = np.angle(img_domain)
    assert np.any(phase != 0), "Phase should not be zero"

def test_simulate_k_space_noise_application(sample_image):
    seed = 42
    noise_std = 0.001
    k_space_no_noise = simulate_k_space(sample_image, seed,  center_region=20, noise_std=0.0, coil_size=10)
    k_space_with_noise = simulate_k_space(sample_image, seed,  center_region=20, noise_std=noise_std, coil_size=10)
    
    # The noisy version should differ from the noise-free version
    assert np.any(np.abs(k_space_no_noise - k_space_with_noise) > 0), "Noise was not applied properly"

