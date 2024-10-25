
import pytest
import numpy as np
from ml_recon.utils.simulated_k_space_from_brats import simulate_k_space

@pytest.fixture
def sample_image():
    # Generate a sample image [contrast, height, width] for testing
    return np.random.rand(3, 64, 64)  # 3 contrasts, 64x64 image

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

