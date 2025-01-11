import numpy as np
import timeit
from ml_recon.utils.undersample_tools import ssdu_gaussian_selection

def benchmark_ssdu_gaussian_selection():
    # Create a random input mask
    input_mask = np.random.randint(0, 2, (256, 256))

    # Parameters for the function
    std_scale = 4
    rho = 0.4

    # Call the function
    trn_mask, loss_mask = ssdu_gaussian_selection(input_mask, std_scale, rho)

if __name__ == "__main__":
    elapsed_time = timeit.timeit(benchmark_ssdu_gaussian_selection, number=10)
    print(f"Elapsed time for ssdu_gaussian_selection: {elapsed_time:.6f} seconds")
