import numpy as np

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

def espirit(k_space, kernel_size, calibration_size, singular_value_threshold=0.1, eigen_threshold=0.99):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sz, sc, sy, size_x), where (size_x, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel 
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the 
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (size_x, sy, sz, nc, nc) with (size_x, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """

    size_z = np.shape(k_space)[0]
    size_chan = np.shape(k_space)[1]
    size_y = np.shape(k_space)[2]
    size_x = np.shape(k_space)[3]

    acs_x_bounds = (size_x//2-calibration_size//2, size_x//2+calibration_size//2) if (size_x > 1) else (0, 1)
    acs_y_bounds = (size_y//2-calibration_size//2, size_y//2+calibration_size//2) if (size_y > 1) else (0, 1)
    acs_z_bounds = (size_z//2-calibration_size//2, size_z//2+calibration_size//2) if (size_z > 1) else (0, 1)

    # Extract calibration region.    
    calibration_region = k_space[
                                 acs_z_bounds[0]:acs_z_bounds[1], 
                                 :,
                                 acs_y_bounds[0]:acs_y_bounds[1], 
                                 acs_x_bounds[0]:acs_x_bounds[1], 
                                 ].astype(np.complex64)

    # Construct Hankel matrix.
    p = (size_x > 1) + (size_y > 1) + (size_z > 1)
    A = np.zeros([(calibration_size-kernel_size+1)**p, kernel_size**p * size_chan]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, calibration_region.shape[3] - kernel_size + 1)):
        for ydx in range(max(1, calibration_region.shape[2] - kernel_size + 1)):
            for zdx in range(max(1, calibration_region.shape[0] - kernel_size + 1)):
                # numpy handles when the indices are too big
                block = calibration_region[zdx:zdx+kernel_size, 
                                           :, 
                                           xdx:xdx+kernel_size, 
                                           ydx:ydx+kernel_size].astype(np.complex64) 
                A[idx, :] = block.flatten()
                idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    num_singular_values = np.sum(S >= singular_value_threshold * S[0])
    V = V[:, 0:num_singular_values]

    kxt = (size_x//2-kernel_size//2, size_x//2+kernel_size//2) if (size_x > 1) else (0, 1)
    kyt = (size_y//2-kernel_size//2, size_y//2+kernel_size//2) if (size_y > 1) else (0, 1)
    kzt = (size_z//2-kernel_size//2, size_z//2+kernel_size//2) if (size_z > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(k_space), num_singular_values)).astype(np.complex64)
    kerdims = [(size_z > 1) * kernel_size + (size_z == 1) * 1, 
               size_chan, 
               (size_y > 1) * kernel_size + (size_y == 1) * 1, 
               (size_x > 1) * kernel_size + (size_x == 1) * 1, 
               ]
    for sing_val_index in range(num_singular_values):
        kernels[kzt[0]:kzt[1], :, kyt[0]:kyt[1], kxt[0]:kxt[1], sing_val_index] = np.reshape(V[:, sing_val_index], kerdims)

    # Take the fft of the kernels
    axes = (0, 1, 2)
    kerimgs = np.zeros((np.shape(k_space) + (num_singular_values,))).astype(np.complex64)
    for sing_val_index in range(num_singular_values):
        for chan_index in range(size_chan):
            ker = kernels[::-1, chan_index, ::-1, ::-1, sing_val_index].conj()
            kerimgs[:,chan_index,:,:,sing_val_index] = fft(ker, axes) * np.sqrt(size_x * size_y * size_z)/np.sqrt(kernel_size**p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.shape(k_space) + (size_chan,)).astype(np.complex64)
    for x in range(0, size_x):
        for y in range(0, size_y):
            for z in range(0, size_z):

                Gq = kerimgs[z,:,y,x,:]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, size_chan):
                    if (s[ldx]**2 > eigen_threshold):
                        maps[z, :, y, x, ldx] = u[:, ldx]

    return maps

def espirit_proj(x, esp):
    """
    Construct the projection of multi-channel image x onto the range of the ESPIRiT operator. Returns the inner
    product, complete projection and the null projection.

    Arguments:
      x: Multi channel image data. Expected dimensions are (size_x, sy, sz, nc), where (size_x, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      esp: ESPIRiT operator as returned by function: espirit

    Returns:
      ip: This is the inner product result, or the image information in the ESPIRiT subspace.
      proj: This is the resulting projection. If the ESPIRiT operator is E, then proj = E E^H x, where H is 
            the hermitian.
      null: This is the null projection, which is equal to x - proj.
    """
    ip = np.zeros(x.shape).astype(np.complex64)
    proj = np.zeros(x.shape).astype(np.complex64)
    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
            ip[:, :, :, qdx] = ip[:, :, :, qdx] + x[:, :, :, pdx] * esp[:, :, :, pdx, qdx].conj()

    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
          proj[:, :, :, pdx] = proj[:, :, :, pdx] + ip[:, :, :, qdx] * esp[:, :, :, pdx, qdx]

    return (ip, proj, x - proj)