import numpy as np

# Adaptive combine method for estimating sensitivities
def adaptive(x, kernel=(3,3,1), thresh=0.75):
    """
    Input:
        x: [Nx, Ny, Nz, Nc]

    Output:
        sens: [Nx, Ny, Nz, Nc]
    """

    Nc = x.shape[-1]
    y = np.zeros(x.shape, dtype=complex)
    m = np.zeros(x.shape[:3])
    x = x/sos(x).reshape(x.shape[:3]+(1,))
    
    for vx in range(x.shape[0]):
        print(f'{0.1*np.round(1000*vx/x.shape[0]):04.1f}%', end='\r')
        for vy in range(x.shape[1]):
            for vz in range(x.shape[2]):
                tmp = x[vx-(kernel[0]-1)//2:vx+(kernel[0]+1)//2, vy-(kernel[1]-1)//2:vy+(kernel[1]+1)//2, vz-(kernel[2]-1)//2:vz+(kernel[2]+1)//2,:].reshape((-1,Nc))
                d, v = np.linalg.eigh(np.conj(tmp.T)@tmp)
                y[vx,vy,vz,:] = v[:,d.argmax()]
                m[vx,vy,vz] = np.sqrt(d.max())
    
    return y*(m>thresh*m.max())[:,:,:,np.newaxis]

# ESPIRiT method for estimating sensitivities
# 2D only for now
def espirit(x, dims=(128,128), kernel=(5,5), eig_thresh=0.02, mask_thresh=0.99):
    """
    Input:
        x: [Nc, Kx, Ky, Nz]
        dims: (Nx, Ny)

    Output:
        sens: [Nx, Ny, Nz, Nc]
    """

    if x.ndim < 4:
        x = x[:,:,:,np.newaxis]
    
    # Get dimensions
    Nc = x.shape[0]
    Nz = x.shape[3]
    Kx = x.shape[1]-kernel[0]+1
    Ky = x.shape[2]-kernel[1]+1
    pad_x = [int(i) for i in (np.ceil((dims[0]-kernel[0])/2), np.floor((dims[0]-kernel[0])/2))]
    pad_y = [int(i) for i in (np.ceil((dims[1]-kernel[1])/2), np.floor((dims[1]-kernel[1])/2))]
    sens  = np.zeros((Nc, dims[0], dims[1], Nz), dtype='complex')
    m  = np.zeros((dims[0], dims[1], Nz), dtype='complex')

    # Loop over slices (Nz)
    for z in range(Nz):
        # Initialise Hankel matrix
        H = np.zeros((Nc, np.prod(kernel), Kx*Ky), dtype='complex')
        for i in range(Kx):
            for j in range(Ky):
                # Populate Hankel matrix
                H[:,:,i*Ky+j] = x[:,i:i+kernel[0],j:j+kernel[1],z].reshape((Nc,-1))

        # Only keep svd singular vectors corresponding to above threshold singular values
        U,S,_ = np.linalg.svd(H.reshape((-1,Kx*Ky)), full_matrices=False)
        U = U[:,S>S[0]*eig_thresh]   

        # Get zero-paded kernel images
        U = U.reshape((Nc,kernel[0],kernel[1],-1))
        U = np.pad(U, ((0,0), pad_x, pad_y, (0,0)))
        U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U, axes=(1,2)), axes=(1,2)), axes=(1,2))
        U = U*np.prod(dims)/np.sqrt(np.prod(kernel))

        # Perform voxel-wise SVD on coil x component matrices, voxelwise, keeping 1st component
        for i in range(dims[0]):
            for j in range(dims[1]):
                W,S,_ = np.linalg.svd(U[:,i,j,:], full_matrices=False)
                sens[:,i,j,z] = W[:,0]
                m[i,j,z] = S[0]
    
    # Rotate phase relative to first channel and return
    return np.squeeze(sens*np.exp(-1j*np.angle(sens[[0],:,:,:]))*(np.abs(m)>mask_thresh))

# ESPIRiT-based method for estimating b0 maps
# 2D only for now
def b0_espirit(x, TE, dims=(128,128), kernel=(5,5), eig_thresh=0.02, mask_thresh=0.99):
    """
    Input:
        x: [Nc, Kx, Ky, echoes]
        TE: [TE_1, TE_2, ..., TE_N]]
        dims: (Nx, Ny)

    Output:
        sens: [Nx, Ny, Nz, Nc]
    """

    # Get dimensions
    Nc = x.shape[0]
    Ne = x.shape[3]
    Kx = x.shape[1]-kernel[0]+1
    Ky = x.shape[2]-kernel[1]+1
    pad_x = [np.int(i) for i in (np.ceil((dims[0]-kernel[0])/2), np.floor((dims[0]-kernel[0])/2))]
    pad_y = [np.int(i) for i in (np.ceil((dims[1]-kernel[1])/2), np.floor((dims[1]-kernel[1])/2))]
    b0 = np.zeros(dims)
    m  = np.zeros(dims)

    # Initialise Hankel matrix
    H = np.zeros((Nc, np.prod(kernel), Kx*Ky, Ne), dtype='complex')

    # Loop over echoes (Ne)
    for z in range(Ne):
        for i in range(Kx):
            for j in range(Ky):
                # Populate Hankel matrix
                H[:,:,i*Ky+j,z] = x[:,i:i+kernel[0],j:j+kernel[1],z].reshape((Nc,-1))

    # Reshape/permute H
    H = H.transpose((3,1,2,0)).reshape((-1,Kx*Ky*Nc))

    # Only keep svd singular vectors corresponding to above threshold singular values
    U,S,_ = np.linalg.svd(H, full_matrices=False)
    U = U[:,S>S[0]*eig_thresh]   

    # Get zero-paded kernel images
    U = U.reshape((Ne,kernel[0],kernel[1],-1))
    U = np.pad(U, ((0,0), pad_x, pad_y, (0,0)))
    U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U, axes=(1,2)), axes=(1,2)), axes=(1,2))
    U = U*np.prod(dims)/np.sqrt(np.prod(kernel))

    # Perform voxel-wise SVD on coil x component matrices, voxelwise, keeping 1st component
    for i in range(dims[0]):
        for j in range(dims[1]):
            W,S,_ = np.linalg.svd(U[:,i,j,:], full_matrices=False)
            b0[i,j] = np.polyfit(TE, np.unwrap(np.angle(W[:,0])), 1)[0]/(2*np.pi)
            m[i,j] = S[0]
    
    # Return masked b0
    #return b0*(np.abs(m)>mask_thresh)
    return b0
