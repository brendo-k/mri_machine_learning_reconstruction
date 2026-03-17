from __future__ import annotations

import numpy as np
import torch
from ml_recon.utils.image_processing import ifft_2d_img, fft_2d_img





def _calibration_bounds(size: int, width: int) -> tuple[int, int]:
    if size <= 1:
        return (0, 1)
    center = size // 2
    offset = (width - 1) // 2
    start = center - offset
    end = start + width
    return (start, end)


def _kernel_dims(sx: int, sy: int, k: int, nc: int) -> list[int]:
    return [
        k if sx > 1 else 1,
        k if sy > 1 else 1,
        nc,
    ]


def _num_patch_dims(sx: int, sy: int) -> int:
    return int(sx > 1) + int(sy > 1) 


def espirit(X, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """

    sx, sy, nc = X.shape

    sxt = _calibration_bounds(sx, r)
    syt = _calibration_bounds(sy, r)

    C = X[sxt[0]:sxt[1], syt[0]:syt[1], :].astype(np.complex64)

    p = _num_patch_dims(sx, sy)
    A = np.zeros(((r - k + 1) ** p, k ** p * nc), dtype=np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            block = C[xdx:xdx + k, ydx:ydx + k, :].astype(np.complex64)
            A[idx, :] = block.flatten()
            idx += 1

    _, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    n = int(np.sum(S >= t * S[0]))
    V = V[:, :n]

    kxt = _calibration_bounds(sx, k)
    kyt = _calibration_bounds(sy, k)

    kernels = np.zeros((*X.shape, n), dtype=np.complex64)
    kerdims = _kernel_dims(sx, sy, k, nc)
    for idx in range(n):
        kernels[kxt[0]:kxt[1], kyt[0]:kyt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    axes = (0, 1)
    kerimgs = np.zeros((*X.shape, n), dtype=np.complex64)
    scale = np.sqrt(sx * sy ) / np.sqrt(k ** p)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1,  jdx, idx].conj()
            kerimgs[:, :, jdx, idx] = fft_2d_img(ker, axes) * scale

    maps = np.zeros((*X.shape, 1), dtype=np.complex64)
    for idx in range(sx):
        for jdx in range(sy):
            Gq = kerimgs[idx, jdx, :, :]
            u, s, _ = np.linalg.svd(Gq, full_matrices=True)
            if s[0] ** 2 > c:
                maps[idx, jdx, :, 0] = u[:, 0]

    return maps


def espirit_torch(X, k, r, t, c, device: str | torch.device | None = None):
    if isinstance(X, np.ndarray):
        tensor = torch.from_numpy(X.astype(np.complex64, copy=False))
    elif isinstance(X, torch.Tensor):
        tensor = X.to(torch.complex64)
    else:
        raise TypeError(f"Unsupported input type: {type(X)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    tensor = tensor.to(device)

    sx, sy, nc = tensor.shape

    sxt = _calibration_bounds(sx, r)
    syt = _calibration_bounds(sy, r)

    C = tensor[sxt[0]:sxt[1], syt[0]:syt[1], :]

    p = _num_patch_dims(sx, sy)
    A = torch.zeros(((r - k + 1) ** p, k ** p * nc), dtype=torch.complex64, device=device)

    # move channels to front and add batch dim:
    patches = C.permute(2, 0, 1).unsqueeze(0)  # shape: (1, ch, X, Y)

    # create sliding blocks of size k along the 3 spatial dims
    for dim in [2, 3]:
        if patches.size(dim) >= k:
            patches = patches.unfold(dim, k, 1)
    # shape: (1, ch, X-k+1, Y-k+1, k, k)

    print(patches.shape)  # -> (1, ch, X-k+1, Y-k+1, k, k)

    # reorder and collapse to rows
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (1, X-k+1, Y-k+1,  ch, k,k)
    A = patches.reshape(-1, nc * k * k)            # (num_patches, ch*k^3)

    print("A.shape:", A.shape)  # -> ((X-k+1)*(Y-k+1), ch*k^3)

    _, S, VH = torch.linalg.svd(A, full_matrices=True)
    V = VH.transpose(0, 1).conj()

    n = int((S >= t * S[0]).sum().item())
    V = V[:, :n]

    kxt = _calibration_bounds(sx, k)
    kyt = _calibration_bounds(sy, k)

    kernels = torch.zeros((*tensor.shape, n), dtype=torch.complex64, device=device)
    kerdims = _kernel_dims(sx, sy, k, nc)
    for idx in range(n):
        kernels[kxt[0]:kxt[1], kyt[0]:kyt[1], :, idx] = V[:, idx].reshape(kerdims)

    # params already defined: sx, sy, sz, k, p, nc, n, device, tensor
    axes = (0, 1)
    scale = torch.sqrt(torch.tensor([sx * sy])) / torch.sqrt(torch.tensor([k ** p]))
    scale = scale.to(device)

    # kernels: shape (sx, sy, sz, nc, n), complex dtype
    # do flip + conjugate for all kernels at once, then FFT over spatial dims
    ker_flipped = kernels.flip(axes).conj()                 # (sx,sy,sz,nc,n)
    kerimgs = torch.fft.fftn(ker_flipped, dim=axes) * scale # (sx,sy,sz,nc,n)
    print(f'kerimgs shape: {kerimgs.shape}')  # -> (sx, sy, sz, nc, n)
    maps = get_eigen_values(kerimgs, c)


    return maps

def get_eigen_values(kerimgs, c):
    sx, sy, nc, n = kerimgs.shape
    device = kerimgs.device
    batch = sx * sy
    G = kerimgs.reshape(batch, nc, n)
    maps = torch.zeros((batch, nc, 1), dtype=kerimgs.dtype, device=device)

    use_eigh = False   # prefer for nc << n
    chunk = 2**16       # tune this

    with torch.no_grad():
        for start in range(0, batch, chunk):
            end = min(batch, start + chunk)
            Gc = G[start:end]   # (b, nc, n)

            if use_eigh:
                GGt = Gc @ Gc.conj().transpose(-1, -2)   # (b, nc, nc)
                w, Q = torch.linalg.eigh(GGt)            # (b, nc), (b, nc, nc)
                idx = torch.argsort(w, dim=-1, descending=True)
                bidx = torch.arange(w.shape[0], device=device)[:, None]
                w_sorted = w[bidx, idx]
                Q_sorted = Q[bidx[..., None], idx[..., None, :]]
                S = torch.sqrt(torch.clamp(w_sorted, min=0.0))
                r = min(nc, n)
                U_thin = Q_sorted[:, :, :r]              # (b, nc, r)
                S_thin = S[:, :r]                        # (b, r)
                keep = (S_thin.square() > c)             # (b, r)
                U_masked = U_thin * keep.unsqueeze(1)    # (b, nc, r)
                maps[start:end, :, :r] = U_masked
            else:
                U, S, Vh = torch.linalg.svd(Gc, full_matrices=False)
                r = U.shape[-1]
                keep = (S.square() > c)
                U_masked = U * keep.unsqueeze(1)
                maps[start:end, :, 0] = U_masked[..., 0]

    # reshape back to spatial grid:
    maps = maps.reshape(sx, sy, nc, n)
    return maps

def espirit_gpu(X, k, r, t, c):
    return espirit_torch(X, k, r, t, c, device="cuda")


def espirit_proj(x, esp):
    """
    Construct the projection of multi-channel image x onto the range of the ESPIRiT operator. Returns the inner
    product, complete projection and the null projection.

    Arguments:
      x: Multi channel image data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      esp: ESPIRiT operator as returned by function: espirit

    Returns:
      ip: This is the inner product result, or the image information in the ESPIRiT subspace.
      proj: This is the resulting projection. If the ESPIRiT operator is E, then proj = E E^H x, where H is
            the hermitian.
      null: This is the null projection, which is equal to x - proj.
    """
    if isinstance(x, np.ndarray):
        ip = np.zeros(x.shape, dtype=np.complex64)
        proj = np.zeros(x.shape, dtype=np.complex64)
    elif isinstance(x, torch.Tensor):
        ip = torch.zeros_like(x, dtype=torch.complex64)
        proj = torch.zeros_like(x, dtype=torch.complex64)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    for qdx in range(esp.shape[4]):
        for pdx in range(esp.shape[3]):
            ip[:, :, :, qdx] = ip[:, :, :, qdx] + x[:, :, :, pdx] * esp[:, :, :, pdx, qdx].conj()

    for qdx in range(esp.shape[4]):
        for pdx in range(esp.shape[3]):
            proj[:, :, :, pdx] = proj[:, :, :, pdx] + ip[:, :, :, qdx] * esp[:, :, :, pdx, qdx]

    return (ip, proj, x - proj)


def compare_espirit_cpu_gpu(
    k_space_slice: np.ndarray,
    k: int,
    r: int,
    t: float,
    c: float,
    is_plot: bool = False
) -> dict[str, float]:
    if k_space_slice.ndim != 3:
        raise ValueError(f"Expected [coils, h, w], got shape {k_space_slice.shape}")

    X = np.transpose(k_space_slice, (1, 2, 0)).astype(np.complex64, copy=False)
    print(X.shape)
    cpu_maps = espirit(X, k=k, r=r, t=t, c=c)
    gpu_maps = espirit_gpu(X, k=k, r=r, t=t, c=c).detach().cpu().numpy()
    print(f'CPU maps shape: {cpu_maps.shape}, GPU maps shape: {gpu_maps.shape}')

    diff = cpu_maps - gpu_maps
    cpu_norm = np.linalg.norm(cpu_maps)
    diff_norm = np.linalg.norm(diff)

    if is_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(np.abs(cpu_maps[:, :, 0, 0]), cmap='gray')
        ax[0].set_title('CPU ESPIRiT Map (abs)')
        ax[1].imshow(np.abs(gpu_maps[:, :, 0, 0]), cmap='gray')
        ax[1].set_title('GPU ESPIRiT Map (abs)')
        ax[2].imshow(np.abs(diff[:, :, 0, 0]), cmap='gray')
        ax[2].set_title('Difference (abs)')
        plt.show()

    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "relative_l2_diff": float(diff_norm / max(cpu_norm, 1e-12)),
    }
