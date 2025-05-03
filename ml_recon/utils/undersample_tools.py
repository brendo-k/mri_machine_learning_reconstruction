import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union


def apply_undersampling_from_dist(
        seed: int,  # index is neeed here to have determenistic (seed for rng)
        prob_map, 
        k_space, 
) -> Tuple[NDArray[np.complex64], NDArray[np.bool_]]:

    mask = get_mask_from_distribution(prob_map, seed)
    undersampled = k_space * np.expand_dims(mask, 1)
    return undersampled, np.expand_dims(mask, 1)

def is_line_constrained(prob_map: NDArray[np.float32]):
    middle_slice = prob_map[0, :, prob_map.shape[-1]//2]
    if (middle_slice == middle_slice[0]).all():
        line_constrained = True
    else:
        line_constrained = False
    return line_constrained
    

def get_mask_from_distribution(
    prob_map: NDArray[np.float32], 
    seed: Union[int, None]
) -> NDArray[np.bool_]:

    rng = np.random.default_rng(seed)

    prob_map[prob_map > 0.999] = 1

    if is_line_constrained(prob_map):
        (_, ny, _) = np.shape(prob_map)
        mask1d = rng.binomial(1, prob_map[:, 0, :]).astype(bool)
        mask = np.repeat(mask1d[:, np.newaxis, :], ny, axis=1)
    else:
        mask = rng.binomial(1, prob_map).astype(bool)
    return mask

def gen_pdf_columns(nx, ny, one_over_R, poylnomial_power, c_sq):
# generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv = np.linspace(-1, 1, nx)
    r = np.abs(xv)
    # normalize to 1
    r /= np.max(r)
    prob_map = (1 - r) ** poylnomial_power
    prob_map[prob_map > 1] = 1
    prob_map[ny // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1

    a = -1
    b = 1
    eta = 1e-7
    ii = 1
    while 1:
        c = (a + b) / 2
        prob_map = (1 - r) ** poylnomial_power + c
        prob_map[prob_map > 1] = 1
        prob_map[prob_map < 0] = 0
        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1
        delta_current = np.mean(prob_map)
        if one_over_R > delta_current + eta:
            a = c
        elif one_over_R < delta_current - eta:
            b = c
        else:
            break
        ii += 1
        if ii == 100:
            break
    prob_map = np.tile(prob_map, (ny, 1))

    return prob_map


def ssdu_part_one_contrast(
    initial_mask: NDArray[np.float32], 
    std_scale: float,
    rho: float, 
    rng: np.random.Generator
):
    ncol, nrow = initial_mask.shape
    
    center_kx = nrow//2
    center_ky = ncol//2

    acs_shape = 10
    temp_mask = np.copy(initial_mask)
    temp_mask[center_kx - acs_shape // 2:center_kx + acs_shape // 2,
    center_ky - acs_shape // 2:center_ky + acs_shape // 2] = 0

    loss_mask = np.zeros_like(initial_mask)
    required_points = int(np.ceil(np.sum(initial_mask[:]) * rho))
    remaning_points = required_points
    while np.sum(loss_mask) < required_points:

        indx = np.round(rng.normal(loc=center_kx, scale=(nrow - 1) / std_scale, size=int(remaning_points))).astype(int)
        indy = np.round(rng.normal(loc=center_ky, scale=(ncol - 1) / std_scale, size=int(remaning_points))).astype(int)

        valid_x = np.logical_and(indx >= 0, indx < nrow)
        valid_y = np.logical_and(indy >= 0, indy < ncol)

        indx = indx[valid_x & valid_y]
        indy = indy[valid_x & valid_y]

        points = temp_mask[indy, indx]
        loss_mask[indy, indx] = points
        remaning_points = required_points - np.sum(loss_mask)

    input_mask = initial_mask - loss_mask

    return input_mask, loss_mask


def ssdu_gaussian_selection(
        initial_mask: NDArray[np.float32], 
        std_scale: float=4., 
        rho: float=0.4, 
        seed:Union[int, None] = None
):
    input_mask = []
    loss_mask = []
    rng = np.random.default_rng(seed)
    for i in range(initial_mask.shape[0]):
        input, loss = ssdu_part_one_contrast(initial_mask[i], std_scale, rho, rng)
        input_mask.append(np.expand_dims(input, 0))
        loss_mask.append(np.expand_dims(loss, 0))

    input_mask = np.stack(input_mask, axis=0)
    loss_mask = np.stack(loss_mask, axis=0)
    return input_mask, loss_mask

def gen_pdf_bern(nx, ny, delta, p, c_sq):
    # generates 2D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), sparse=False, indexing='xy')
    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= np.max(r)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

    a = -1
    b = 1
    eta = 1e-3
    ii = 1
    while 1:
        c = a / 2 + b / 2
        prob_map = (1 - r) ** p + c
        prob_map[prob_map > 1] = 1
        prob_map[prob_map < 0] = 0

        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        delta_current = np.mean(prob_map)
        if delta > delta_current + eta:
            a = c
        elif delta < delta_current - eta:
            b = c
        else:
            break

        ii += 1
        if ii == 100:
            print('gen_pdf_bern did not converge after 100 iterations')
            break

    return prob_map
