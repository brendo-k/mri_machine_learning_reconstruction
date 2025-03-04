import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def calc_k(lambda_probability, omega_probability):
    K = (1 - omega_probability) / (1 - omega_probability * lambda_probability)
    K = K.astype(float)
    return K
    
def apply_undersampling_from_dist(
        index: int,  # index is neeed here to have determenistic (seed for rng)
        prob_map, 
        k_space, 
        line_constrained: bool, 
) -> Tuple[NDArray[np.complex64], NDArray[np.bool_]]:

    rng = get_random_generator(index)
    mask = get_mask_from_distribution(prob_map, rng, line_constrained)
    undersampled = k_space * np.expand_dims(mask, 1)
    return undersampled, np.expand_dims(mask, 1)
    
   
def get_random_generator(index):
    rng = np.random.default_rng(index)
    return rng


def gen_pdf_columns(nx, ny, one_over_R, poylnomial_power, center_square):
# generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv = np.linspace(-1, 1, nx)
    r = np.abs(xv)
    # normalize to 1
    r = r / (np.max(r) + 1/nx)
    prob_map = (1 - r) ** poylnomial_power
    prob_map[prob_map > 1] = 1

    prob_map = np.tile(prob_map, (ny, 1))
    prob_map = scale_pdf(prob_map, 1/one_over_R, center_square, True) 
    return prob_map


def gen_pdf_bern(nx, ny, delta, p, c_sq):
    # generates 2D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= (np.max(r) + 2/ny)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[prob_map < 0] = 0

    prob_map = scale_pdf(prob_map, 1/delta, c_sq, False)

    assert np.isclose(prob_map.mean(), delta, 1e-2, 0)
    assert prob_map.max() <= 1 and prob_map.min() >= 0

    return prob_map


def get_mask_from_distribution(prob_map: NDArray[np.float32], rng, line_constrained) -> NDArray[np.bool_]:
    prob_map[prob_map > 0.999] = 1
    if line_constrained:
        (_, ny, nx) = np.shape(prob_map)
        mask1d = rng.binomial(1, prob_map[:, 0, :]).astype(bool)
        mask = np.repeat(mask1d[:, np.newaxis, :], ny, axis=1)
    else:
        mask = rng.binomial(1, prob_map).astype(bool)
    return mask

def gen_pdf_columns_charlie(nx, ny, one_over_R, poylnomial_power, c_sq):
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



def scale_pdf(input_prob, R, center_square, line_constrained=False):
    """
    This function takes an input pdf and an R value and scales the pdf to have 
    a mean sampling probability to be R. It functions by lowering the number of
    sampled locations if the input probability is over R and lowering the inverse
    (number of zeros) if the input probability is less that R.

    Args:
        input_prob (np.array): input probability
        R (float): target R value
        center_square (int): center region to be kept fully sampled
        line_constrained (bool): flag for line constrained or 2d sampling

    Returns:
        (np.array) scaled pdf
    """

    prob_map = input_prob.copy() 
    if not line_constrained:
        shape = prob_map.shape
        ny = shape[-2]
        nx = shape[-1]
        center = [ny//2, nx//2]
        center_y_slice = slice(center[0] - center_square//2, center[0] + center_square//2)
        center_x_slice = slice(center[1] - center_square//2, center[1] + center_square//2)
        if prob_map.ndim == 2: 
            prob_map = np.expand_dims(prob_map, axis=0)

        prob_map[:, center_y_slice, center_x_slice] = 0
        
        probability_sum = prob_map.sum(axis=(-1, -2))
        
        probability_total = nx * ny * (1/R)
        probability_total -= center_square * center_square
        
        for i in range(probability_sum.size): 
            if probability_sum[i] > probability_total:
                scaling_factor = probability_total / probability_sum[i]
                prob_map[i, ...] = prob_map[i, ...]*scaling_factor
            else:
                # inverse scale_factor
                inverse_total = nx*ny - nx*ny*(1/R)
                inverse_sum = nx*ny - probability_sum[i] - center_square * center_square
                scaling_factor = inverse_total / inverse_sum
                prob_map[i, ...] = 1 - (1 - prob_map[i, ...])*scaling_factor

        prob_map[:, center_y_slice, center_x_slice] = 1

        if input_prob.ndim==2:
            prob_map = np.squeeze(prob_map)

    if line_constrained:
        nx = prob_map.shape[-1]
        center = [nx//2]
        center_x_slice = slice(center[0] - center_square//2, center[0] + center_square//2)
        prob_map[..., center_x_slice] = 0

        if prob_map.ndim == 2: 
            prob_map = np.expand_dims(prob_map, axis=0)

        probability_sum = prob_map.sum(-1)
        probability_sum = probability_sum[:, 0]

        probability_total = nx * (1/R)
        probability_total -= center_square

        for i in range(probability_sum.size): 
            if probability_sum[i] > probability_total: 
                scaling_factor = probability_total/probability_sum[i]
                prob_map = prob_map[i, ...] * scaling_factor
            else:
                inverse_total = nx - nx* (1/R)
                inverse_sum = nx - probability_sum[i] - center_square
                scaling_factor = inverse_total / inverse_sum
                prob_map[i, ...] = 1 - (1 - prob_map[i, ...]) * scaling_factor

        prob_map[..., center_x_slice] = 1
        if input_prob.ndim == 2:
            prob_map = np.squeeze(prob_map)


    return prob_map

def ssdu_gaussian_selection(initial_mask: NDArray[np.float32], std_scale=4, rho=0.4):

    ncol, nrow = initial_mask.shape
    
    center_kx = nrow//2
    center_ky = ncol//2

    acs_shape = 10
    temp_mask = np.copy(initial_mask)
    temp_mask[center_kx - acs_shape // 2:center_kx + acs_shape // 2,
    center_ky - acs_shape // 2:center_ky + acs_shape // 2] = 0

    loss_mask = np.zeros_like(initial_mask)
    count = 0
    required_points = int(np.ceil(np.sum(initial_mask[:]) * rho))
    remaning_points = required_points
    while np.sum(loss_mask) < required_points:

        indx = np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale, size=int(remaning_points))).astype(int)
        indy = np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale, size=int(remaning_points))).astype(int)

        valid_x = np.logical_and(indx >= 0, indx < nrow)
        valid_y = np.logical_and(indy >= 0, indy < ncol)

        indx = indx[valid_x & valid_y]
        indy = indy[valid_x & valid_y]

        points = temp_mask[indy, indx]
        loss_mask[indy, indx] = points
        remaning_points = required_points - np.sum(loss_mask)


    input_mask = initial_mask - loss_mask

    return input_mask, loss_mask
