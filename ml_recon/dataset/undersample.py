import numpy as np

def calc_k(lambda_probability, omega_probability):
    K = (1 - omega_probability) / (1 - omega_probability * lambda_probability)
    K = K.astype(float)
    return K
    
def apply_undersampling(index, prob_map, k_space, deterministic):
    rng = get_random_generator(index, deterministic)
    mask = get_mask_from_distribution(prob_map, rng)
    undersampled = k_space * np.expand_dims(mask, 1)
    return undersampled, np.expand_dims(mask, 1)
    
   
def get_random_generator(index, deterministic):
    if deterministic:
        rng = np.random.default_rng(index)
    else:
        rng = np.random.default_rng()
    return rng


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

def get_mask_from_distribution(prob_map, rng):
    prob_map[prob_map > 0.99] = 1
    (_, nx, _) = np.shape(prob_map)
    mask1d = rng.binomial(1, prob_map[:, 0, :]).astype(bool)
    mask = np.repeat(mask1d[:, np.newaxis, :], nx, axis=1)
    return mask
