import numpy as np

def gen_pdf(line_constrained, nx, ny, one_over_r, poly_power, center_square):
    if line_constrained:
        pdf_method = gen_pdf_columns
    else:
        pdf_method = gen_pdf_bern 
    
    return pdf_method(nx, ny, one_over_r, poly_power, center_square)


def calc_k(lambda_probability, omega_probability):
    K = (1 - omega_probability) / (1 - omega_probability * lambda_probability)
    K = K.astype(float)
    return K
    
def apply_undersampling(index, prob_map, k_space, line_constrained, deterministic):
    rng = get_random_generator(index, deterministic)
    mask = get_mask_from_distribution(prob_map, rng, line_constrained)
    undersampled = k_space * np.expand_dims(mask, 1)
    return undersampled, np.expand_dims(mask, 1)
    
   
def get_random_generator(index, deterministic):
    if deterministic:
        rng = np.random.default_rng(index)
    else:
        rng = np.random.default_rng()
    return rng


def gen_pdf_columns(nx, ny, one_over_R, poylnomial_power, center_square):
# generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv = np.linspace(-1, 1, nx)
    r = np.abs(xv)
    # normalize to 1
    r /= np.max(r)
    prob_map = (1 - r) ** poylnomial_power
    prob_map[prob_map > 1] = 1
    prob_map[nx // 2 - center_square // 2:nx // 2 + center_square // 2] = 0

    probability_sum = prob_map.sum()

    probability_total = nx * one_over_R
    probability_total -= center_square

    if probability_sum > probability_total: 
        scaling_factor = probability_total/probability_sum
        prob_map = prob_map/scaling_factor
    else:
        inverse_total = nx - probability_total
        inverse_sum = nx - probability_sum
        scaling_factor = inverse_total / inverse_sum
        prob_map = 1 - (1 - prob_map)*scaling_factor

    prob_map[nx // 2 - center_square // 2:nx // 2 + center_square // 2] = 1
    prob_map = np.tile(prob_map, (ny, 1))
    return prob_map

def gen_pdf_columns_charlie(nx, ny, delta, p, c_sq):
    # generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, 1), np.linspace(-1, 1, ny), sparse=False, indexing='xy')
    r = np.abs(yv)
    r /= np.max(r)
    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

    a = -1
    b = 1
    eta = 1e-3
    ii = 1
    while 1:
        c = (a + b) / 2
        prob_map = (1 - r) ** p + c
        prob_map[prob_map > 1] = 1
        prob_map[prob_map < 0] = 0
        prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1
        delta_current = np.mean(prob_map)
        if delta > delta_current + eta:
            a = c
        elif delta < delta_current - eta:
            b = c
        else:
            break
        ii += 1
        if ii == 100:
            warnings.warn('gen_pdf_columns did not converge after 100 iterations')
            break
    prob_map = np.repeat(prob_map, nx, axis=1)
    prob_map = np.rot90(prob_map)
    return prob_map

def gen_pdf_bern(nx, ny, delta, p, c_sq):
    # generates 2D polynomial variable density with sampling factor delta, fully sampled central square c_sq
    xv, yv = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    r = np.sqrt(xv ** 2 + yv ** 2)
    r /= np.max(r)

    prob_map = (1 - r) ** p
    prob_map[prob_map > 1] = 1
    prob_map[prob_map < 0] = 0

    prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2, nx // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 0
    
    probability_sum = prob_map.sum()
    
    probability_total = nx * ny * delta
    probability_total -= c_sq * c_sq

    if probability_total < probability_sum:
        scaling_factor = probability_total / probability_sum
        prob_map = prob_map*scaling_factor
    else:
        inverse_total = nx*ny - probability_total
        inverse_sum = nx*ny - probability_sum
        scaling_factor = inverse_total / inverse_sum
        prob_map = 1 + (1 - prob_map)*scaling_factor
        # inverse scale_factor

    prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2, nx // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1

    assert np.isclose(prob_map.mean(), delta, 1e-3, 1e-3)
    assert prob_map.max() < 1 and prob_map.min() > 0

    return prob_map


def get_mask_from_distribution(prob_map, rng, line_constrained):
    prob_map[prob_map > 0.999] = 1
    if line_constrained:
        (_, nx, _) = np.shape(prob_map)
        mask1d = rng.binomial(1, prob_map[:, 0, :]).astype(bool)
        mask = np.repeat(mask1d[:, np.newaxis, :], nx, axis=1)
    else:
        mask = rng.binomial(1, prob_map).astype(bool)
    return mask
