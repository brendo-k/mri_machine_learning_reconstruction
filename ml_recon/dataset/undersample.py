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
    
def apply_undersampling(index, prob_map, k_space, line_constrained, deterministic, segregated):
    rng = get_random_generator(index, deterministic)
    if not segregated:
        mask = get_mask_from_distribution(prob_map, rng, line_constrained)
    else:
        mask, _ = get_mask_from_segregated_sampling(prob_map, rng, line_constrained)
        
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
        inverse_total = nx - nx*one_over_R
        inverse_sum = nx - probability_sum - center_square
        scaling_factor = inverse_total / inverse_sum
        inv_prob_map = (1 - prob_map) * scaling_factor
        prob_map = 1 - inv_prob_map

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

    if probability_sum > probability_total:
        scaling_factor = probability_total / probability_sum
        prob_map = prob_map*scaling_factor
    else:
        inverse_total = nx*ny - nx*ny*delta
        inverse_sum = nx*ny - probability_sum - c_sq * c_sq
        scaling_factor = inverse_total / inverse_sum
        prob_map = 1 - (1 - prob_map)*scaling_factor
        # inverse scale_factor

    prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2, nx // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1

    assert np.isclose(prob_map.mean(), delta, 1e-2, 0)
    assert prob_map.max() <= 1 and prob_map.min() >= 0

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


def get_mask_from_segregated_sampling(prob_map, rng, line_constrained):
    prob_map[prob_map > 0.999] = 1
    init_prob_map = prob_map[0]
    omega = 0.5
    masks = np.zeros_like(prob_map, dtype=bool)
    new_probs = np.zeros_like(prob_map)
    if line_constrained:
        for i in range((prob_map.shape[0])):
            (_, nx, _) = np.shape(prob_map)
            mask1d = rng.binomial(1, init_prob_map[:, 0, :]).astype(bool)
            masks[i, :, :] = np.repeat(mask1d[:, np.newaxis, :], nx, axis=1)

            sampled = np.any(masks[:, 0, :] == True, axis=0).astype(np.int32)
            init_prob_map[sampled == 1] = init_prob_map[sampled == 1] * omega
            init_prob_map[sampled == 0] = init_prob_map[sampled == 0] * (1 - omega * sampled.mean())/(1-sampled.mean())
    else:
        for i in range(1, prob_map.shape[0] + 1):
            if i == 1:
                masks[i-1, :, :] = rng.binomial(1, init_prob_map).astype(bool)
                new_probs[i - 1, :, :] = init_prob_map
            else:
                sampled = np.any(masks[:, :, :] == True, axis=0)
                new_prob_map = prob_map[0].copy()

                expected_value_sampled = 1/omega - (1/omega)*np.power(1 - omega * init_prob_map, i)
                expected_value_sampled_prev = 1/omega - (1/omega)*np.power(1 - omega * init_prob_map, i-1)
                
                # expected value is less than one. Can safely reduce sampled probability and 
                # increase non-sampled
                index = expected_value_sampled < 1 
                new_prob_map[index & sampled] = init_prob_map[index & sampled] * omega
                new_prob_map[index & ~sampled] = init_prob_map[index & ~sampled] * ((1 - omega * expected_value_sampled_prev[index & ~sampled]) / (1 - expected_value_sampled_prev[index & ~sampled]))

                # expected value is over 1 in these locations! Have to change sampling equation
                # for these voxels
                index = expected_value_sampled > 1 
                new_prob_map[index & sampled] = (expected_value_sampled_prev[index & sampled] - 1 + init_prob_map[index & sampled]) / (expected_value_sampled_prev[index & sampled])
                new_prob_map[index & ~sampled] = 1
            
                sampled_mask = rng.binomial(1, new_prob_map).astype(bool)
                masks[i - 1, :, :] = sampled_mask
                new_probs[i - 1, :, :] = new_prob_map

    return masks, new_probs
