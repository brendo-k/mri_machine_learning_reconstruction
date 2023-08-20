import numpy as np
from torch.utils.data import Dataset

class UndersamplingDecorator(Dataset):
    def __init__(self, 
                 dataset, 
                 R:int = 4, 
                 R_hat:int = 2, 
                 acs_width = 10, 
                 deterministic:bool = True, 
                 transforms = None):
        super().__init__()

        self.dataset = dataset
        self.R = R
        self.R_hat = R_hat
        self.determinestic = deterministic
        self.acs_width = acs_width
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        k_space = self.dataset[index]
        assert not np.isnan(k_space).any()
        
        omega_probability, undersampled = self.apply_undersampling(index, k_space, R=self.R, deterministic=self.determinestic)
        lambda_probability, double_undersampled = self.apply_undersampling(index, undersampled, R=self.R_hat, deterministic=False)

        K = self.calc_k(lambda_probability, omega_probability)

        data = (double_undersampled, undersampled, k_space, K)
        if self.transforms:
            data = self.transforms(data)
        return data

    def calc_k(self, lambda_probability, omega_probability):
        one_minus_eps = 1 - 1e-3
        lambda_probability[lambda_probability > one_minus_eps] = one_minus_eps
        K = (1 - omega_probability) / (1 - omega_probability * lambda_probability)
        K = K.astype(float)
        return K
    
    def apply_undersampling(self, index, k_space, deterministic, R):
        rng = self.get_random_generator(index, deterministic)
        prob_map = self.gen_pdf_columns(k_space.shape[-1], k_space.shape[-2], 1/R, 8, self.acs_width)
        mask = self.get_mask_from_distribution(prob_map, rng)
        undersampled =  k_space * mask
        return prob_map, undersampled
    
    #SOMETHING TO THINK ABOUT: SHOULD WE ALWAYS HAVE THE SEED TO BE THE INDEX IN 
    #DTERMINISTIC CASE?
    def get_random_generator(self, index, deterministic):
        if deterministic:
            rng = np.random.default_rng(index)
        else:
            rng = np.random.default_rng()
        return rng


    def gen_pdf_columns(self, nx, ny, one_over_R, poylnomial_power, c_sq):
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
        eta = 1e-3
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

    def get_mask_from_distribution(self, prob_map, rng_generator):
        prob_map[prob_map > 0.99] = 1
        (nx, _) = np.shape(prob_map)
        mask1d = rng_generator.binomial(1, prob_map[0:1])
        mask = np.repeat(mask1d, nx, axis=0)
        return np.array(mask, dtype=bool)