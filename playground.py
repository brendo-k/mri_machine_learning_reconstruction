from ml_recon.dataset.undersample import gen_pdf_columns, gen_pdf_columns_charlie, get_mask_from_distribution, get_random_generator
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nx = 128
    ny = 128
    p = 8
    delta = 1/8
    c_sq = 10
    
    pdf_mine = gen_pdf_columns(nx, ny, delta, p, c_sq)
    pdf_charlie = gen_pdf_columns_charlie(nx, ny, delta, p, c_sq)

    pdf_mine = np.tile(pdf_mine[np.newaxis, :, :], (4, 1, 1))
    pdf_charlie = np.tile(pdf_charlie[np.newaxis, :, :], (4, 1, 1))

    rng = get_random_generator(0, deterministic=True)

    mask_mine = get_mask_from_distribution(pdf_mine, rng, line_constrained=True)
    mask_charlie = get_mask_from_distribution(pdf_mine, rng, line_constrained=True)

    print(mask_mine.sum()/mask_mine.size)
    print(mask_charlie.sum()/mask_charlie.size)

    plt.figure()
    plt.imshow(pdf_mine[0])
    plt.figure()
    plt.imshow(pdf_charlie[0])
    plt.show()

    plt.figure()
    plt.imshow(mask_mine[0])
    plt.figure()
    plt.imshow(mask_charlie[0])
    plt.show()
