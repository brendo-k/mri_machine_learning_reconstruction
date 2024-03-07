from ml_recon.pl_modules.pl_supervised import SupervisedReconModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from ml_recon.pl_modules.pl_unet import pl_Unet

import numpy as np
from ml_recon.dataset.undersample import get_mask_from_segregated_sampling, gen_pdf_bern, gen_pdf_columns, gen_pdf_columns_charlie
from functools import partial
import matplotlib.pyplot as plt


def main():
    pdf = gen_pdf_bern(128, 128, 1/8, 8, 10)
    pdf2 = gen_pdf_columns_charlie(128, 128, 1/8, 8, 10)

    #fig, ax = plt.subplots(1, 3)
    #ax[0].plot(pdf[0, :])
    #ax[0].set_title('mine')
    #ax[1].plot(pdf2[0, :])
    #ax[1].set_title('charlies')
    #ax[2].plot(np.abs(pdf2[0, :] - pdf[0, :]))

    #fig, ax = plt.subplots(1, 2)
    #mask_1 = np.random.binomial(1, pdf)
    #mask_2 = np.random.binomial(1, pdf2)

    #ax[0].imshow(mask_1, cmap='gray', interpolation='none')
    #ax[1].imshow(mask_2, cmap='gray', interpolation='none')

    #plt.show()

    pdf = np.repeat(pdf[np.newaxis, :, :], 4, axis=0)
    rng = np.random.default_rng()

    masks, probs = get_mask_from_segregated_sampling(pdf, rng, line_constrained = False)
    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(probs[0, :, :])
    ax[1,0].imshow(probs[1, :, :])
    ax[0,1].imshow(probs[2, :, :])
    ax[1,1].imshow(probs[3, :, :])

    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(masks[0, :, :])
    ax[1,0].imshow(masks[1, :, :])
    ax[0,1].imshow(masks[2, :, :])
    ax[1,1].imshow(masks[3, :, :])
    plt.show()


if __name__ == '__main__':
    main()
