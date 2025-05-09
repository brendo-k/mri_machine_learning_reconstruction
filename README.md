# Publications and Conferences from this Code
**Brenden Kadota**, Charles Millard, Mark Chiew. Joint Multi-Contrast Image Reconstruction with Self-Supervised Learning, International Conference for Magnetic Resonance in Medicine, Singapore, 2024 (Conference)

**B. Kadota**, C. Millard, M. Chiew. Learned K-Space Partitioning for Improved Dual-Domain Self-Supervised Image Reconstruction, International Conference for Magnetic Resonance in Medicine, Hawaii, 2025 (Conference)

**B. Kadota**, C. Millard, M. Chiew. Learned k-space Partitioning for Optimized self-supervised MRI Reconstruction, Conference in IEEE Engineering in Medicine and Biology, 2025 (Conference Paper)

**B. Kadota**, C. Millard, M. Chiew. Learned k-space Partitioning for Optimized Multi-Contrast Self-Supervised MRI Reconstruction,  IEEE Transactions in Medical Imaging, 2025 (in preparation)

## Getting Started
Below is how to get started on a simulated BRaTS dataset

### Environment
We have a few required packages for installation.
```
numpy
pytorch
pytorch-lightning
wandb (not required)
scipy
einops
```
You can also install all dependencies using
```bash
pip install -r requirements.txt
```

### Training
To train a network, run

```bash
python train_network.py -c=config_path
```

where config_name is name of one of the configuration files in the configs folder. For instance,  
```bash
python train.py -c=configs/ssl_learn_partitioning.yaml
```
trains according to the configuration in the ssl_learn_partitioning.yaml file. 
We have provided an example configuration file for each of the training methods in the paper. In the config file there is a field
data_dir which specifies the path to the training data. 

To change datasets (fastMRI, BRaTS, m4raw) change the path to the directories. Additionally, 
change the dataset argument in the configuration file.

To change the number of contrast jointly reconstructed use the --contrast argument. Ie. 
```
--contrast t1
```
reconstructs just a t1 while 
```
--contrasts t1 t2 flair t1ce
```
reconstructs all four contrasts jointly.
 
The configuration file, pytorch-lightnign checkpoint, and model parameters are saved in the log directory.

All parameters in the config file can be configured through the command line too. The command line overrides the configuration file arugments.

Logging is done with wandb. If wandb isn't installed default to tensorboard. 

To setup wandb [go here](https://docs.wandb.ai/quickstart/).

### Testing

To test a network, run 

```bash
python test_model.py log_location
```
where log_location is the location of the saved network. For instance,

```bash
python test_model.py logs/cpu/1D_partitioned_ssdu/20221214-120656
```


### Data
We simulate our own multi-contrast dataset for the BRaTS dataset. The BRaTS dataset can be downloaded [https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1](here)
We have 2 coil files in this directory, coil_compressed_10.npy and coil_compressed_4, which are
10 and 4 channel sensetivies SVD compressed form 32 channels. To simulate a multi-channel brats dataset use the simulate_k_space.py function. 
```bash
python simulate_k_space.py /path/to/brats/data/ /save/path/ coil_compressed_10 noise_level
```
The noise_level used in the paper was 1e-2.


## Contact

If you have any questions/comments, please feel free to contact Brenden
Kadota at [brenden.kadota@mail.utoronto.ca](brenden.kadota@mail.utoronto.ca) or Mark Chiew at
[mark.chiew@utoronto.ca](mark.chiew@utoronto.ca)

## Citation
If you use this code, please cite our article:
```
@inproceedings{kadota2025framework,
    author = {Brenden Kadota and Charles Millard and Mark Chiew},
    title = {Learned k-space Partitioning for Optimized Self-Supervised MRI Reconstruction},
    booktitle = {Proceedings of the 2025 IEEE Engineering in Medicine and Biology Conference (EMBC)},
    year = {2025},
}
```

## Copyright and Licensing

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the file GNU_General_Public_License,
and is also availabe [here](https://www.gnu.org/licenses/).

