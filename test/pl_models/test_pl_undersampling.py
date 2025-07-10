import pytest
import os
import h5py
import tempfile
import numpy as np
import torch

from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.dataset.test_dataset import TestDataset

BATCH_SIZE = 2
DATA_SIZE = (8, 4, 3, 128, 128)

@pytest.fixture
def temp_h5_directories(scope='session'):
    # Create a temporary directory to hold the structure
    with tempfile.TemporaryDirectory() as temp_dir:
        sub_dirs = ['train', 'test', 'val']
        file_identifiers = [1, 2]
        contrasts = ['t1', 't2', 'flair', 't1ce']

        # Create subdirectories and files
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(temp_dir, sub_dir)
            for identifier in file_identifiers:
                file_dir = os.path.join(sub_dir_path, str(identifier))
                os.makedirs(file_dir)
                file_path = os.path.join(sub_dir_path, str(identifier), f"{sub_dir}_{identifier}.h5")
                with h5py.File(file_path, 'w') as h5_file:
                    data_shape = DATA_SIZE
                    complex_data = (
                        np.random.rand(*data_shape) + 1j * np.random.rand(*data_shape)
                    ).astype(np.complex64)
                    complex_data[..., 64, 64] = 5 + 5j
                    h5_file.create_dataset("k_space", data=complex_data)
                    h5_file.create_dataset("ground_truth", data=np.abs(complex_data[:, :, 0, :, :]))
                    h5_file.create_dataset("contrasts", data=contrasts)

        # Yield the path to the temporary directory
        yield temp_dir

@pytest.fixture
def build_supervised_datamodule(temp_h5_directories):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=False,
            R=4
            ) 
    data_module.setup('train')
    return data_module


@pytest.mark.parametrize("is_self_supervised", [True, False])
def test_inital_undersampling_R(temp_h5_directories, is_self_supervised):
    """
    This tests ensures the inital undersampling of k-space has a consistent
    R value. 
    """
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=is_self_supervised,
            R=4
            ) 
    data_module.setup('train')

    for split_name, dataloader in {
        'train': data_module.train_dataloader(),
        'val': data_module.val_dataloader(),
        'test': data_module.test_dataloader()
    }.items():
        batch = next(iter(dataloader))
        undersampled = batch['undersampled']
        assert undersampled.ndim == 5
        assert undersampled.shape == (BATCH_SIZE,) + DATA_SIZE[1:]

        initial_mask = (undersampled != 0).to(torch.float32)


        given_R = 1/initial_mask.mean(dim=(-1, -2))
        print(given_R.dtype)
        print(torch.full_like(given_R, data_module.R).dtype)
        torch.testing.assert_close(
            given_R,
            torch.full_like(given_R, data_module.R), 
            atol=0.5, 
            rtol=0,
            msg=f"R value for {split_name} set is not as expected. Expected: {data_module.R}, Got: {given_R.mean().item()}"
            )



def test_supervisedMasks(build_supervised_datamodule):
    data_module = build_supervised_datamodule

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))
    test_batch = next(iter(data_module.test_dataloader()))

    undersampled_train = train_batch['undersampled']
    mask_train = train_batch['mask']
    initial_mask = undersampled_train != 0
    fully_sampled = train_batch['fs_k_space']

    val_batch = val_batch[0]
    undersampled_val = val_batch['undersampled']
    initial_mask_val = undersampled_val != 0
    mask_val = val_batch['mask']
    fully_sampled_val = val_batch['fs_k_space']

    undersampled_test = test_batch[0]['undersampled']
    initial_mask_test = undersampled_test != 0
    mask_test = test_batch[0]['mask']
    fully_sampled_test = test_batch[0]['fs_k_space']

    assert initial_mask.sum() > 0
    torch.testing.assert_close(undersampled_train, mask_train*fully_sampled)
    torch.testing.assert_close(mask_train, (initial_mask != 0).to(torch.float32))

    assert initial_mask_val.sum() > 0
    torch.testing.assert_close(undersampled_val, mask_val*fully_sampled_val)
    torch.testing.assert_close(mask_val, (initial_mask_val != 0).to(torch.float32))

    assert initial_mask_test.sum() > 0
    torch.testing.assert_close(undersampled_test, mask_test*fully_sampled_test)
    torch.testing.assert_close(mask_test, (initial_mask_test != 0).to(torch.float32))

@pytest.mark.parametrize("is_self_supervised", [True, False])
def test_scaling(temp_h5_directories, is_self_supervised):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=is_self_supervised,
            R=4
            ) 
    data_module.setup('train')

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))
    test_batch = next(iter(data_module.test_dataloader()))

    undersampled = train_batch['undersampled']
    mask = train_batch['mask']
    initial_mask = undersampled != 0
    fully_sampled = train_batch['fs_k_space']

    undersampled_val = val_batch[0]['undersampled']
    initial_mask_val = undersampled_val != 0
    mask_val = val_batch[0]['mask']
    fully_sampled_val = val_batch[0]['fs_k_space']

    undersampled_test = test_batch[0]['undersampled']
    initial_mask_test = undersampled_test != 0
    mask_test = test_batch[0]['mask']
    fully_sampled_test = test_batch[0]['fs_k_space']

    print(undersampled.shape)

    torch.testing.assert_close(undersampled.abs().amax((-1, -2, -3)), torch.ones(undersampled.shape[:2]))
    torch.testing.assert_close(fully_sampled.abs().amax((-1, -2, -3)), torch.ones(fully_sampled.shape[:2]))

    torch.testing.assert_close(undersampled_val.abs().amax((-1, -2, -3)), torch.ones(undersampled_val.shape[:2]))
    torch.testing.assert_close(fully_sampled_val.abs().amax((-1, -2, -3)), torch.ones(undersampled_val.shape[:2]))

    torch.testing.assert_close(undersampled_test.abs().amax((-1, -2, -3)), torch.ones(undersampled_test.shape[:2]))
    torch.testing.assert_close(fully_sampled_test.abs().amax((-1, -2, -3)), torch.ones(undersampled_test.shape[:2]))

@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_ssduSets(temp_h5_directories, set_name):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=True,
            R=4
            ) 
    data_module.setup(set_name)

    if set_name == "train":
        batch = next(iter(data_module.train_dataloader()))
    elif set_name == "val":
        batch = next(iter(data_module.val_dataloader()))
    else:
        batch = next(iter(data_module.test_dataloader()))

    if set_name == 'test' or set_name == 'val':
        batch = batch[0]
        
    undersampled = batch['undersampled']
    mask = batch['mask']
    loss_mask = batch['loss_mask']
    initial_mask = (undersampled != 0).to(torch.float32)
    fully_sampled = batch['fs_k_space']

    torch.testing.assert_close(loss_mask + mask, initial_mask)
    torch.testing.assert_close(undersampled, (loss_mask + mask) * fully_sampled)
    assert (loss_mask * mask == 0).all()

@pytest.mark.parametrize("set_name", ["val", "test"])
def test_ssduDetermenistic(temp_h5_directories, set_name):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=True,
            R=4
            ) 
    data_module.setup(set_name)

    if set_name == "val":
        batch = next(iter(data_module.val_dataloader()))
        batch2 = next(iter(data_module.val_dataloader()))
    else:
        batch = next(iter(data_module.test_dataloader()))
        batch2 = next(iter(data_module.test_dataloader()))
    
    if set_name == 'test' or set_name == 'val':
        batch = batch[0]
        batch2 = batch2[0]

    undersampled = batch['undersampled']
    undersampled2 = batch2['undersampled']
    mask = batch['mask']
    mask2 = batch2['mask']
    loss_mask = batch['loss_mask']
    loss_mask2 = batch2['loss_mask']

    torch.testing.assert_close(undersampled, undersampled2)
    torch.testing.assert_close(mask2 + loss_mask2, mask + loss_mask)

@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_ssduNonDetermenistic(temp_h5_directories, set_name):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=True,
            R=4,
            ssdu_partitioning=True
            ) 
    data_module.setup(set_name)

    if set_name == "train":
        dataset = data_module.train_dataset
    elif set_name == "val":
        dataset = data_module.val_dataset
    else:
        dataset = data_module.test_dataset

    if isinstance(dataset, TestDataset):
        dataset.undersampled_dataset.set_epoch(0)
    else:
        dataset.set_epoch(0)
    batch = dataset[0]
    if isinstance(dataset, TestDataset):
        dataset.undersampled_dataset.set_epoch(1)
    else:
        dataset.set_epoch(1)
    batch2 = dataset[0]
    if set_name == 'test' or set_name == 'val':
        batch = batch[0]# type: ignore
        batch2 = batch2[0]# type: ignore
        
    undersampled = batch['undersampled'] # type: ignore
    undersampled2 = batch2['undersampled']# type: ignore
    mask = batch['mask']# type: ignore
    mask2 = batch2['mask']# type: ignore
    loss_mask = batch['loss_mask']# type: ignore
    loss_mask2 = batch2['loss_mask']# type: ignore

    torch.testing.assert_close(undersampled, undersampled2)
    assert (mask2 != mask).any()
    assert (loss_mask != loss_mask2).any()
    torch.testing.assert_close(mask2 + loss_mask2, mask + loss_mask)

@pytest.mark.parametrize("set_name", ["train", "val", "test"])
@pytest.mark.parametrize("is_self_supervised", [True, False])
def test_mask_type(temp_h5_directories, set_name, is_self_supervised):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=is_self_supervised,
            R=4
            ) 
    data_module.setup(set_name)

    if set_name == "train":
        dataset = data_module.train_dataloader()
    elif set_name == "val":
        dataset = data_module.val_dataloader()
    else:
        dataset = data_module.test_dataloader()

    batch = next(iter(dataset))
    if set_name == 'test' or set_name == 'val':
        batch = batch[0]
    undersampled = batch['undersampled']
    mask = batch['mask']
    loss_mask = batch['loss_mask']
    fs_k_space = batch['fs_k_space']

    assert undersampled.dtype == torch.complex64
    assert fs_k_space.dtype == torch.complex64
    assert mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32


@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_ssl_non_determenistic(temp_h5_directories, set_name):
    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=True,
            R=4
            ) 
    data_module.setup(set_name)

    if set_name == "train":
        dataset = data_module.train_dataset
    elif set_name == "val":
        dataset = data_module.val_dataset
    else:
        dataset = data_module.test_dataset
    

    if isinstance(dataset, TestDataset):
        dataset.undersampled_dataset.set_epoch(0)
    else:
        dataset.set_epoch(0)
    batch = dataset[0]
    if isinstance(dataset, TestDataset):
        dataset.undersampled_dataset.set_epoch(1)
    else:
        dataset.set_epoch(1)
    batch2 = dataset[0]

    if set_name == 'test' or set_name == 'val':
        batch = batch[0]# type: ignore
        batch2 = batch2[0]# type: ignore
        
    undersampled = batch['undersampled'] # type: ignore
    undersampled2 = batch2['undersampled']# type: ignore
    mask = batch['mask']# type: ignore
    mask2 = batch2['mask']# type: ignore
    loss_mask = batch['loss_mask']# type: ignore
    loss_mask2 = batch2['loss_mask']# type: ignore

    torch.testing.assert_close(undersampled, undersampled2)
    assert (mask2 != mask).any()
    assert (loss_mask != loss_mask2).any()
    torch.testing.assert_close(mask2 + loss_mask2, mask + loss_mask)

@pytest.mark.parametrize("set_name", ["train", "val", "test"]) # idk how to get the same slice from the train dataloader other than turning off shuffle
def test_ssl_non_determenistic_dataloaders(temp_h5_directories, set_name):
    """
    Make sure that the dataloaders generates a new mask every epoch
    """

    data_module = UndersampledDataModule(
            'brats', 
            temp_h5_directories, 
            temp_h5_directories, 
            batch_size=BATCH_SIZE, 
            resolution=DATA_SIZE[3:],
            num_workers=2,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            self_supervised=True,
            R=4
            ) 
    data_module.setup(set_name)
    
    if set_name == "train":
        dataloader = data_module.train_dataloader()
    elif set_name == "val":
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    if set_name == 'test' or set_name == 'val':
        dataloader.dataset.undersampled_dataset.set_epoch(0)  # type: ignore
        batch = next(iter(dataloader))[0]
        dataloader.dataset.undersampled_dataset.set_epoch(1)  # type: ignore
        batch2 = next(iter(dataloader))[0]
    else:
        # set the seed so shuffle gets the same batch every time
        torch.manual_seed(0)
        # set the epoch to the first epoch
        dataloader.dataset.set_epoch(0)  # type: ignore
        batch = next(iter(dataloader))

        # reset the seed so same batch is generated
        torch.manual_seed(0)
        # increment epoch
        dataloader.dataset.set_epoch(1)  # type: ignore
        batch2 = next(iter(dataloader))        

    undersampled = batch['undersampled'] 
    undersampled2 = batch2['undersampled']
    mask = batch['mask']
    mask2 = batch2['mask']
    loss_mask = batch['loss_mask']
    loss_mask2 = batch2['loss_mask']

    torch.testing.assert_close(undersampled, undersampled2)
    assert (mask2 != mask).any()
    assert (loss_mask != loss_mask2).any()
    torch.testing.assert_close(mask2 + loss_mask2, mask + loss_mask)

def test_same_partitioning_mask(temp_h5_directories):
    """
    Make sure that the dataloaders generates a new mask every epoch
    """

    data_module = UndersampledDataModule(
        'brats', 
        temp_h5_directories, 
        temp_h5_directories, 
        batch_size=BATCH_SIZE, 
        resolution=DATA_SIZE[3:],
        num_workers=2,
        contrasts=['t1', 't2', 't1ce', 'flair'],
        self_supervised=True,
        R=4,
        same_mask_every_epoch=True
    ) 

    data_module.setup('train')
    dataloader = data_module.train_dataloader()
    
    torch.manual_seed(0)
    dataloader.dataset.set_epoch(0)  # type: ignore
    batch = next(iter(dataloader))
    torch.manual_seed(0)
    dataloader.dataset.set_epoch(1)  # type: ignore
    batch2 = next(iter(dataloader))
        
    undersampled = batch['undersampled'] # type: ignore
    undersampled2 = batch2['undersampled']# type: ignore
    mask = batch['mask']# type: ignore
    mask2 = batch2['mask']# type: ignore
    loss_mask = batch['loss_mask']# type: ignore
    loss_mask2 = batch2['loss_mask']# type: ignore

    torch.testing.assert_close(undersampled, undersampled2)
    torch.testing.assert_close(mask, mask2)
    torch.testing.assert_close(loss_mask, loss_mask2)
    torch.testing.assert_close(mask2 + loss_mask2, mask + loss_mask)
