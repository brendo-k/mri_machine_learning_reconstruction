import os
import h5py

def move_16_chans(path, output=None):
    # check to make sure it is directory
    if path[-1] != '/':
        path += '/'

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for file in files:
        print(file)
        full_path = os.path.join(path, file)
        try:
            with h5py.File(full_path) as fr:
                # loop through all the slices
                coils = fr['kspace'].shape[1]
                if coils == 16:
                    os.rename(full_path, os.path.join(path, '16_chans', file))
        except OSError as e:
            print(e)
            print(full_path)

    return 1
    
if __name__ == "__main__":
    move_16_chans("/home/kadotab/scratch/fastMRIDataset/multicoil_train/T1/")
