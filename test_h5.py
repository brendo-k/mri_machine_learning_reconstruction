import h5py
import tempfile
import numpy as np
import time

data = np.random.uniform(-1, 1, (4, 8, 256, 256, 65)) + 1.j * np.random.uniform(-1, 1, (4, 8, 256, 256, 65))
print(data.shape)
with h5py.File('temp.h5', 'w') as fr:
    dst = fr.create_dataset('baseline', data=data)
    dst = fr.create_dataset('chunk', data=data, chunks=(1, 8, 256, 256, 1))

with h5py.File('temp.h5', 'r') as fr:
    random_integers = np.random.randint(0, 65, size=(1000))
    random_integers_first = np.random.randint(0, 4, size=(1000))

    start = time.time()
    for s, c in zip(random_integers, random_integers_first):
        fr['baseline'][c, :, :, :, s]
    print((time.time() - start))

    start = time.time()
    for s, c in zip(random_integers, random_integers_first):
        fr['chunk'][c, :, :, :, s]
    print((time.time() - start))
        

