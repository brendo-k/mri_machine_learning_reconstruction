from torchvision.transforms import Compose
import os
import time
from collections import defaultdict
import psutil
import torch
from tabulate import tabulate

from ml_recon.transforms import normalize
from ml_recon.dataset.sliceloader import SliceDataset

def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res

class MemoryMonitor():
  def __init__(self, pids: list[int] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int):
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self):
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def table(self) -> str:
    self._refresh()
    table = []
    keys = list(list(self.data.values())[0].keys())
    now = str(int(time.perf_counter() % 1e5))
    for pid, data in self.data.items():
      table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
    return tabulate(table, headers=["time", "PID"] + keys)

  def str(self):
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)

  @staticmethod
  def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)


def worker(_, dataset: torch.utils.data.Dataset):
    while True:
        for sample in dataset:
            x = sample[0]

import sys
if __name__ == "__main__":
    transforms = Compose(
        (
            normalize(),
        )
    )
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/' 
    print(data_dir)

    monitor = MemoryMonitor() 
    train_dataset = SliceDataset(os.path.join(data_dir, 'multicoil_train'),
                                 transforms=transforms,
                                 )
    
    print(monitor.table())
    val_dataset = SliceDataset(os.path.join(data_dir, 'multicoil_val'),
                                 transforms=transforms
                                 )


    

    start_method = 'fork'
    ctx = torch.multiprocessing.start_processes(
          worker, (train_dataset, ), nprocs=4, join=False,
          daemon=True, start_method=start_method)

    [monitor.add_pid(pid) for pid in ctx.pids()]

    try: 
        for i in range(30):
            print(monitor.table())
            time.sleep(1)
    finally:
        ctx.join()



