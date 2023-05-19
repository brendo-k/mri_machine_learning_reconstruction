import torch
from datetime import datetime
from ml_recon.Utils.save_model import save_model

class train:
    def __init__(self, model, loss_function, optimizer, dataloder, 
                 weights_path='/home/kadotab/python/ml/ml_recon/Model_Weights'):
        self.model = model 
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dataloader = dataloder
        self.device = 'cpu'
        self.weights_path = weights_path
        self.epoch_observer = []
    
    def to(self, device):
        self.device = device

    def attach_epoch(self, observer):
        self.epoch_observer.append(observer)

    def notify_epoch(self):
        for observer in self.epoch_observer:
            observer.notify()

    def fit(self, epoch=1):
        cur_loss = 0
        current_index = 0
        try:
            for e in range(epoch):
                for data in self.dataloader:
                    sampled = data['sampled']
                    mask = data['mask']
                    undersampled = data['undersampled']
                    for i in range(sampled.shape[0]):
                        self.optimizer.zero_grad()
                        sampled_slice = sampled[[i],...]
                        mask_slice = mask[[i],...]
                        undersampled_slice = undersampled[[i],...]
                        mask_slice = mask_slice.to(self.device)
                        undersampled_slice = undersampled_slice.to(self.device)

                        predicted_sampled = self.model(undersampled_slice, mask_slice)
                        loss = self.loss_function(torch.view_as_real(predicted_sampled), torch.view_as_real(sampled_slice))

                        loss.backward()
                        self.optimizer.step()
                        cur_loss += loss.item()
                        current_index += 1
                self.notify_epoch()
                save_model(self.weights_path, self.model, self.optimizer, e)
        except KeyboardInterrupt:
            pass

        save_model(self.weights_path, self.model, self.optimizer, -1)