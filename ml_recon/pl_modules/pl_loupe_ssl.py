# lightning_module.py
import pytorch_lightning as pl
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from ml_recon.losses import L1L2Loss
from ml_recon.utils.evaluate import nmse
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts
from ml_recon.models.loupe_ssl import SSLModel
import torch

class SSLLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training the SSL model.
    Handles training, validation, and testing logic.
    """
    def __init__(
            self,
            model: SSLModel,
            lr: float = 1e-2,
            ssim_scaling_set: float = 1e-4,
            ssim_scaling_full: float = 1e-4,
            ssim_scaling_inverse: float = 1e-4,
            lambda_scaling: float = 0.0,
            pass_all_data: bool = False,
            pass_inverse_data: bool = False,
            supervised: bool = False
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.loss_func = L1L2Loss
        
    def training_step(self, batch, batch_idx):
        if self.hparams.supervised:
            return self._train_supervised_step(batch)
        return self._train_ssl_step(batch, batch_idx)
    
    def _train_ssl_step(self, batch, batch_idx):
        undersampled = batch['input']
        initial_mask = (undersampled != 0).to(torch.float32)
        nbatch = undersampled.shape[0]
        
        # Split data into lambda and inverse sets
        lambda_set, inverse_set = self._split_into_sets(initial_mask, nbatch)
        estimate_lambda = self.model(undersampled, lambda_set)
        
        # Calculate primary loss
        loss = self._calculate_lambda_loss(undersampled, estimate_lambda, inverse_set)
        
        # Calculate additional losses if needed
        if self.hparams.pass_inverse_data:
            loss += self._calculate_inverse_loss(batch, lambda_set, inverse_set)
            
        if self.hparams.pass_all_data:
            loss += self._calculate_full_loss(batch, lambda_set)
            
        # Log metrics
        self._log_training_metrics(loss, batch_idx, estimate_lambda, lambda_set, inverse_set)
        
        return loss
    
    def _train_supervised_step(self, batch):
        undersampled = batch['input']
        mask = (undersampled != 0).to(torch.float32)
        fully_sampled = batch['fs_k_space']
        
        estimate = self.model(undersampled, mask)
        loss = self.loss_func(torch.view_as_real(fully_sampled), torch.view_as_real(estimate))
        
        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        under = batch['input']
        fs_k_space = batch['fs_k_space']
        initial_mask = (under != 0).to(torch.float32)
        
        # Generate estimates
        estimates = self._generate_validation_estimates(under, initial_mask)
        
        # Calculate and log metrics
        self._log_validation_metrics(batch, estimates, batch_idx)
        
        if batch_idx == 0:
            self._log_validation_images(batch, estimates)
    
    def test_step(self, batch, batch_idx):
        undersampled = batch['input']
        k_space = batch['fs_k_space']
        initial_mask = (undersampled != 0).to(torch.float32)
        
        # Generate different estimates
        estimates = self._generate_test_estimates(undersampled, initial_mask)
        
        # Calculate and log metrics for each estimate
        self._log_test_metrics(batch, estimates, k_space)
        
        if batch_idx == 0:
            self._log_test_images(estimates)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def _split_into_sets(self, omega_mask: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the data into lambda and inverse sets."""
        lambda_mask = self.model.get_mask(batch_size, mask_center=True)
        return omega_mask * lambda_mask, omega_mask * (1 - lambda_mask)