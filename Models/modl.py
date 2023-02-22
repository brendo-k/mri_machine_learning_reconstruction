import torch.nn as nn
import torch
import torch.fft as fft
from .Modlparts import parts

class modl(nn.Module):
    def __init__(self, num_models, model_depth) -> None:
        super().__init__()
        self.lambda_regularizer = torch.rand(1, requires_grad=True)
        self.model = nn.ModuleList()
        for _ in range(num_models):
            self.model.append(parts(model_depth))

    def forward(self, x: torch.Tensor, mask, itterations):
        b = x
        for i in range(itterations):
            x = modl._view_as_real(x)
            z_n = self.model[i%len(self.model)](x)
            x = modl._CG(mask, b, modl._view_as_complex(z_n), self.lambda_regularizer)

        return x
    
    @staticmethod
    def _view_as_real(x):
        x = torch.view_as_real(x)
        return x.permute((0, 3, 1, 2))

    @staticmethod
    def _view_as_complex(x):
        x = x.permute((0, 2, 3, 1))
        return torch.view_as_complex(x)

    def _CG(mask, b, z_n, lamreg, cg_iters=5, cg_tol=1E-9):

        # this is b used to solve for x in the paper (A^H * b + lambda * z_n)
        b = b + lamreg*z_n
        # start guess at zero
        z = z_n*0
        # Residual is equal to b? 
        r = b
        # search direction is residual from CG
        p = r
        # what is this?
        rsold = torch.dot(torch.conj(torch.reshape(r,(-1,))), torch.reshape(r,(-1,))).real

        for i in range(cg_iters):
            Ap = modl.A_matrix_function(p, mask, lamreg)
            alpha = rsold/torch.dot(torch.conj(torch.reshape(p,(-1,))), torch.reshape(Ap,(-1,)))
            z = z + alpha*p
            r = r - alpha*Ap
            rsnew = torch.dot(torch.conj(torch.reshape(r,(-1,))), torch.reshape(r,(-1,))).real
            
            # Pytorch dynamic graph allows conditional loops in model
            if torch.sqrt(rsnew) < cg_tol:
                break

            p = r + (rsnew/rsold)*p
            rsold = rsnew

        return z

    # Using FFT for A matrix instead of explicitly defining (very large and will be slow if explicitly defined!)
    @staticmethod
    def A_matrix_function(z_n, mask, lamreg):
        return fft.ifft2(torch.mul(mask, fft.fft2(z_n)), axis=(-2, -1)) + lamreg*z_n

