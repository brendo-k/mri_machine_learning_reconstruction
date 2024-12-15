import pytest
import torch
from ml_recon.models import XNet

@pytest.fixture
def model(device):
    model = XNet(['t1', 't2', 'flair'], channels=4, depth=3, drop_prob=0)
    model.to(device)
    return model

def test_forward_shape(model, image_input_3_contrast, device):
    image_input_3_contrast = image_input_3_contrast.to(device)
    output = model(image_input_3_contrast)

    assert output.shape == image_input_3_contrast.shape

def test_loss(model, image_input_3_contrast, device):
    image_input_3_contrast = image_input_3_contrast.to(device)
    target = torch.randn_like(image_input_3_contrast)
    output = model(image_input_3_contrast)
    optimizer = torch.optim.SGD(model.parameters())

    loss = torch.nn.L1Loss()(target, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output = model(image_input_3_contrast)
    loss2 = torch.nn.L1Loss()(target, output)
    
    assert loss2 < loss 
    