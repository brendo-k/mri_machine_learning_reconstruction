from ml_recon.models.SensetivityModel import SensetivityModel
import torch

def test_passthrough():
    x = torch.rand((2, 6, 640, 320))
    mask = torch.rand((2, 6, 640, 320))
    sense_model = SensetivityModel(2, 2, 4)
    output = sense_model(x, mask)

    assert output.shape == x.shape

def test_mask():
    x = torch.rand((3, 20, 640, 320))
    mask = torch.zeros(3, 20, 640, 320)
    # test simple case
    mask[0, :, :, 150:170] = 1

    # test non-center lines
    mask[1, :, :, 155:165] = 1
    mask[1, :, :, 20:40] = 1

    # test uneven center lines
    mask[2, :, :, 156:165] = 1
    mask = mask.to(torch.bool)
    
    sense_model = SensetivityModel(2, 2, 4)
    x_masked = sense_model.mask(x, mask)
    
    x_masked_indecies = x_masked != 0
    real_mask = torch.zeros(3, 640, 320)
    real_mask[0, :, 150:170] = 1
    real_mask[1, :, 155:165] = 1
    real_mask[2, :, 156:164] = 1
    real_mask = real_mask.to(torch.bool)

    torch.testing.assert_close(x_masked_indecies[:, 0, :, :], real_mask)

def test_scaling():
    x = torch.rand((2, 20, 640, 320))
    mask = torch.zeros(2, 20, 640, 320)
    mask[:, :, :, 150:170] = 1
    mask = mask.to(torch.bool)

    sense_model = SensetivityModel(2, 2, 4)
    x_masked = sense_model(x, mask)

    x_summed = (x_masked * x_masked.conj()).sum(1)

    torch.testing.assert_allclose(x_summed, torch.ones(x_summed.shape))
