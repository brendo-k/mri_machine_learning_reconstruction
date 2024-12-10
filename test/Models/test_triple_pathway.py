from ml_recon.models.TriplePathway import TriplePathway, DualDomainConifg, VarnetConfig
import pytest
import torch

@pytest.fixture()
def triple_pathway_model():
    model = TriplePathway(
        DualDomainConifg(is_pass_inverse=True, is_pass_original=True), 
        VarnetConfig(['t1'], cascades=1, channels=4, sense_chans=4)
    )
    return model

@pytest.fixture()
def model_inputs():
    batch_size = (2, 1, 4, 128, 128)
    k_space = torch.randn(batch_size, dtype=torch.complex64)
    initial_mask = torch.rand(batch_size) > 0.5 
    initial_mask[..., batch_size[-1]//2 - 5:batch_size[-1]//2 - 5, batch_size[-2]//2 - 5:batch_size[-2]//2 - 5] = 1 
    second_mask = torch.rand(batch_size) > 0.5 
    second_mask[..., batch_size[-1]//2 - 5:batch_size[-1]//2 - 5, batch_size[-2]//2 - 5:batch_size[-2]//2 - 5] = 1 
    initial_mask = initial_mask.to(torch.float32)
    second_mask = second_mask.to(torch.float32)

    return k_space, initial_mask, second_mask


def test_pass_lambda_only(triple_pathway_model: TriplePathway, model_inputs):
    triple_pathway_model.config.is_pass_inverse = False
    triple_pathway_model.config.is_pass_original = False

    k_space, initial_mask, second_mask = model_inputs
    
    undersampled_k = k_space * initial_mask

    outputs = triple_pathway_model.forward(
        undersampled_k=undersampled_k, 
        fully_sampled_k=k_space, 
        input_set=initial_mask, 
        target_set=torch.ones_like(undersampled_k)
    )

    assert outputs['inverse_path'] is None
    assert outputs['full_path'] is None

    lambda_prediction = outputs['lambda_path']
    # make sure dc step is correct
    torch.testing.assert_close(lambda_prediction * initial_mask, undersampled_k)
