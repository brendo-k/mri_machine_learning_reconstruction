import torch
from datetime import datetime

def train(model, loss_function, optimizer, dataloader, device, epoch=1):
    cur_loss = 0
    current_index = 0
    try:
        for e in range(epoch):
            for data in dataloader:

                sampled = data['sampled']
                mask = data['mask']
                undersampled = data['undersampled']
                for i in range(sampled.shape[0]):
                    optimizer.zero_grad()
                    sampled_slice = sampled[[i],...]
                    mask_slice = mask[[i],...]
                    undersampled_slice = undersampled[[i],...]
                    mask_slice = mask_slice.to(device)
                    undersampled_slice = undersampled_slice.to(device)

                    predicted_sampled = model(undersampled_slice, mask_slice)
                    loss = loss_function(torch.view_as_real(predicted_sampled), torch.view_as_real(sampled_slice))

                    loss.backward()
                    optimizer.step()
                    cur_loss += loss.item()
                    if current_index % 10 == 9:
                        print(f"Iteration: {current_index + 1:>d}, Loss: {cur_loss:>7f}")
                        cur_loss = 0
                    current_index += 1
    except KeyboardInterrupt:
        pass

    model_name = model.__class__.__name__
    date = str(datetime.now()).replace(' ', ',')
    torch.save(model.state_dict(), './Model_Weights/' + date + model_name)