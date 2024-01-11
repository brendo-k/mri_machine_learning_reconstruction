def convert_weights_from_distributed(checkpoint):
    """
    Converst pytorch checkpoint saved from ddp (distributed) model training and 
    converts to single gpu checkpoint
    """
    
    new_dict = {}
    for key, values in checkpoint['model'].items():
        new_key = '.'.join(key.split('.')[1:]) 
        new_dict[new_key] = values
    return new_dict
