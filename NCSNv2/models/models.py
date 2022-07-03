def get_model(model_name):
    """
    Get PyTorch model.
    :param model_name: The name of the model.
    :return: The model (nn.Module).
    """
    if model_name == 'NCSNv2':
        from .ncsnv2 import NCSNv2
        model = NCSNv2(128)
    elif model_name == 'ToyModel':
        from .toy_model import ToyModel
        model = ToyModel()
    else:
        raise ValueError('Unknown model_name:', model_name)
    return model
