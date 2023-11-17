import torch
import rff


def rff_layer(x, Hyperparams):
    """
    This function applies a Gaussian Encoding to the input tensor and then concatenates the first two columns of the encoded tensor.

    Parameters:
    x (torch.Tensor): The input tensor to be encoded and processed.
    Hyperparams (object): An object that contains the hyperparameters for the Gaussian Encoding. It should have an attribute 'encoded_size' which specifies the size of the encoded tensor.

    Returns:
    torch.Tensor: The processed tensor. The first two columns of the encoded input tensor are concatenated and the resulting tensor is converted to double precision.

    Example:
    >>> Hyperparams = type('', (), {})()
    >>> Hyperparams.encoded_size = 100
    >>> x = torch.randn(10, 10)
    >>> output = rff_layer(x, Hyperparams)
    """

    encoding = rff.layers.GaussianEncoding(sigma=1.0, input_size=1, encoded_size=Hyperparams.batch_size_pos)
    # make x a float32 tensor
    x = x.float()
    x = x.unsqueeze(1)
    x = encoding(x)
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)), dim=0)
    x = x.squeeze(1)
    # make x a float64 tensor
    x = x.double()
    return x