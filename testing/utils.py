import torch

def catch_gradcheck(message, *args):
    check = False
    try:
        check = torch.autograd.gradcheck(*args)
    except RuntimeError as e:
        msg = f"{str(e)} {message}"
    assert check, msg
