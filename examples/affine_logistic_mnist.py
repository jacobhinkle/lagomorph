"""
This uses MNIST to test a logistic regression model having a latent affine
transform applied to the inputs.
"""
raise Exception("DEPRECATED: This example has not been updated to use the new pytorch interface")

import torch
from torch import nn
import h5py
import numpy as np
import lagomorph.torch as lt
import math
from tqdm import tqdm
import atexit

class MCGEMSGDOptimizer(torch.optim.Optimizer):
    def __init__(self):
        super(MCGEMSGDOptimizer, self).__init__()
        # this gets filled out as we go
        self.expected_grad = {}
        self.sum_weights = 0.0
    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for (k,v) in self.expected_grad.items():
            v.detach_()
            v.zero_()
        self.sum_weights = 0.0
    def accumulate_gradients(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p in self.expected_grad:
                        self.expected_grad[p] += p.grad
                    else:
                        self.expected_grad[p] = p.grad
    def expectation_step(self, sample_losses, latent_params, num_samples):
        r"""This samples the posterior for `affine_params` and compute the
        gradients with respect to the optimized parameters for each sample,
        summing those gradients.

        Provide the tensor for sample losses (unsummed). This will be used to
        convert to a weighted sum, where the weights are the likelihoods of each
        sample's pose.
        """
        for n in range(num_samples):
            # sample new affine parameters
            #A, T = sample...
            # get weight by interpreting loss as log-likelihood
            sample_losses
            weights = torch.exp(-lossval)
            loss.backward()
            # now accumulate these gradients
            self.accumulate_gradients(weights)
            # propose new points to sample via a Langevin type of adjustment
    def step(self, closure):
        r"""Performs a single optimization step (parameter update).
        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss.
        """
        # this is the SGD step method
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                #d_p = p.grad.data
                d_p = self.expected_grad[p].data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

class AffineLogistic(nn.Module):
    """Just a really simple model that has some affine parameters that we can
    fiddle with
    """
    def __init__(self, image_shape, num_classes):
        super(AffineLogistic, self).__init__()
        self.image_shape = image_shape
        self.num_features = prod(image_shape)
        self.features = lt.AffineInterpImage()
        self.classifier = nn.Sequential(
                nn.Linear(self.num_features, num_classes)
                )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), prod(self.image_shape))
        return self.classifier(x)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    device = torch.device("cuda" if use_cuda else "cpu")
    sh = (28,28)
    num_classes = 10
    # build the model
    model = AffineLogistic(sh, num_classes).to(device)
    # optimize it
    opt = MCGEMSGDOptimizer()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
