import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable

def generate_vat_perturbation(forward, x, xi=1e-6, epsilon=3.5, num_iterations=1):
    """
    Generate adversarial perturbation for Virtual Adversarial Training (VAT).

    Parameters:
        forward (callable): The model function to generate predictions.
        x (Variable): The input data.
        xi (float): Small constant to initialize the perturbation.
        epsilon (float): Scaling factor for the final perturbation.
        num_iterations (int): Number of iterations for perturbation refinement.

    Returns:
        Variable: The generated perturbation.
    """
    # Create a random unit vector as initial perturbation
    xp = chainer.backends.cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape).astype(np.float32)
    d /= xp.linalg.norm(d, axis=(1, 2, 3), keepdims=True) + 1e-12
    with chainer.using_config('train', False):
        logits = forward(x)
    
    for _ in range(num_iterations):
        x_perturbed = x + xi * Variable(d.astype(np.float32))
        logits_perturbed = forward(x_perturbed)
        kl_div = F.sum(F.kl_div(F.log_softmax(logits_perturbed), F.softmax(logits), axis=1))
        x.cleargrads()
        kl_div.backward()
        d = x.grad
        d /= xp.linalg.norm(d, axis=(1, 2, 3), keepdims=True) + 1e-12
    perturbation = epsilon * d
    return Variable(perturbation)


def consistency_regularization(forward, x, perturbation):
    """
    Compute the consistency regularization term for VAT.

    Parameters:
        forward (callable): The model function to generate predictions.
        x (Variable): The original input data.
        perturbation (Variable): The generated adversarial perturbation.

    Returns:
        Variable: The consistency regularization term (KL divergence).
    """
    # Compute logits for original input
    with chainer.using_config('train', False):
        logits_original = forward(x)

    # Compute logits for perturbed input
    x_perturbed = x + perturbation
    logits_perturbed = forward(x_perturbed)

    # Compute KL divergence between the two sets of logits
    kl_div = F.sum(F.kl_div(F.log_softmax(logits_perturbed), F.softmax(logits_original), axis=1))
    return kl_div

