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

    # Compute the initial logits
    with chainer.using_config('train', False):
        logits = forward(x)
    
    # Refine the perturbation using gradient ascent
    for _ in range(num_iterations):
        # Add small perturbation
        x_perturbed = x + xi * Variable(d.astype(np.float32))
        
        # Compute logits for the perturbed input
        logits_perturbed = forward(x_perturbed)
        
        # Calculate KL divergence between original and perturbed logits
        kl_div = F.sum(F.kl_div(F.log_softmax(logits_perturbed), F.softmax(logits), axis=1))
        
        # Compute gradients of KL divergence with respect to perturbation
        x.cleargrads()
        kl_div.backward()
        
        # Update the perturbation in the direction of the gradient
        d = x.grad
        d /= xp.linalg.norm(d, axis=(1, 2, 3), keepdims=True) + 1e-12

    # Scale the perturbation to have a norm of epsilon
    perturbation = epsilon * d
    return Variable(perturbation)
