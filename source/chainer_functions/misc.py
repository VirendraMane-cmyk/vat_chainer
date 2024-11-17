import chainer.functions as F

def call_bn(bn, x, test=False, update_batch_stats=True):
    if test:
        # For inference, use fixed batch normalization
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var)
    elif not update_batch_stats:
        # For training without updating statistics, use batch normalization directly
        return bn(x)  # Automatically handles the internal state
    else:
        # Normal training, updating statistics
        return bn(x)  # Automatically handles the internal state

