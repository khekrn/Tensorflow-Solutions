from tensorflow.python.training import optimizer


class L4Optimizer(optimizer.Optimizer):

    def __init__(self, fraction=0.15, minloss_factor=0.9, init_factor=0.75,
                 minloss_forget_time=1000.0, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, use_locking=False, name="L4Optimizer"):
        super(L4Optimizer, self).__init__(use_locking, name)
        pass
