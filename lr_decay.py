class LRDecay:
    def __init__(self, optimizers: list, hidden_dim: int, base_learning_rate: float, warmup: int, initial_step=0):
        self.optimizers = optimizers
        self.dim = hidden_dim
        self.base_learning_rate = base_learning_rate
        self.warmup = warmup
        self.step = initial_step

    def __call__(self) -> float:
        self.step += 1
        if self.step < self.warmup:
            learning_rate = self.dim ** (-0.5) * self.step * self.warmup ** (-1.5) * self.base_learning_rate
        else:
            learning_rate = 0.5*self.dim ** (-0.5) * self.step ** (-0.5) * self.base_learning_rate

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        return learning_rate