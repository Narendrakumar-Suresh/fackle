from fackle.core import Module


class Sequential(Module):
    def __init__(self, *layers):
        self.num_layers = len(layers)

        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)

    def __call__(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"layer_{i}")(x)
        return x
