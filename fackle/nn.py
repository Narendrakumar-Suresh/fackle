from fackle.core import Module, Parameter, Buffer
import jax.numpy as jnp
import jax

class Linear(Module):
    def __init__(self, din, dout):
        self.din = din
        self.dout = dout

        self.kernel = Parameter(jnp.ones((din, dout)))
        self.bias = Parameter(jnp.zeros((dout,)))

        self.call_count = Buffer(jnp.array(0))

    def __call__(self, x: jax.Array):
        # State update happens inside the module itself.
        if self.call_count.value is not None:
            self.call_count = Buffer(self.call_count.value + 1)

        return x @ self.kernel.value + self.bias.value
`