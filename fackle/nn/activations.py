import jax.nn as jnn
import jax.numpy as jnp

from fackle.core import Module


# ----------------------------
# Activation Modules
# ----------------------------
class ReLU(Module):
    def __call__(self, x):
        return jnn.relu(x)


class GELU(Module):
    def __call__(self, x):
        return jnn.gelu(x)


class Sigmoid(Module):
    def __call__(self, x):
        return jnn.sigmoid(x)


class Tanh(Module):
    def __call__(self, x):
        return jnp.tanh(x)


class SiLU(Module):
    def __call__(self, x):
        return jnn.silu(x)


class Softmax(Module):
    def __call__(self, x, axis=-1):
        return jnn.softmax(x, axis=axis)


# ----------------------------
# Functional aliases
# ----------------------------
relu = jnn.relu
gelu = jnn.gelu
sigmoid = jnn.sigmoid
tanh = jnp.tanh
silu = jnn.silu
softmax = jnn.softmax
