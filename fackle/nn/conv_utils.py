import math
import jax
import jax.numpy as jnp
from fackle import Module
from typing import Union, Sequence

IntOrTuple = Union[int, Sequence[int]]


def _ntuple(value: IntOrTuple, n: int) -> tuple:
    if isinstance(value, int):
        return (value,) * n
    value = tuple(value)
    if len(value) != n:
        raise ValueError(f"Expected {n} values, got {len(value)}")
    return value


class Flatten(Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x.reshape((x.shape[0], -1))


"""1D Pooling"""


class MaxPool1D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 1)
        self.stride = _ntuple(stride, 1)
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )


class AvgPool1D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = None, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 1)
        self.stride = _ntuple(stride, 1) if stride is not None else self.window
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        summed = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )
        return summed / math.prod(*self.window)


class MinPool1D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 1)
        self.stride = _ntuple(stride, 1)
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=jnp.inf,
            computation=jax.lax.min,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )


"""2D Pooling"""


class MaxPool2D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 2)
        self.stride = _ntuple(stride, 2)
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )


class AvgPool2D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = None, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 2)
        self.stride = _ntuple(stride, 2) if stride is not None else self.window
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        summed = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )
        return summed / math.prod(*self.window)


class MinPool2D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 2)
        self.stride = _ntuple(stride, 2)
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=jnp.inf,
            computation=jax.lax.min,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )


"""3D Pooling"""


class MaxPool3D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 3)
        self.stride = _ntuple(stride, 3) if stride is not None else self.window
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )


class AvgPool3D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = None, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 3)
        self.stride = _ntuple(stride, 3) if stride is not None else self.window
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        summed = jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )
        return summed / math.prod(*self.window)


class MinPool3D(Module):
    def __init__(
        self, window: IntOrTuple = 2, stride: IntOrTuple = 1, padding: str = "VALID"
    ):
        self.window = _ntuple(window, 3)
        self.stride = _ntuple(stride, 3)
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            init_value=jnp.inf,
            computation=jax.lax.min,
            window_dimensions=(1, *self.window, 1),
            window_strides=(1, *self.stride, 1),
            padding=self.padding,
        )
