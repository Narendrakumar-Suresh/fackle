from fackle.core import Module, Parameter, Buffer
import jax
import jax.numpy as jnp
from typing import Union, Tuple, Sequence, Optional
from fackle import random


class Linear(Module):
    def __init__(self, din, dout, key=None):
        self.din = din
        self.dout = dout

        k = key or random.next_key()
        scale = jnp.sqrt(2.0 / din)  # He init

        self.kernel = Parameter(jax.random.normal(k, (din, dout)) * scale)
        self.bias = Parameter(jnp.zeros((dout,)))
        self.call_count = Buffer(jnp.array(0))

    def __call__(self, x: jax.Array):
        # State update happens inside the module itself.
        if self.call_count.value is not None:
            self.call_count = Buffer(self.call_count.value + 1)

        return x @ self.kernel.value + self.bias.value


class Conv1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        dilation: Union[int, Tuple[int]] = 1,
        use_bias: bool = True,
        key: Optional[jax.Array] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.stride = (stride,) if isinstance(stride, int) else tuple(stride)
        self.dilation = (dilation,) if isinstance(dilation, int) else tuple(dilation)
        self.padding = padding.upper() if isinstance(padding, str) else padding

        k_shape = self.kernel_size + (in_channels, out_channels)
        scale = jnp.sqrt(2.0 / (in_channels * self.kernel_size[0]))

        k = key or random.next_key()

        # Wrapped in Parameter so Fackel tracks them for gradients & optimization
        self.weight = Parameter(jax.random.normal(k, k_shape) * scale)
        self.bias = Parameter(jnp.zeros((1, 1, out_channels))) if use_bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        dimension_numbers = ("NWC", "WIO", "NWC")

        out = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight.value,  # Access underlying array value
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=dimension_numbers,
        )
        if self.bias is not None:
            out = out + self.bias.value
        return out


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        dilation: Union[int, Tuple[int, int]] = 1,
        use_bias: bool = True,
        key: Optional[jax.Array] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.dilation = (
            (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        )
        self.padding = padding.upper() if isinstance(padding, str) else padding

        k_shape = self.kernel_size + (in_channels, out_channels)
        scale = jnp.sqrt(
            2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        )

        k = key or random.next_key()

        self.weight = Parameter(jax.random.normal(k, k_shape) * scale)
        self.bias = Parameter(jnp.zeros((1, 1, 1, out_channels))) if use_bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        dimension_numbers = ("NHWC", "HWIO", "NHWC")

        out = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight.value,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=dimension_numbers,
        )
        if self.bias is not None:
            out = out + self.bias.value
        return out


class Conv3D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        dilation: Union[int, Tuple[int, int, int]] = 1,
        use_bias: bool = True,
        key: Optional[jax.Array] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (
            (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        )
        self.dilation = (
            (dilation, dilation, dilation)
            if isinstance(dilation, int)
            else tuple(dilation)
        )
        self.padding = padding.upper() if isinstance(padding, str) else padding

        k_shape = self.kernel_size + (in_channels, out_channels)
        scale = jnp.sqrt(2.0 / (in_channels * jnp.prod(jnp.array(self.kernel_size))))

        k = key or random.next_key()

        self.weight = Parameter(jax.random.normal(k, k_shape) * scale)
        self.bias = (
            Parameter(jnp.zeros((1, 1, 1, 1, out_channels))) if use_bias else None
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

        out = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight.value,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=dimension_numbers,
        )
        if self.bias is not None:
            out = out + self.bias.value
        return out
