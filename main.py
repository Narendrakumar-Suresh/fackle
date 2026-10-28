import jax
import jax.numpy as jnp
import optax

from fackle import Module
from fackle.nn import Conv2D


class ConvNet(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Conv layers accepting NHWC input format
        self.conv1 = Conv2D(in_channels, hidden_channels, kernel_size=3, padding="SAME")
        self.conv2 = Conv2D(
            hidden_channels, out_channels, kernel_size=3, padding="SAME"
        )

    def __call__(self, x: jax.Array):
        x = self.conv1(x)
        x = jnp.tanh(x)
        return self.conv2(x)


# ----------------------------
# Loss
# ----------------------------
def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


# ----------------------------
# Training
# ----------------------------
def main():
    # Input shape: (Batch=1, Height=5, Width=5, Channels=3)
    model = ConvNet(in_channels=3, hidden_channels=8, out_channels=3)

    optimizer = optax.sgd(learning_rate=0.01)

    x = jnp.ones((1, 5, 5, 3))
    y = jnp.zeros((1, 5, 5, 3))
    model = model.compile(tx=optimizer, loss_fn=mse)

    for step in range(100):
        loss = model.train_step(x, y)

        if step % 10 == 0:
            print(f"step={step}, loss={loss:.4f}")


if __name__ == "__main__":
    main()
