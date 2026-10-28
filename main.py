import jax
import jax.numpy as jnp
import optax

from fackle.core import Module
from fackle.nn import Linear


class MLP(Module):
    def __init__(self, din, hidden, dout):
        self.layer1 = Linear(din, hidden)
        self.layer2 = Linear(hidden, dout)

    def __call__(self, x: jax.Array):
        x = self.layer1(x)
        x = jnp.tanh(x)
        return self.layer2(x)


# ----------------------------
# Loss
# ----------------------------
def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


# ----------------------------
# Training
# ----------------------------
def main():
    model = MLP(din=2, hidden=4, dout=3)

    optimizer = optax.sgd(learning_rate=0.01)
    loss_fn = mse

    x = jnp.array([[1.0, 2.0]])
    y = jnp.array([[0.5, 0.5, 0.5]])

    for step in range(100):
        loss = model.train_step(x, y, optimizer, loss_fn)

        if step % 10 == 0:
            print(
                f"step={step}, "
                f"loss={loss:.4f}, "
                f"layer1_call_count={model.layer1.call_count.value}, "
                f"layer2_call_count={model.layer2.call_count.value}"
            )


if __name__ == "__main__":
    main()
