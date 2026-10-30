import jax
import jax.numpy as jnp
import optax

from fackle import Module
from fackle import nn
from fackle.nn import Conv2D, Linear


class ConvNet(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Conv layers accepting NHWC input format
        self.conv1 = Conv2D(in_channels, hidden_channels, kernel_size=3, padding="SAME")
        self.conv2 = Conv2D(hidden_channels, out_channels, kernel_size=3, padding="SAME")
        self.l1 = Linear(out_channels, 3)

    def __call__(self, x: jax.Array):
        x = self.conv1(x)
        x = nn.tanh(x)  # functional alias from fackle.nn.activations
        x = self.conv2(x)
        x = nn.tanh(x)
        return self.l1(x)


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

    # JIT mode
    model = model.compile(tx=optimizer, loss_fn=mse)

    # (For eager mode instead, delete the compile line above and use:
    #   loss = model.train_step(x, y, optimizer, mse))

    for step in range(100):
        loss = model.train_step(x, y)

        if step % 10 == 0:
            print(f"step={step}, loss={loss:.4f}")

    # ----------------------------
    # Testing / inference (no save/load needed)
    # ----------------------------
    print("\n--- Testing the trained model ---")

    test_x = jnp.ones((2, 5, 5, 3))

    # Just call the model! It uses the trained weights.
    predictions = model(test_x)

    print(f"Input shape: {test_x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample prediction (first pixel):\n{predictions[0, 0, 0, :]}")

    # ----------------------------
    # Bonus: prove the new nn/ package structure works
    # ----------------------------
    print("\n--- Sequential + activations demo ---")

    mlp = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    demo_x = jnp.ones((1, 4))
    print("Sequential output shape:", mlp(demo_x).shape)


if __name__ == "__main__":
    main()
