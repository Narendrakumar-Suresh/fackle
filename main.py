import jax
import jax.numpy as jnp
import optax

from fackle.nn import Conv2D, ReLU, MaxPool2D, Flatten, Linear, Sequential


def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


def main():
    cnn = Sequential(
        Conv2D(3, 8, kernel_size=3, padding="SAME"),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(128, 10),
    )

    x = jnp.ones((2, 5, 5, 3))
    y = jnp.zeros((2, 10))

    print("forward shape:", cnn(x).shape)

    cnn.compile(tx=optax.sgd(learning_rate=0.01), loss_fn=mse)

    for step in range(20):
        loss = cnn.train_step(x, y)
        if step % 5 == 0:
            print(f"step={step}, loss={loss:.4f}")

    preds = cnn(x)
    print("preds shape:", preds.shape)
    print("first sample:", preds)


if __name__ == "__main__":
    main()
