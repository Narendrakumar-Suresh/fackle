# Fackel 

**Fackel** is a lightweight, object-oriented deep learning framework built on top of JAX and PyTrees[cite: 1]. It bridges JAX's native functional programming model with a clean, modular, PyTorch-like developer experience.

---

> **⚠️ Note on Early Development:** 
> Fackel is currently in early-stage development and is being actively built. APIs, core modules, and feature sets are subject to change. Contributions, feedback, and experimentation are welcome!

---

## Key Features

* **Object-Oriented Abstractions:** Define neural network layers and containers using class-based `Module` definitions without managing explicit, manual parameter state dicts everywhere.
* **Native JAX PyTree Integration:** Custom modules, `Parameter`, and `Buffer` states automatically register as JAX PyTrees[cite: 1], making transformations (`grad`, `vmap`, `jit`) seamless.
* **Explicit Layer APIs:** Dedicated modules (`Linear`, `Conv1D`, `Conv2D`, `Conv3D`) designed for optimal IDE intellisense and static type checking.
* **Built-in Training Loop:** Streamlined `.compile()` and `.train_step()` workflows integrating directly with `optax` optimizers[cite: 1].

---

## Installation

```bash
pip install fackel
```

Quickstart Example
Here is how to define a convolutional network and run a training step using Fackel:
```python
import jax
import jax.numpy as jnp
import optax

from fackle.core import Module
from fackle.nn import Conv2D


class ConvNet(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1 = Conv2D(in_channels, hidden_channels, kernel_size=3, padding="SAME")
        self.conv2 = Conv2D(hidden_channels, out_channels, kernel_size=3, padding="SAME")

    def __call__(self, x: jax.Array):
        x = self.conv1(x)
        x = jnp.tanh(x)
        return self.conv2(x)


# 1. Initialize model
model = ConvNet(in_channels=3, hidden_channels=8, out_channels=3)

# 2. Setup optimizer and loss
optimizer = optax.sgd(learning_rate=0.01)
loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)

# 3. Compile or run training step (NHWC layout format)
x = jnp.ones((1, 5, 5, 3))
y = jnp.zeros((1, 5, 5, 3))

for step in range(100):
    loss = model.train_step(x, y, optimizer, loss_fn)
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss:.4f}")
```

### License
Distributed under the MIT License. See LICENSE for details.
