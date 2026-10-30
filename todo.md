# 2: Finish the CNN toolkit (easy win)
You have convs but can't build a classic CNN yet. Add to nn/layers.py:
- Flatten() — just x.reshape((x.shape[0], -1))
- MaxPool2D(window) / AvgPool2D(window) — jax.lax.reduce_window or the reshape+max trick
Now Conv2D → ReLU → MaxPool2D → Flatten → Linear becomes possible.

# 3: Normalization layers (great test of your Buffer system)
- LayerNorm — only Parameters, easy.
- BatchNorm2D — running mean/var as Buffers. Your engine already collects updated Buffers via has_aux, so this will prove your state system works on a real layer. This is where you'll also need a training: bool flag in __call__.

# 4: Dropout + train/eval mode (the forward-pass RNG problem)
Dropout needs randomness inside __call__, where a global RNG is dangerous under jit. This forces your second big design decision (NNX-style Rngs object threaded through the model, vs. explicit training/key args). Do this after BatchNorm, since both need the train/eval flag anyway.

# 5: Checkpointing (save / load)
Extract all .values into a flat dict with jax.tree_util, dump with jnp.savez. Small feature, huge practical value.

# 6: A tiny data utility
A batch_loader(dataset, batch_size, key) generator so people can train on real data instead of jnp.ones.
