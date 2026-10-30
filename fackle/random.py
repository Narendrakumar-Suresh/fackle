import jax

_STATE = {"key": jax.random.PRNGKey(0)}


def seed(seed_int: int):
    """PyTorch-style manual seed."""
    _STATE["key"] = jax.random.PRNGKey(seed_int)


def next_key() -> jax.Array:
    """Draw a fresh unique key from the global RNG."""
    _STATE["key"], sub = jax.random.split(_STATE["key"])
    return sub
