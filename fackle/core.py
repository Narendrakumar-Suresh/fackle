import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from types import SimpleNamespace


# ----------------------------
# Variable system
# ----------------------------
@jtu.register_pytree_node_class
class Variable:
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        children = list(children)
        return cls(children[0])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


@jtu.register_pytree_node_class
class Parameter(Variable):
    pass


@jtu.register_pytree_node_class
class Buffer(Variable):
    pass


def filter_tree(tree, keep_cls):
    """
    Keep only variables matching keep_cls.
    Replace other Variables with same-type placeholders containing None.
    """

    def fn(x):
        if isinstance(x, keep_cls):
            return x

        if isinstance(x, Variable):
            return type(x)(None)

        return x

    return jax.tree_util.tree_map(fn, tree, is_leaf=lambda x: isinstance(x, Variable))


def merge_trees(*trees):
    """
    Merge multiple filtered trees back into one full tree.
    """

    def fn(*xs):
        for x in xs:
            if isinstance(x, Variable) and x.value is not None:
                return x

        return xs[0]

    return jax.tree_util.tree_map(fn, *trees, is_leaf=lambda x: isinstance(x, Variable))


# ----------------------------
# Module base class
# ----------------------------
class Module:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node_class(cls)

    def tree_flatten(self):
        child_values = []
        dynamic_keys = []
        static_items = []

        for key, value in vars(self).items():
            # Ignore private framework internals
            if key.startswith("_"):
                continue

            if isinstance(value, (Module, Variable)):
                child_values.append(value)
                dynamic_keys.append(key)
            else:
                static_items.append((key, value))

        aux_data = (tuple(dynamic_keys), tuple(static_items))
        return tuple(child_values), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamic_keys, static_items = aux_data
        children = list(children)

        obj = cls.__new__(cls)

        for key, value in static_items:
            setattr(obj, key, value)

        for key, value in zip(dynamic_keys, children):
            setattr(obj, key, value)

        return obj

    def __repr__(self):
        public = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{self.__class__.__name__}({public})"

    # ----------------------------
    # Training API
    # ----------------------------
    def compile(self, tx, loss_fn):
        params = filter_tree(self, Parameter)
        stats = filter_tree(self, Buffer)

        opt_state = tx.init(params)

        loss_fn_impl = loss_fn
        tx_update = tx.update

        @jax.jit
        def backward_kernel(params, stats, x, y):
            def loss_and_new_state(params):
                full_model = merge_trees(params, stats)

                # Forward pass.
                # Layers may update their own Buffer variables here.
                pred = full_model(x)

                loss = loss_fn_impl(pred, y)

                # Automatically collect updated state
                new_stats = filter_tree(full_model, Buffer)

                return loss, new_stats

            (loss, new_stats), grads = jax.value_and_grad(
                loss_and_new_state, has_aux=True
            )(params)

            return loss, grads, new_stats

        @jax.jit
        def step_kernel(params, opt_state, grads):
            updates, opt_state = tx_update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        self._training = SimpleNamespace(
            params=params,
            stats=stats,
            opt_state=opt_state,
            grads=None,
            new_stats=None,
            last_loss=None,
            backward_kernel=backward_kernel,
            step_kernel=step_kernel,
        )

    def backward(self, x, y, tx=None, loss_fn=None):
        # Lazy compile support
        if not hasattr(self, "_training"):
            if tx is None or loss_fn is None:
                raise RuntimeError(
                    "Call model.compile(tx, loss_fn) first, "
                    "or pass tx and loss_fn on the first backward call."
                )
            self.compile(tx, loss_fn)

        t = self._training

        loss, grads, new_stats = t.backward_kernel(t.params, t.stats, x, y)

        t.grads = grads
        t.new_stats = new_stats
        t.last_loss = loss

        return loss

    def step(self):
        if not hasattr(self, "_training"):
            raise RuntimeError("Call backward first.")

        t = self._training

        if t.grads is None:
            raise RuntimeError("No gradients found. Call backward first.")

        new_params, new_opt_state = t.step_kernel(t.params, t.opt_state, t.grads)

        new_stats = t.new_stats if t.new_stats is not None else t.stats

        t.params = new_params
        t.stats = new_stats
        t.opt_state = new_opt_state
        t.grads = None
        t.new_stats = None

        # Update public model attributes
        full_model = merge_trees(new_params, new_stats)

        for key, value in vars(full_model).items():
            if not key.startswith("_"):
                setattr(self, key, value)

    def train_step(self, x, y, tx=None, loss_fn=None):
        loss = self.backward(x, y, tx=tx, loss_fn=loss_fn)
        self.step()
        return loss
