#  ----------------------------
# # Layers
# # ----------------------------
# class Linear(Module):
#     def __init__(self, din, dout):
#         self.din = din
#         self.dout = dout
#         self.kernel = Param(jnp.ones((din, dout)))
#         self.bias = Param(jnp.zeros((dout,)))
#         self.call_count = BatchStat(jnp.array(0))
#
#     def __call__(self, x: jax.Array):
#         # State update happens inside the module itself.
#         # No external update_stats needed.
#         if self.call_count.value is not None:
#             self.call_count = BatchStat(self.call_count.value + 1)
#
#         return x @ self.kernel.value + self.bias.value
#
#
# class MLP(Module):
#     def __init__(self, din, hidden, dout):
#         self.layer1 = Linear(din, hidden)
#         self.layer2 = Linear(hidden, dout)
#
#     def __call__(self, x: jax.Array):
#         x = self.layer1(x)
#         x = jnp.tanh(x)
#         return self.layer2(x)
#
#
# # ----------------------------
# # Tree helpers
# # ----------------------------

#
#
# # ----------------------------
# # Loss
# # ----------------------------
# def mse(pred, y):
#     return jnp.mean((pred - y) ** 2)
#
#
# # ----------------------------
# # Training
# # ----------------------------
# model = MLP(din=2, hidden=4, dout=3)
#
# model.compile(tx=optax.sgd(learning_rate=0.01), loss_fn=mse)
#
# x = jnp.array([[1.0, 2.0]])
# y = jnp.array([[0.5, 0.5, 0.5]])
#
# for step in range(100):
#     loss = model.train_step(x, y)
#
#     if step % 10 == 0:
#         print(
#             f"step={step}, "
#             f"loss={loss:.4f}, "
#             f"layer1_call_count={model.layer1.call_count.value}, "
#             f"layer2_call_count={model.layer2.call_count.value}"
#         )
