# Unmapped cross-backend results are invalid

The results in `cross_eval_full`, `cross_eval_full_seed2`,
`cross_eval_newton_train`, and `cross_eval_newton_train_seed2` must not support
a physics-backend claim. PhysX and Newton exposed different runtime joint
orders, so joint observations and actions changed semantics across backends.

`cross_eval_contract/summary.json` records the failed contract check.
`cross_eval_mapped_physx_train` and `cross_eval_mapped_newton_train` are the
bounded replacement pilots with checkpoint-source joint ordering enforced.
