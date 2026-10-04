# _legacy — quarantined, NOT autoloaded by Forge

Forge only loads top-level `scripts/*.py`; nothing in this directory registers
a `Script`, so it is inert.

- `custom_sampler.py` — experimental RES Solver (Covariance-Aware). Still
  imports `modules.sd_samplers*` at top level (breaks when upstream changes
  the sampler API). To revive it, port to the `core.forge_compat` pattern +
  lazy sampler registration.
- `sampling.py` — look-ahead Euler reference, not registered anywhere.
