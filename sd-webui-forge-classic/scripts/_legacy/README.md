# _legacy — quarantined, NOT autoloaded by Forge

Forge chỉ load `scripts/*.py` top-level; thư mục này không có `Script` nào được
đăng ký nên an toàn.

- `custom_sampler.py` — RES Solver (Covariance-Aware) thử nghiệm, còn import
  `modules.sd_samplers*` ở top-level (dễ gãy khi upstream đổi sampler API).
  Muốn dùng lại thì port sang pattern `core.forge_compat` + đăng ký sampler lazy.
- `sampling.py` — Euler look-ahead tham khảo, không được rào đăng ký ở đâu cả.
