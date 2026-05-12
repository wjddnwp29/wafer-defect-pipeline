# scripts/

CLI entry points that compose the library into runnable pipelines. To be filled
in during Phase 1 (after porting from HW08) and Phase 2 (Hydra wiring).

Planned scripts:

- `run_train_ddpm.py`    -- end-to-end DDPM training driven by `configs/`
- `run_train_cm.py`      -- Consistency Model distillation from a trained DDPM
- `run_eval.py`          -- compute FID per class for DDPM / DDIM / CM
- `run_generate.py`      -- batch synthetic sample generation by class
