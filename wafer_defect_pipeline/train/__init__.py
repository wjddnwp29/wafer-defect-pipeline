"""Training loops for DDPM and Consistency Models."""

from wafer_defect_pipeline.train.trainer import training_loop
from wafer_defect_pipeline.train.trainer_cm import train_consistency_model

__all__ = ["training_loop", "train_consistency_model"]
