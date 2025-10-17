# src/main.py
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import ClearMLLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    # reproducibility
    pl.seed_everything(cfg.seed)

    # ---------- Instantiate DataModule (pre) ----------
    data_module = instantiate(cfg.data_module)

    # ---------- Sync dims/memory/treatment_cond ----------
    # 1) пробрасываем memory из модели в датамодуль (если есть)
    if "memory" in cfg.model_module:
        cfg.data_module.memory = cfg.model_module.memory

    # 2) подставляем входную размерность
    x_dim = getattr(data_module, "dims", [None])[0]
    if x_dim is not None:
        if "dim" in cfg.model_module:
            cfg.model_module.dim = x_dim
        elif "input_dim" in cfg.model_module and "output_dim" in cfg.model_module:
            cfg.model_module.input_dim = x_dim
            cfg.model_module.output_dim = x_dim

    # 3) для условных моделей treatment_cond = len(cond_headings)
    if "treatment_cond" in cfg.model_module:
        cfg.model_module.treatment_cond = len(getattr(data_module, "cond_headings", []))

    # ---------- Instantiate model ----------
    model = instantiate(cfg.model_module)

    # ---------- train_consecutive switch ----------
    # как в оригинале: у условных моделей train_consecutive=False, иначе True
    is_cond_model = ("naming" in dir(model)) and ("Cond" in getattr(model, "naming", ""))
    cfg.data_module.train_consecutive = not is_cond_model
    data_module = instantiate(cfg.data_module)  # пересоздать с обновлённым флагом

    # ---------- Logger: ClearML ----------
    clearml_logger = None
    if cfg.clearml.enable and not cfg.skip_training:
        clearml_logger = ClearMLLogger(
            project=cfg.clearml.project,
            task_name=cfg.clearml.task_name,
            tags=cfg.clearml.tags,
        )
        # Логоним полный, резолвленный конфиг
        clearml_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # ---------- Checkpoint & EarlyStopping ----------
    run_dir = os.getcwd() 
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stop_patience,
        mode="min",
        verbose=True,
    )

    # ---------- Strategy ----------
    strategy = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto"

    # ---------- Trainer ----------
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        max_time=cfg.trainer.max_time,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=clearml_logger,
        limit_train_batches=0.0 if cfg.skip_training else 1.0,
        strategy=strategy,
        deterministic=True,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # ---------- Train / Test ----------
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    train_model()
