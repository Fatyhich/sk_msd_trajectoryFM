# src/data/datamodule.py
import os
from typing import Optional
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from trajdata import UnifiedDataset
from .batch_proccessing import make_model_collate


class TrajDataModule(pl.LightningDataModule):
    """
    Lightning-совместимый DataModule для работы с UnifiedDataset (trajdata).
    Поддерживает сплит по сэмплам или по сценам.
    """

    def __init__(
        self,
        root: str,
        desired_dt: float = 0.1,
        state_format: str = "x,y",
        obs_format: str = "x,y",
        centric: str = "scene",
        history_sec=(0.8, 0.8),
        future_sec=(0.8, 0.8),
        standardize: bool = False,
        train_consecutive: bool = True,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = None,
        memory: int = 4,
        dim: int = 2,
        # --- split options ---
        split_method: str = "scene",   # "scene" | "sample"
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        super().__init__()

        # базовые параметры
        self.root = root
        self.desired_dt = desired_dt
        self.state_format = state_format
        self.obs_format = obs_format
        self.centric = centric
        self.history_sec = tuple(history_sec)
        self.future_sec = tuple(future_sec)
        self.standardize = standardize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_consecutive = train_consecutive
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.memory = memory
        self.dim = dim

        # split params
        self.split_method = split_method
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        self.prepare_data_per_node = True

        # placeholders
        self.dataset = None
        self.collate = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    @property
    def dims(self):
        """Размерность признаков (x, y)."""
        return (self.dim,)

    def setup(self, stage: Optional[str] = None):
        """Создаёт основной датасет и делит его на train/val/test"""
        print(f"[Setup] Loading UnifiedDataset from {self.root}")
        self.dataset = UnifiedDataset(
            desired_data=[
                "eupeds_eth", "eupeds_hotel", "eupeds_univ",
                "eupeds_zara1", "eupeds_zara2",
            ],
            data_dirs={name: self.root for name in [
                "eupeds_eth", "eupeds_hotel", "eupeds_univ",
                "eupeds_zara1", "eupeds_zara2",
            ]},
            desired_dt=self.desired_dt,
            state_format=self.state_format,
            obs_format=self.obs_format,
            centric=self.centric,
            history_sec=self.history_sec,
            future_sec=self.future_sec,
            standardize_data=self.standardize,
        )

        self.collate = make_model_collate(
            dataset=self.dataset, memory=self.memory, dim=self.dim
        )

        if self.split_method == "scene":
            self._split_by_scene()
        elif self.split_method == "sample":
            self._split_by_sample()
        else:
            raise ValueError(f"Unknown split_method={self.split_method}")

    # ---------------- Split Methods ----------------
    def _split_by_sample(self):
        """Простое случайное разбиение по сэмплам."""
        N = len(self.dataset)
        n_train = int(round(N * self.train_ratio))
        n_val = int(round(N * self.val_ratio))
        n_test = max(0, N - n_train - n_val)

        gen = torch.Generator().manual_seed(self.split_seed)
        self.ds_train, self.ds_val, self.ds_test = random_split(
            self.dataset, lengths=[n_train, n_val, n_test], generator=gen
        )

        print(f"[Sample Split] train={n_train}, val={n_val}, test={n_test} (N={N})")

    def _split_by_scene(self):
        """Гарантирует, что все агенты из одной сцены попадут в один сплит."""
        if not hasattr(self.dataset, "scene_ids"):
            print("[warn] dataset не содержит scene_ids, откат к sample split")
            self._split_by_sample()
            return

        scene_ids = np.array(self.dataset.scene_ids)
        unique_scenes = np.unique(scene_ids)

        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(unique_scenes)

        n_scenes = len(unique_scenes)
        n_train = int(round(n_scenes * self.train_ratio))
        n_val = int(round(n_scenes * self.val_ratio))
        n_test = max(0, n_scenes - n_train - n_val)

        train_scenes = set(unique_scenes[:n_train])
        val_scenes = set(unique_scenes[n_train:n_train + n_val])
        test_scenes = set(unique_scenes[n_train + n_val:])

        idx_train, idx_val, idx_test = [], [], []
        for i, sid in enumerate(scene_ids):
            if sid in train_scenes:
                idx_train.append(i)
            elif sid in val_scenes:
                idx_val.append(i)
            else:
                idx_test.append(i)

        self.ds_train = Subset(self.dataset, idx_train)
        self.ds_val = Subset(self.dataset, idx_val)
        self.ds_test = Subset(self.dataset, idx_test)

        print(
            f"[Scene Split] {len(train_scenes)} train scenes, {len(val_scenes)} val scenes, {len(test_scenes)} test scenes"
        )
        print(
            f"[Scene Split] samples: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
        )

    # ---------------- Lightning Hooks ----------------
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
