# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional
import random
import math
import numpy as np
import torch
# import torchdata.datapipes as dp
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = int(os.environ.get("NODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, max_predict_range: int = 6, random_lead_time: bool = False, hrs_each_step: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.random_lead_time = random_lead_time
        self.hrs_each_step = hrs_each_step

    def __iter__(self):
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x[: -self.max_predict_range]  # N, C, H, W

            if self.random_lead_time:
                predict_ranges = torch.randint(low=1, high=self.max_predict_range, size=(inputs.shape[0],))
            else:
                predict_ranges = torch.ones(inputs.shape[0]).to(torch.long) * self.max_predict_range
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(inputs.dtype)
            output_ids = torch.arange(inputs.shape[0]) + predict_ranges
            outputs = y[output_ids]

            yield inputs, outputs, lead_times, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info

    def __iter__(self):
        for (inp, out, lead_times, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables, self.region_info
                else:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))]) 
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
    )


class GlobalForecastDataModule():
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        variables (list): List of input variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        predict_range (int, optional): Predict range.
        hrs_each_step (int, optional): Hours each step.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        root_dir,
        variables,
        buffer_size,
        out_variables=None,
        predict_range: int = 6,
        hrs_each_step: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
        self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
        self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(out_variables)

        self.val_clim = self.get_climatology("val", out_variables)
        self.test_clim = self.get_climatology("test", out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def get_climatology(self, partition="val", variables=None):
        path = os.path.join(self.hparams.root_dir, partition, "climatology.npz")
        clim_dict = np.load(path)
        if variables is None:
            variables = self.hparams.variables
        clim = np.concatenate([clim_dict[var] for var in variables])
        clim = torch.from_numpy(clim)
        return clim

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                            multi_dataset_training=False,
                        ),
                        max_predict_range=self.hparams.predict_range,
                        random_lead_time=False,
                        hrs_each_step=self.hparams.hrs_each_step,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.hparams.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.hparams.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.hparams.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
