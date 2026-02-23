# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import time
import warnings

import matplotlib
import torch

matplotlib.use("AGG")

import numpy as np
import torch.nn as nn
from torch import distributed as dist

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    callbacks,
    colorstr,
)
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.distill_loss import FeatureLoss, LogicalLoss

# from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, callbacks, __version__
# from ultralytics.nn.extra_modules.kernel_warehouse import get_temperature
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    de_parallel,
    unset_deterministic,
)


def get_activation(feat, backbone_idx=-1):
    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)):
                outputs.insert(0, None)
            # for idx, i in enumerate(outputs):
            #     if i is None:
            #         print(idx, 'None')
            #     else:
            #         print(idx, i.size())
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)

    return hook


class DetectionDistiller(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.logical_disloss = None
        self.feature_disloss = None
        # self.pretrain_weights = None

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names) + 2)) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "log_loss",
            "fea_loss",
            "Instances",
            "Size",
        )

    def setup_teacher_model(self):
        """Load/create/download model for any task."""
        # model, weights = self.args.teacher_weights, None
        # ckpt = None
        # if str(model).endswith('.pt'):
        #     weights, ckpt = attempt_load_one_weight(model)
        #     cfg = weights.yaml
        # else:
        #     cfg = model
        # self.teacher_model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)

        ckpt = torch.load(self.args.teacher_weights, map_location=self.device)
        self.teacher_model = ckpt["ema" if ckpt.get("ema") else "model"].float()
        self.teacher_model.train()
        self.teacher_model.info()
        return ckpt

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        LOGGER.info("Setup student model")
        ckpt = self.setup_model()
        # if self.pretrain_weights:
        #     self.model.load(torch.load(self.pretrain_weights, map_location=self.device))
        self.model = self.model.to(self.device)
        LOGGER.info("Setup teacher model")
        _ = self.setup_teacher_model()
        self.teacher_model.to(self.device)
        self.set_model_attributes()
        self.model.criterion = self.model.init_criterion()
        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            # self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # TODO：多GPU蒸馏可能需要修改
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Init Distill Loss
        self.kd_logical_loss, self.kd_feature_loss = None, None
        if self.args.kd_loss_type == "logical" or self.args.kd_loss_type == "all":
            self.kd_logical_loss = LogicalLoss(self.args, self.model, self.args.logical_loss_type)
        if self.args.kd_loss_type == "feature" or self.args.kd_loss_type == "all":
            s_feature, t_feature = [], []
            hooks = []
            self.teacher_kd_layers, self.student_kd_layers = (
                self.args.teacher_kd_layers.split(","),
                self.args.student_kd_layers.split(","),
            )
            s_feature_idx, t_feature_idx = [], []
            assert len(self.teacher_kd_layers) == len(self.student_kd_layers), (
                f"teacher{self.teacher_kd_layers} and student{self.student_kd_layers} layers not equal.."
            )
            for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                if "-" in t_layer:
                    t_layer_first, t_layer_second = t_layer.split("-")
                    t_feature_idx.append(int(t_layer_second) / 10)
                    hooks.append(
                        de_parallel(self.teacher_model)
                        .model[int(t_layer_first)]
                        .register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second)))
                    )
                else:
                    hooks.append(
                        de_parallel(self.teacher_model)
                        .model[int(t_layer)]
                        .register_forward_hook(get_activation(t_feature))
                    )
                    t_feature_idx.append(int(t_layer))

                if "-" in s_layer:
                    s_layer_first, s_layer_second = s_layer.split("-")
                    s_feature_idx.append(int(s_layer_second) / 10)
                    hooks.append(
                        de_parallel(self.model)
                        .model[int(s_layer_first)]
                        .register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second)))
                    )
                else:
                    hooks.append(
                        de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature))
                    )
                    s_feature_idx.append(int(s_layer))

            inputs = torch.randn((2, 6, self.args.imgsz, self.args.imgsz)).to(self.device)
            with torch.no_grad():
                _ = self.teacher_model(inputs)
                _ = self.model(inputs)
            s_feature_sort_idx, t_feature_sort_idx = sorted(s_feature_idx), sorted(t_feature_idx)
            s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
            t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]
            self.kd_feature_loss = FeatureLoss(
                [s_feature[i].size(1) for i in s_feature_idx],
                [t_feature[i].size(1) for i in t_feature_idx],
                distiller=self.args.feature_loss_type,
            )
            for hook in hooks:
                hook.remove()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model with the specified world size."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        print("设备序列", self.device)
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()

            # TODO: 修改为DDP
            if self.args.kd_loss_type in ["feature", "all"]:
                self.kd_feature_loss.train()
                hooks = []
                s_feature, t_feature = [], []
                s_feature_idx, t_feature_idx = [], []
                for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                    if "-" in t_layer:
                        t_layer_first, t_layer_second = t_layer.split("-")
                        t_feature_idx.append(int(t_layer_second) / 10)
                        hooks.append(
                            de_parallel(self.teacher_model)
                            .model[int(t_layer_first)]
                            .register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second)))
                        )
                    else:
                        hooks.append(
                            de_parallel(self.teacher_model)
                            .model[int(t_layer)]
                            .register_forward_hook(get_activation(t_feature))
                        )
                        t_feature_idx.append(int(t_layer))

                    if "-" in s_layer:
                        s_layer_first, s_layer_second = s_layer.split("-")
                        s_feature_idx.append(int(s_layer_second) / 10)
                        hooks.append(
                            de_parallel(self.model)
                            .model[int(s_layer_first)]
                            .register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second)))
                        )
                    else:
                        hooks.append(
                            de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature))
                        )
                        s_feature_idx.append(int(s_layer))

                s_feature_sort_idx, t_feature_sort_idx = sorted(s_feature_idx), sorted(t_feature_idx)
                s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
                t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            # start:新增蒸馏损失
            self.logical_disloss = torch.zeros(1, device=self.device)
            self.feature_disloss = torch.zeros(1, device=self.device)
            self.optimizer.zero_grad()
            # end
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                if hasattr(de_parallel(self.model), "net_update_temperature"):
                    temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                    de_parallel(self.model).net_update_temperature(temp)

                if self.args.kd_loss_decay == "constant":
                    distill_decay = 1.0
                elif self.args.kd_loss_decay == "cosine":
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
                elif self.args.kd_loss_decay == "linear":
                    distill_decay = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.01 - 1) + 1
                elif self.args.kd_loss_decay == "cosine_epoch":
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * ni / T_max)) / 2
                elif self.args.kd_loss_decay == "linear_epoch":
                    distill_decay = ((1 - math.cos(ni * math.pi / (self.epochs * nb))) / 2) * (0.01 - 1) + 1

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    pred = de_parallel(self.model).predict(batch["img"])

                    with torch.no_grad():
                        t_pred = de_parallel(self.teacher_model).predict(batch["img"])
                    # print("criterion:", de_parallel(self.model).criterion)
                    main_loss, self.loss_items = de_parallel(self.model).criterion(pred, batch)

                    log_distill_loss, fea_distill_loss = (
                        torch.zeros(1, device=self.device),
                        torch.zeros(1, device=self.device),
                    )
                    if self.kd_logical_loss is not None:
                        if type(pred) is dict and type(t_pred) is dict:
                            log_distill_loss = (
                                self.kd_logical_loss(pred["one2one"], t_pred["one2one"], batch, True)
                                * self.args.logical_loss_ratio
                            )
                            log_distill_loss += (
                                self.kd_logical_loss(pred["one2many"], t_pred["one2many"], batch)
                                * self.args.logical_loss_ratio
                                * 0.5
                            )
                        else:
                            log_distill_loss = self.kd_logical_loss(pred, t_pred, batch) * self.args.logical_loss_ratio
                    if self.kd_feature_loss is not None:
                        fea_distill_loss = (
                            self.kd_feature_loss(
                                [s_feature[i] for i in s_feature_idx], [t_feature[i] for i in t_feature_idx]
                            )
                            * self.args.feature_loss_ratio
                        )

                    # print("main loss", main_loss, log_distill_loss, fea_distill_loss)
                    self.loss = (
                        main_loss.sum()
                        + (log_distill_loss + fea_distill_loss) * batch["img"].size(0) * distill_decay * 0.2
                    )
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    self.logical_disloss = (
                        (self.logical_disloss * i + log_distill_loss) / (i + 1)
                        if self.logical_disloss is not None
                        else log_distill_loss
                    )
                    self.feature_disloss = (
                        (self.feature_disloss * i + fea_distill_loss) / (i + 1)
                        if self.feature_disloss is not None
                        else fea_distill_loss
                    )
                    # print(self.loss)
                    # Backward
                    self.scaler.scale(self.loss).backward()

                    # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break
                    # start: 新增
                    mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                    loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                    logical_dislosses = (
                        self.logical_disloss if loss_len > 1 else torch.unsqueeze(self.logical_disloss, 0)
                    )
                    feature_dislosses = (
                        self.feature_disloss if loss_len > 1 else torch.unsqueeze(self.feature_disloss, 0)
                    )
                    # end
                    # Log
                    if RANK in {-1, 0}:
                        self.tloss.shape[0] if len(self.tloss.shape) else 1
                        # pbar.set_description(
                        #     ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        #     % (
                        #         f"{epoch + 1}/{self.epochs}",
                        #         f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                        #         *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                        #         batch["cls"].shape[0],  # batch size, i.e. 8
                        #         batch["img"].shape[-1],  # imgsz, i.e 640
                        #     )
                        # )
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_len + 2))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                mem,
                                *losses,
                                *logical_dislosses,
                                *feature_dislosses,
                                batch["cls"].shape[0],
                                batch["img"].shape[-1],
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")
                    if self.kd_feature_loss is not None:
                        s_feature.clear()
                        t_feature.clear()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks("on_train_epoch_end")

            if self.kd_feature_loss is not None:
                for hook in hooks:
                    hook.remove()

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def distill(self, weights=None):
        # self.pretrain_weights = weights
        self.train()
