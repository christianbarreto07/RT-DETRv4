"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math
import os
from pathlib import Path

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


def _load_interest_class_name_set():
    """Load interest class names from dataset config.

    Returns a set of lowercased names. If config is missing/invalid, returns None
    so the caller can fall back to all classes.
    """
    candidates: list[Path] = []

    env_cfg = os.getenv("DATASET_CLASSES_CONFIG_PATH")
    if env_cfg:
        candidates.append(Path(env_cfg).resolve())

    # det_solver.py -> solver/engine/RT-DETRv4/third_party/<project_root>
    project_root = Path(__file__).resolve().parents[4]
    candidates.append((project_root / "configs" / "dataset_classes.json").resolve())

    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            target_classes = payload.get("target_classes", [])
            interest_classes = payload.get("interest_classes", [])
            if not isinstance(target_classes, list):
                target_classes = []
            if not isinstance(interest_classes, list):
                interest_classes = []
            if not interest_classes:
                interest_classes = target_classes

            normalized = {
                str(name).strip().lower()
                for name in interest_classes
                if str(name).strip()
            }
            if normalized:
                print(f"[RTv4] Interest classes loaded ({len(normalized)}) from {cfg_path}")
                return normalized
        except Exception as e:
            print(f"[RTv4] Warning: could not parse interest classes from {cfg_path}: {e}")

    print("[RTv4] Interest classes config not found/invalid. Falling back to all classes.")
    return None


def _compute_interest_custom_fitness(coco_evaluator, interest_name_set):
    """Compute custom_fitness like YOLO callback, but from RTv4 COCOeval tensors.

    Formula: custom_fitness = 0.8 * mean_f1_interest + 0.2 * mean_ap50_interest
    """
    if coco_evaluator is None:
        return None

    coco_bbox_eval = None
    try:
        coco_bbox_eval = coco_evaluator.coco_eval.get("bbox")
    except Exception:
        coco_bbox_eval = None
    if coco_bbox_eval is None:
        return None

    eval_payload = getattr(coco_bbox_eval, "eval", None)
    params = getattr(coco_bbox_eval, "params", None)
    if not isinstance(eval_payload, dict) or params is None:
        return None

    precision = eval_payload.get("precision")
    if precision is None:
        return None

    def _seq_to_list(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    cat_ids = [int(v) for v in _seq_to_list(getattr(params, "catIds", None))]
    if not cat_ids:
        return None
    catid_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    iou_thrs = [float(v) for v in _seq_to_list(getattr(params, "iouThrs", None))]
    if not iou_thrs:
        return None
    iou_idx = min(range(len(iou_thrs)), key=lambda i: abs(float(iou_thrs[i]) - 0.5))

    area_labels = [str(v) for v in _seq_to_list(getattr(params, "areaRngLbl", None))]
    area_idx = area_labels.index("all") if "all" in area_labels else 0

    max_dets = [int(v) for v in _seq_to_list(getattr(params, "maxDets", None))]
    max_det_idx = len(max_dets) - 1 if max_dets else 0

    rec_thrs = [float(v) for v in _seq_to_list(getattr(params, "recThrs", None))]

    id_to_name = {}
    try:
        raw_cats = getattr(coco_evaluator.coco_gt, "cats", {})
        if isinstance(raw_cats, dict):
            id_to_name = {
                int(k): str(v.get("name", "")).strip().lower()
                for k, v in raw_cats.items()
                if isinstance(v, dict)
            }
    except Exception:
        id_to_name = {}

    if interest_name_set:
        selected_cat_ids = [
            cid for cid in cat_ids
            if id_to_name.get(int(cid), "") in interest_name_set
        ]
        # If mapping fails for any reason, keep training robust and fallback to all classes.
        if not selected_cat_ids:
            selected_cat_ids = list(cat_ids)
    else:
        selected_cat_ids = list(cat_ids)

    f1_scores, ap50_scores = [], []
    for cid in selected_cat_ids:
        try:
            k_idx = catid_to_idx[int(cid)]
            pr_curve = precision[iou_idx, :, k_idx, area_idx, max_det_idx]
            ap50_cls, f1_cls = calculate_scores(pr_curve, rec_thrs)
            ap50_scores.append(ap50_cls)
            f1_scores.append(f1_cls)
        except Exception:
            continue

    if not f1_scores or not ap50_scores:
        return None

    mean_f1_interest = sum(f1_scores) / len(f1_scores)
    mean_ap50_interest = sum(ap50_scores) / len(ap50_scores)
    custom_fitness = 0.8 * mean_f1_interest + 0.2 * mean_ap50_interest

    return {
        "custom_fitness": custom_fitness,
        "mean_f1_interest": mean_f1_interest,
        "mean_ap50_interest": mean_ap50_interest,
        "n_interest_classes_used": len(f1_scores),
    }


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        interest_name_set = _load_interest_class_name_set()
        best_custom_fitness = float('-inf')
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                try:
                    metric_value = float(test_stats[k][0])
                except Exception:
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = metric_value
                top1 = metric_value

            resume_custom = _compute_interest_custom_fitness(coco_evaluator, interest_name_set)
            if resume_custom is not None:
                best_custom_fitness = float(resume_custom['custom_fitness'])
                best_stat['custom_fitness'] = best_custom_fitness
                top1 = best_custom_fitness

            print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # Keep raw COCO metrics in logs/TensorBoard for compatibility.
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Val/{k}_{i}'.format(k), v, epoch)

                try:
                    metric_value = float(test_stats[k][0])
                except Exception:
                    continue

                if k in best_stat:
                    best_stat['epoch'] = epoch if metric_value > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], metric_value)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = metric_value

            custom_metrics = _compute_interest_custom_fitness(coco_evaluator, interest_name_set)
            epoch_custom_fitness = None
            if custom_metrics is not None:
                epoch_custom_fitness = float(custom_metrics['custom_fitness'])
                best_custom_fitness = max(best_custom_fitness, epoch_custom_fitness)
                best_stat['custom_fitness'] = float(best_custom_fitness)

                if self.writer and dist_utils.is_main_process():
                    self.writer.add_scalar('Val/custom_fitness', epoch_custom_fitness, epoch)
                    self.writer.add_scalar('Val/mean_f1_interest', float(custom_metrics['mean_f1_interest']), epoch)
                    self.writer.add_scalar('Val/mean_ap50_interest', float(custom_metrics['mean_ap50_interest']), epoch)

                print(
                    f"--> [EPOCH {epoch}] Custom Fitness (interest): {epoch_custom_fitness:.4f} "
                    f"(Best: {best_custom_fitness:.4f})"
                )

            # Use custom_fitness as the primary checkpoint criterion when available.
            epoch_selection_score = epoch_custom_fitness
            if epoch_selection_score is None:
                for k in test_stats:
                    try:
                        epoch_selection_score = float(test_stats[k][0])
                        break
                    except Exception:
                        continue

            improved = epoch_selection_score is not None and float(epoch_selection_score) > float(top1)

            if improved:
                best_stat_print['epoch'] = epoch
                top1 = float(epoch_selection_score)
                if self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

            for k in test_stats:
                try:
                    best_stat_print[k] = float(best_stat.get(k, test_stats[k][0]))
                except Exception:
                    pass

            if epoch_custom_fitness is not None:
                best_stat_print['custom_fitness'] = float(epoch_custom_fitness)
                best_stat_print['custom_fitness_best'] = float(best_custom_fitness)
                best_stat_print['mean_f1_interest'] = float(custom_metrics['mean_f1_interest'])
                best_stat_print['mean_ap50_interest'] = float(custom_metrics['mean_ap50_interest'])

            print(f'best_stat: {best_stat_print}')  # global best

            if (not improved) and epoch >= self.train_dataloader.collate_fn.stop_epoch:
                best_stat = {'epoch': -1, }
                if best_custom_fitness != float('-inf'):
                    best_stat['custom_fitness'] = float(best_custom_fitness)
                self.ema.decay -= 0.0001
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            if custom_metrics is not None:
                log_stats['val_custom_fitness'] = float(custom_metrics['custom_fitness'])
                log_stats['val_mean_f1_interest'] = float(custom_metrics['mean_f1_interest'])
                log_stats['val_mean_ap50_interest'] = float(custom_metrics['mean_ap50_interest'])
                log_stats['val_n_interest_classes_used'] = int(custom_metrics['n_interest_classes_used'])

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state