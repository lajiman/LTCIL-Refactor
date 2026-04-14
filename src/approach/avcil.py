import torch
import torch.nn.functional as F
from copy import deepcopy
from argparse import ArgumentParser
from itertools import cycle

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """
    AVCIL-style incremental training (single-head classifier).
    - 对齐你给的 AVCIL 训练逻辑（CE + KD + 可选对比/注意力蒸馏）
    - 适配 Long-Tailed-CIL 的 Inc_Learning_Appr 接口
    - 暂不包含 2-stage
    """

    def __init__(self, model, device, nepochs=100, lr=0.001, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.0, wd=1e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None,
                 T=2.0, lam=0.5, lam_I=0.5, lam_C=1.0,
                 instance_contrastive=False, class_contrastive=False, attn_score_distil=False,
                 instance_contrastive_temperature=0.1, class_contrastive_temperature=0.1):
        super().__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                         momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,
                         logger, exemplars_dataset)

        self.model_old = None

        self.T = T
        self.lam = lam
        self.lam_I = lam_I
        self.lam_C = lam_C

        self.instance_contrastive = instance_contrastive
        self.class_contrastive = class_contrastive
        self.attn_score_distil = attn_score_distil

        self.instance_contrastive_temperature = instance_contrastive_temperature
        self.class_contrastive_temperature = class_contrastive_temperature

        # cache vars for criterion() in incremental mode
        self._cache_data_bs = None
        self._cache_ex_bs = None
        self._cache_old_out = None
        self._cache_old_spatial = None
        self._cache_old_temporal = None
        self._cache_audio_feature = None
        self._cache_visual_feature = None
        self._cache_spatial_attn = None
        self._cache_temporal_attn = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()

        parser.add_argument('--T', default=2.0, type=float, required=False,
                            help='KD temperature (default=%(default)s)')
        parser.add_argument('--lam', default=0.5, type=float, required=False,
                            help='Weight for spatial/temporal attn distillation mixing (default=%(default)s)')
        parser.add_argument('--lam-I', default=0.5, type=float, required=False,
                            help='Weight for instance contrastive loss (default=%(default)s)')
        parser.add_argument('--lam-C', default=1.0, type=float, required=False,
                            help='Weight for class contrastive loss (default=%(default)s)')

        parser.add_argument('--instance-contrastive', action='store_true', required=False,
                            help='Enable instance contrastive loss')
        parser.add_argument('--class-contrastive', action='store_true', required=False,
                            help='Enable class contrastive loss')
        parser.add_argument('--attn-score-distil', action='store_true', required=False,
                            help='Enable attention score distillation')

        parser.add_argument('--instance-contrastive-temperature', default=0.1, type=float, required=False,
                            help='Temperature for instance contrastive loss (default=%(default)s)')
        parser.add_argument('--class-contrastive-temperature', default=0.1, type=float, required=False,
                            help='Temperature for class contrastive loss (default=%(default)s)')

        return parser.parse_known_args(args)

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def train_loop(self, t, trn_loader, val_loader):
        """
        重写 train_loop:
        - step 0: 常规训练
        - step >0: 当前 task loader + exemplar loader 双流训练（对齐你给的 AVCIL 脚本 tzip/cycle）
        """
        lr = self.lr
        best_loss = float('inf')
        self.optimizer = self._get_optimizer()

        # step>0 且有 exemplars 时，建立 replay loader
        ex_loader = None
        use_replay = (t > 0) and (self.exemplars_dataset is not None) and (len(self.exemplars_dataset) > 0)
        if use_replay:
            # 2 data loaders: current task loader + exemplar loader
            ex_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True
            )

        for e in range(self.nepochs):
            # time the training epoch (optional, for logging)
            clock0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            clock1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if torch.cuda.is_available():
                clock0.record()

            self.train_epoch(t, trn_loader, ex_loader)

            if torch.cuda.is_available():
                clock1.record()
                torch.cuda.synchronize()
                train_time = clock0.elapsed_time(clock1) / 1000.0
            else:
                train_time = 0.0

            if self.eval_on_train:
                tr_loss, tr_acc, _ = self.eval(t, trn_loader)
                print('| Epoch {:3d}, time={:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, train_time, tr_loss, 100 * tr_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name='loss', value=tr_loss, group='train')
                self.logger.log_scalar(task=t, iter=e + 1, name='acc', value=100 * tr_acc, group='train')
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, train_time), end='')

            val_loss, val_acc, _ = self.eval(t, val_loader)
            print(' Valid: loss={:.3f}, TAw acc={:5.1f}% |'.format(val_loss, 100 * val_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name='loss', value=val_loss, group='valid')
            self.logger.log_scalar(task=t, iter=e + 1, name='acc', value=100 * val_acc, group='valid')

            if val_loss < best_loss:
                best_loss = val_loss
                print(' *', end='')

            if hasattr(self.model, 'schedule_step') and (e + 1) in self.model.schedule_step:
                lr /= self.lr_factor
                self.optimizer.param_groups[0]['lr'] = lr
                print(' lr={:.1e}'.format(lr), end='')

            self.logger.log_scalar(task=t, iter=e + 1, name='lr', value=lr, group='train')
            print()

        # after a training loop, collect exemplars for the current task if exemplars_dataset is provided
        if self.exemplars_dataset is not None:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader, ex_loader=None):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if t == 0 or ex_loader is None:
            iterator = trn_loader
            for data, labels in iterator:
                visual = data[0].to(self.device)
                audio = data[1].to(self.device)
                labels = labels.to(self.device)

                # align with avcil_network.py
                out, audio_feature, visual_feature = self.model(
                    (visual, audio), out_feature_before_fusion=True
                )
                outputs = [out]  # keep criterion style as multi-head-like list
                loss = self.criterion(t, outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
        else:
            iterator = zip(trn_loader, cycle(ex_loader))
            for (curr_data, curr_labels), (ex_data, ex_labels) in iterator:
                curr_visual = curr_data[0].to(self.device)
                curr_audio = curr_data[1].to(self.device)
                curr_labels = curr_labels.to(self.device)

                ex_visual = ex_data[0].to(self.device)
                ex_audio = ex_data[1].to(self.device)
                ex_labels = ex_labels.to(self.device)

                # data_bs and ex_bs are the batch sizes of current task data and exemplar data in this training step, respectively. They are used to split the combined batch for loss computation.
                data_bs = curr_labels.shape[0]
                ex_bs = ex_labels.shape[0]

                total_visual = torch.cat((curr_visual, ex_visual), dim=0)
                total_audio = torch.cat((curr_audio, ex_audio), dim=0)

                out, audio_feature, visual_feature, spatial_attn, temporal_attn = self.model(
                    (total_visual, total_audio),
                    out_feature_before_fusion=True,
                    out_attn_score=True
                )

                # Already have model_old.eval(), model_old.freeze_all() in post_train_process, so no_grad is just for safety to avoid tracking in autograd, since this is just a forward pass to get old logits/features/attn scores for distillation, not a real training step.
                with torch.no_grad():
                    old_out, old_spatial, old_temporal = self.model_old(
                        (total_visual, total_audio),
                        out_attn_score=True
                    )
                    # detach old outputs to avoid tracking in autograd, since they are just used as targets for distillation losses, not for gradient backpropagation.
                    old_out = old_out.detach()
                    old_spatial = old_spatial.detach()
                    old_temporal = old_temporal.detach()

                # cache for criterion()
                self._cache_data_bs = data_bs
                self._cache_ex_bs = ex_bs
                self._cache_old_out = old_out
                self._cache_old_spatial = old_spatial
                self._cache_old_temporal = old_temporal
                self._cache_audio_feature = audio_feature
                self._cache_visual_feature = visual_feature
                self._cache_spatial_attn = spatial_attn
                self._cache_temporal_attn = temporal_attn

                outputs = [out]
                targets = torch.cat((curr_labels, ex_labels), dim=0)
                loss = self.criterion(t, outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()

    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for data, targets in val_loader:
                visual = data[0].to(self.device)
                audio = data[1].to(self.device)
                targets = targets.to(self.device)

                # now the LLLNet in avcil_network.py has already wrap the logit as a list for criterion style, so we can directly use that without modification. If your model returns a single tensor, you can wrap it in a list like outputs = [self.model((visual, audio))] to keep the criterion style consistent.
                outputs_list = self.model((visual, audio))   # single head logits tensor [B, C]

                loss = self.criterion(t, outputs_list, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs_list, targets)

                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """
        Keep criterion style:
        - outputs: list with one tensor [B, C]
        - targets: global labels
        """
        out = outputs[0]

        if t == 0 or self._cache_data_bs is None:
            # Just ce loss for step 0
            return self._ce_loss(out, targets)

        data_bs = self._cache_data_bs
        ex_bs = self._cache_ex_bs
        old_out = self._cache_old_out
        old_spatial = self._cache_old_spatial
        old_temporal = self._cache_old_temporal
        audio_feature = self._cache_audio_feature
        visual_feature = self._cache_visual_feature
        spatial_attn = self._cache_spatial_attn
        temporal_attn = self._cache_temporal_attn

        curr_labels = targets[:data_bs]
        ex_labels = targets[data_bs:data_bs + ex_bs]

        # 当前 step 的新类索引范围
        # 单头设置下：old 类个数 = 已学总类 - 当前task类数
        total_cls = out.shape[1]
        # 依赖 dataloader 固定每task类数
        # 从 model.task_cls 提取总类（单头是[total]）
        # old 类数 = total - 当前task真实类别数（由当前batch标签推断不稳），这里按数据集协议用 task_offset:
        # 在该工程中 main_incremental 的 taskcla 每task类别固定，labels天然全局id，因此可用：
        # old_cls = t * ncls_per_task
        # ncls_per_task 可以从当前 task 中 unique 估计；更稳妥是用 model.task_offset（单头时只有[0]）
        # 因为你是固定切分，这里用当前batch新类范围按 t 推断：
        # 通过 labels 可估 ncls_per_task，但更稳定的方法是在命令行配置里固定。
        # 这里采用与原AVCIL一致：使用当前task真实新类数 = curr_labels unique上限不可靠 -> 用增量边界 from dataset split
        # 简化并稳定：从 old_out.shape[1] 和 t 推断不行；故直接按当前任务新类logits切片：
        # 你当前实验固定每task同类数时有效（与原脚本一致）

        # n_new aims to estimate the number of new classes. To infer, rather than using a specific number, is more flexible and can adapt to different splits. 
        # however, this part is tricky, and different from original AVCIL code
        n_new = torch.unique(curr_labels).numel()
        # 为避免 batch 未覆盖全部新类，fallback 用近似: 平均task大小. However, it doesn't work if task sizes are different, or the first task contains many classes (for example, 50/10/10 split)
        if t > 0:
            n_new = max(n_new, int(total_cls // (t + 1)))
        else:
            n_new = total_cls
        old_cls = total_cls - n_new

        # For example, [20, 21, 22] -> [0, 1, 2] for a task with 3 new classes starting from global class 20.
        curr_labels_local = curr_labels - old_cls
        '''
            Original AVCIL code:
                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_)

                prev_out = out[data_batch_size:data_batch_size+exemplar_data_batch_size, :last_step_out_class_num]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

            def CE_loss(num_classes, logits, label):
                targets = F.one_hot(label, num_classes=num_classes)
                loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

                return loss
        '''
        # the implementations here should get the correct answer, but we should aware that it's implementation is different from original AVCIL code
        curr_out = out[:data_bs, old_cls:]
        prev_out = out[data_bs:data_bs + ex_bs, :old_cls]

        loss_curr = self._ce_loss_with_nclass(curr_out, curr_labels_local, curr_out.shape[1])
        loss_prev = self._ce_loss_with_nclass(prev_out, ex_labels, prev_out.shape[1]) if old_cls > 0 else 0.0
        if old_cls > 0:
            loss_ce = (loss_curr * data_bs + loss_prev * ex_bs) / (data_bs + ex_bs)
        # actually we don't need to consider this branch, bucause '_forward_loss_incremental' contains that old_cls>0
        else:
            loss_ce = loss_curr

        '''
            Original AVCIL code:
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss_KD = loss_KD.sum()
        '''
        loss_kd = 0.0
        if old_cls > 0 and t > 0:
            # original AVCIL code use args.class_num_per_step to slice logits
            step_size = old_cls // t
            old_logits = old_out[:, :old_cls]
            cur_logits = out[:, :old_cls]
            for j in range(t):
                s = j * step_size
                e = (j + 1) * step_size if j < t - 1 else old_cls
                soft_target = F.softmax(old_logits[:, s:e] / self.T, dim=1)
                output_log = F.log_softmax(cur_logits[:, s:e] / self.T, dim=1)
                loss_kd = loss_kd + F.kl_div(output_log, soft_target, reduction='batchmean') * (self.T ** 2)

        loss = loss_ce + loss_kd

        # this three parts(instance contrastive, class contrastive, attn score distil) are actually the same as original AVCIL code
        if self.instance_contrastive:
            loss_ic = self._instance_contrastive_loss(audio_feature, visual_feature,
                                                      temperature=self.instance_contrastive_temperature)
            loss = loss + self.lam_I * loss_ic

        if self.class_contrastive:
            all_labels = targets[:data_bs + ex_bs]
            loss_cc = self._class_contrastive_loss(audio_feature, visual_feature, all_labels,
                                                   temperature=self.class_contrastive_temperature)
            loss = loss + self.lam_C * loss_cc

        if self.attn_score_distil and ex_bs > 0:
            ex_sp = spatial_attn[data_bs:data_bs + ex_bs].transpose(2, 3).reshape(-1, spatial_attn.shape[2])
            ex_old_sp = old_spatial[data_bs:data_bs + ex_bs].transpose(2, 3).reshape(-1, old_spatial.shape[2])

            ex_tp = temporal_attn[data_bs:data_bs + ex_bs].transpose(1, 2).reshape(-1, temporal_attn.shape[1])
            ex_old_tp = old_temporal[data_bs:data_bs + ex_bs].transpose(1, 2).reshape(-1, old_temporal.shape[1])

            loss_sp = F.kl_div(ex_sp.log(), ex_old_sp, reduction='sum') / ex_bs
            loss_tp = F.kl_div(ex_tp.log(), ex_old_tp, reduction='sum') / ex_bs
            loss = loss + self.lam * loss_sp + (1.0 - self.lam) * loss_tp

        return loss

    # ------- helper losses -------
    @staticmethod
    def _ce_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _ce_loss_with_nclass(logits, labels, num_classes):
        targets = F.one_hot(labels, num_classes=num_classes).float()
        return -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    def _instance_contrastive_loss(self, feature_1, feature_2, temperature=0.1):
        score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
        n = score.shape[0]
        label = torch.arange(n).to(score.device)
        return self._ce_loss_with_nclass(score, label, n)

    def _class_contrastive_loss(self, feature_1, feature_2, label, temperature=0.1):
        class_matrix = label.unsqueeze(0)
        class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
        class_matrix = (class_matrix == label.unsqueeze(-1)).float()

        score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
        return -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))