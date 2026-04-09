# datasets/vgg_dataset.py

import os
import csv
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VGGDataset(Dataset):
    """
    FACIL-style multimodal dataset for VGGSound / ksounds / AVE-like feature datasets.

    This dataset stores only:
        - sample ids (vids)
        - labels
    in memory, and loads pretrained features online in __getitem__.

    Parameters
    ----------
    data : dict
        {
            'x': [vid1, vid2, ...],
            'y': [label1, label2, ...]
        }

    transform : optional
        Usually None for pretrained features, but kept for interface consistency.

    class_indices : optional
        FACIL-compatible placeholder.

    feature_root : str
        Root directory of pretrained features.

    modality : str
        'visual', 'audio', or 'audio-visual'

    dataset_name : str
        Used to distinguish AVE vs VGGSound-like visual feature backend.
    """

    def __init__(
        self,
        data: Dict[str, List[Any]],
        transform=None,
        class_indices=None,
        feature_root: Optional[str] = None,
        modality: str = "audio-visual",
        dataset_name: str = "VGGSound",
        specific: Optional[Dict[str, Any]] = None,
    ):
        self.vids = data["x"]
        self.labels = data["y"]
        self.transform = transform
        self.class_indices = class_indices

        self.feature_root = feature_root
        self.modality = modality
        self.dataset_name = dataset_name
        self.specific = specific or {}

        if self.feature_root is None:
            raise ValueError("feature_root must be provided for VGGDataset")

        if self.modality not in {"visual", "audio", "audio-visual"}:
            raise ValueError("modality must be one of {'visual', 'audio', 'audio-visual'}")

        # visual backend
        self._visual_h5 = None
        self._visual_h5_path = None
        self._visual_dict = None

        if "visual" in self.modality:
            if self.dataset_name == "AVE":
                visual_path = os.path.join(self.feature_root, "visual_pretrained_feature_dict.npy")
                if not os.path.exists(visual_path):
                    raise FileNotFoundError(visual_path)
                self._visual_dict = np.load(visual_path, allow_pickle=True).item()
            else:
                visual_path = os.path.join(self.feature_root, "visual_features.h5")
                if not os.path.exists(visual_path):
                    raise FileNotFoundError(visual_path)
                self._visual_h5_path = visual_path

        # audio backend
        self._audio_dict = None
        if "audio" in self.modality:
            audio_path = os.path.join(
                self.feature_root,
                "audio_pretrained_feature",
                "audio_pretrained_feature_dict.npy",
            )
            if not os.path.exists(audio_path):
                raise FileNotFoundError(audio_path)
            self._audio_dict = np.load(audio_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.vids)

    def _ensure_visual_h5(self):
        if self._visual_h5 is None and self._visual_h5_path is not None:
            self._visual_h5 = h5py.File(self._visual_h5_path, "r")

    def _load_visual_feature(self, vid: str) -> torch.Tensor:
        if self.dataset_name == "AVE":
            feat = self._visual_dict[vid]
        else:
            self._ensure_visual_h5()
            feat = self._visual_h5[vid][()]
        return torch.tensor(feat, dtype=torch.float32)

    def _load_audio_feature(self, vid: str) -> torch.Tensor:
        feat = self._audio_dict[vid]
        return torch.tensor(feat, dtype=torch.float32)

    def __getitem__(self, index):
        vid = self.vids[index]
        y = self.labels[index]

        if self.modality == "visual":
            visual_feature = self._load_visual_feature(vid)
            return visual_feature, y

        if self.modality == "audio":
            audio_feature = self._load_audio_feature(vid)
            return audio_feature, y

        visual_feature = self._load_visual_feature(vid)
        audio_feature = self._load_audio_feature(vid)
        return (visual_feature, audio_feature), y   # return as tuple. In networks/approach, we should unpack accordingly.

    def close(self):
        if self._visual_h5 is not None:
            try:
                self._visual_h5.close()
            except Exception:
                pass
            self._visual_h5 = None


# ---------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------

def _get_specific(specific: Optional[Dict[str, Any]], key: str, default=None):
    if specific is None:
        return default
    return specific.get(key, default)


def parse_vggsound_csv(csv_path: str):
    """
    Expected row format:
        youtube_id, t, label, split

    example: --0PQM4-hqg,30,waterfall burbling,train
             ---g-f_I2yQ,1,people marching,test

    vid = f"{youtube_id}_{t:06d}"

    Returns
    -------
    rows : list of tuples
        [(vid, label_name, split), ...]
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 4:
                continue
            youtube_id = parts[0].strip()
            t = int(float(parts[1]))
            label = parts[2].strip()
            split = parts[3].strip().lower()
            vid = f"{youtube_id}_{t:06d}"
            rows.append((vid, label, split))
    return rows


def load_feature_vid_sets(feature_root: str, dataset_name: str):
    """
    Return available vid ids from visual/audio feature stores.
    vggsound csv may contain vids that do not have extracted features, so we need to check online and filter them out when building the pools.
    """
    visual_vids = None
    audio_vids = None

    if dataset_name == "AVE":
        visual_path = os.path.join(feature_root, "visual_pretrained_feature_dict.npy")
        if os.path.exists(visual_path):
            visual_dict = np.load(visual_path, allow_pickle=True).item()
            visual_vids = set(visual_dict.keys())
    else:
        visual_h5_path = os.path.join(feature_root, "visual_features.h5")
        if os.path.exists(visual_h5_path):
            with h5py.File(visual_h5_path, "r") as f:
                visual_vids = set(f.keys())

    audio_npy_path = os.path.join(feature_root, "audio_pretrained_feature", "audio_pretrained_feature_dict.npy")
    if os.path.exists(audio_npy_path):
        audio_dict = np.load(audio_npy_path, allow_pickle=True).item()
        audio_vids = set(audio_dict.keys())

    return visual_vids, audio_vids


def vid_ok_by_feature_requirement(
    vid: str,
    require_features: str,
    visual_vids: Optional[set],
    audio_vids: Optional[set],
):
    has_v = True if visual_vids is None else (vid in visual_vids)   # if visual_vids is None, it means we are not checking visual feature availability, so treat as all vids having visual features. Otherwise, check if vid is in visual_vids set.
    has_a = True if audio_vids is None else (vid in audio_vids)

    if require_features == "visual":
        return has_v
    if require_features == "audio":
        return has_a
    if require_features == "both":
        return has_v and has_a
    if require_features == "either":
        return has_v or has_a

    raise ValueError("require_features must be one of {'visual', 'audio', 'both', 'either'}")


def build_split_label_vids(
    rows: List[Tuple[str, str, str]],
    visual_vids: Optional[set],
    audio_vids: Optional[set],
    require_features: str,
):
    """
    Filter rows by feature availability online, then build:
        split_label_vids[split][label_name] = [vid1, vid2, ...]

    Actually the vggsound dataset doesn't have val split. While still keep the val split logic here:
    1. it's more flexible for other av datasets that do have val split
    2. this function still works, because we have this conditional statement "has_val_split = use_val_split and (len(split_label_vids["val"]) > 0)" 
    """
    split_label_vids = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for vid, label, split in rows:
        if split not in split_label_vids:
            continue
        if vid_ok_by_feature_requirement(vid, require_features, visual_vids, audio_vids):
            split_label_vids[split][label].append(vid)

    return split_label_vids


def choose_top_labels(
    split_label_vids,
    top_k,
    ranking,
    min_train_per_class,
    val_per_class,
    test_per_class,
    has_val_split,
):
    eligible_labels = []

    # keep the offline behavior: iterate in test key insertion order
    for lbl in split_label_vids["test"].keys():
        n_train = len(split_label_vids["train"].get(lbl, []))
        n_val = len(split_label_vids["val"].get(lbl, []))
        n_test = len(split_label_vids["test"].get(lbl, []))

        if n_test < test_per_class:
            continue
        if n_train < min_train_per_class:
            continue
        if has_val_split and n_val < val_per_class:
            continue

        eligible_labels.append(lbl)

    if len(eligible_labels) < (top_k or 0):
        raise RuntimeError(
            f"Only {len(eligible_labels)} eligible labels remain, cannot satisfy top_k={top_k}."
        )

    if ranking == "train":
        counter = Counter({
            lbl: len(split_label_vids["train"].get(lbl, []))
            for lbl in eligible_labels
        })
    elif ranking == "all":
        counter = Counter()
        for sp in ["train", "val", "test"]:
            for lbl in eligible_labels:
                counter[lbl] += len(split_label_vids[sp].get(lbl, []))
    else:
        raise ValueError("ranking must be one of {'all', 'train'}")

    ordered = [lbl for lbl, _ in counter.most_common()]
    if top_k is not None:
        ordered = ordered[:top_k]

    return ordered

def build_online_base_pools(
    csv_path: str,
    feature_root: str,
    dataset_name: str,
    require_features: str,
    top_k: Optional[int],
    ranking: str,
    val_per_class: int,
    test_per_class: int,
    min_train_per_class: int,
    class_order_seed: int,
    pool_shuffle_seed: int,
    use_val_split: bool,
):
    """
    Build the base pools online, analogous to build_base_pools(...) in the old offline script.

    Returns
    -------
    class_order_names : list[str]
        Fixed shuffled label-name order.

    val_dict : dict
        {"0": [vids...], "1": [vids...], ...}

    test_dict : dict
        {"0": [vids...], "1": [vids...], ...}

    train_pool_per_label : dict
        {label_name: [vids...]} ; already shuffled by pool_shuffle_seed

    avail_by_cid : list[int]
        Available train counts per cid after val carving (if needed)

    category_encode_dict : dict
        {label_name: cid}
    """
    rows = parse_vggsound_csv(csv_path)
    visual_vids, audio_vids = load_feature_vid_sets(feature_root, dataset_name)
    split_label_vids = build_split_label_vids(rows, visual_vids, audio_vids, require_features)

    has_val_split = use_val_split and (len(split_label_vids["val"]) > 0)

    top_labels = choose_top_labels(
        split_label_vids=split_label_vids,
        top_k=top_k,
        ranking=ranking,
        min_train_per_class=min_train_per_class,
        val_per_class=val_per_class if has_val_split else 0,
        test_per_class=test_per_class,
        has_val_split=has_val_split,
    )

    # fixed class order: shuffle labels ONCE
    rng_cls = random.Random(class_order_seed)
    class_order_names = top_labels[:]
    rng_cls.shuffle(class_order_names)

    category_encode_dict = {label_name: cid for cid, label_name in enumerate(class_order_names)}

    rng_pool = random.Random(pool_shuffle_seed)
    val_dict, test_dict, train_pool_per_label = {}, {}, {}

    for cid, label_name in enumerate(class_order_names):
        # TEST pool
        test_pool = split_label_vids["test"][label_name][:]
        if len(test_pool) < test_per_class:
            raise RuntimeError(
                f"Label '{label_name}' has only {len(test_pool)} test samples (<{test_per_class})."
            )
        rng_pool.shuffle(test_pool)
        test_dict[str(cid)] = test_pool[:test_per_class]

        # VAL / TRAIN pools
        if has_val_split:
            val_pool = split_label_vids["val"][label_name][:]
            if len(val_pool) < val_per_class:
                raise RuntimeError(
                    f"Label '{label_name}' has only {len(val_pool)} val samples (<{val_per_class})."
                )
            rng_pool.shuffle(val_pool)
            val_dict[str(cid)] = val_pool[:val_per_class]

            train_pool = split_label_vids["train"][label_name][:]
            if len(train_pool) < 1:
                raise RuntimeError(f"Label '{label_name}' has 0 train samples.")
            rng_pool.shuffle(train_pool)
            train_pool_per_label[label_name] = train_pool
        else:
            # no explicit val split: carve val from train here, consistent with original AV-CIL logic
            train_full = split_label_vids["train"][label_name][:]
            if len(train_full) < val_per_class + 1:
                raise RuntimeError(
                    f"Label '{label_name}' has only {len(train_full)} train samples; "
                    f"cannot carve val={val_per_class} and still keep >=1 for train."
                )
            rng_pool.shuffle(train_full)
            val_dict[str(cid)] = train_full[:val_per_class]
            train_rem = train_full[val_per_class:]
            if len(train_rem) < 1:
                raise RuntimeError(
                    f"Label '{label_name}' has 0 remaining train samples after carving val."
                )
            train_pool_per_label[label_name] = train_rem

    avail_by_cid = [len(train_pool_per_label[class_order_names[cid]]) for cid in range(len(class_order_names))]
    if min(avail_by_cid) == 0:
        raise RuntimeError("Some classes have 0 remaining train samples after pooling (unexpected).")

    return class_order_names, val_dict, test_dict, train_pool_per_label, avail_by_cid, category_encode_dict


def lt_targets(C, Nmax, gamma=10.0):
    """
    Long-tail targets:
        N_i = floor(Nmax * gamma^{-i/(C-1)})
    i=0 head, i=C-1 tail
    """
    out = []
    for i in range(C):
        val = Nmax * (gamma ** (-i / (C - 1)))
        out.append(max(1, int(np.floor(val))))
    return out


def adjust_to_total_with_caps(targets, caps, total, rng):
    """
    Adjust integer targets so that:
        - sum(targets) == total
        - 1 <= targets[i] <= caps[i]
    Best-effort if exact adjustment is impossible.
    """
    targets = [max(1, min(int(t), int(c))) for t, c in zip(targets, caps)]
    cur = sum(targets)
    if cur == total:
        return targets

    def rebuild_inc_dec():
        inc = [i for i in range(len(targets)) if targets[i] < caps[i]]
        dec = [i for i in range(len(targets)) if targets[i] > 1]
        rng.shuffle(inc)
        rng.shuffle(dec)
        return inc, dec

    inc, dec = rebuild_inc_dec()

    if cur < total:
        need = total - cur
        j = 0
        while need > 0:
            if not inc:
                break
            i = inc[j % len(inc)]
            if targets[i] < caps[i]:
                targets[i] += 1
                need -= 1
            else:
                inc, dec = rebuild_inc_dec()
                j = 0
                continue
            j += 1
            if j % 10000 == 0:
                inc, dec = rebuild_inc_dec()
    else:
        need = cur - total
        j = 0
        while need > 0:
            if not dec:
                break
            i = dec[j % len(dec)]
            if targets[i] > 1:
                targets[i] -= 1
                need -= 1
            else:
                inc, dec = rebuild_inc_dec()
                j = 0
                continue
            j += 1
            if j % 10000 == 0:
                inc, dec = rebuild_inc_dec()

    return targets


def make_targets_lt_and_bal(avail_by_cid, gamma=10.0, mean_cap=512, seed=42, nmax_policy="min"):
    """
    Generate:
        - lt_t  : long-tail targets
        - bal_t : balanced targets
    """
    rng = random.Random(seed)
    C = len(avail_by_cid)

    if nmax_policy == "cid_last":
        Nmax = avail_by_cid[C - 1]
    elif nmax_policy == "min":
        Nmax = min(avail_by_cid)
    elif nmax_policy == "max":
        Nmax = max(avail_by_cid)
    else:
        raise ValueError("nmax_policy must be one of: cid_last/min/max")

    lt_t = lt_targets(C, Nmax, gamma=gamma)
    lt_t = [min(lt_t[i], avail_by_cid[i]) for i in range(C)]
    lt_mean = sum(lt_t) / C

    bal_per_class = int(round(lt_mean))
    bal_t = [min(bal_per_class, avail_by_cid[i]) for i in range(C)]
    bal_mean = sum(bal_t) / C

    if bal_mean > mean_cap:
        scale = mean_cap / bal_mean
        lt_scaled = [max(1, int(round(x * scale))) for x in lt_t]
        bal_scaled = [max(1, int(round(x * scale))) for x in bal_t]
        total_target = mean_cap * C

        lt_t = adjust_to_total_with_caps(lt_scaled, caps=avail_by_cid, total=total_target, rng=rng)
        bal_t = adjust_to_total_with_caps(bal_scaled, caps=avail_by_cid, total=total_target, rng=rng)
    else:
        lt_t = [max(1, min(lt_t[i], avail_by_cid[i])) for i in range(C)]
        bal_t = [max(1, min(bal_t[i], avail_by_cid[i])) for i in range(C)]

    return lt_t, bal_t


def permute_targets(targets, seed):
    rng = random.Random(seed)
    idx = list(range(len(targets)))
    rng.shuffle(idx)
    return [targets[i] for i in idx]


def sample_train_dict_from_targets(train_pool_per_label, class_order_names, targets_by_cid):
    """
    Build train dict from already-shuffled train pools.
    """
    C = len(class_order_names)
    out = {}
    for cid in range(C):
        lbl = class_order_names[cid]
        vids = train_pool_per_label[lbl]
        k = min(int(targets_by_cid[cid]), len(vids))
        out[str(cid)] = vids[:k]
    return out


def build_train_dict_by_mode(
    train_pool_per_label,
    class_order_names,
    avail_by_cid,
    train_mode="base",
    gamma=100.0,
    mean_cap=512,
    nmax_policy="min",
    random_seed=101,
    balance_max_per_class=578,
):
    """
    Build the final training dict according to train_mode.

    train_mode:
        - "base"
        - "balance"
        - "ordered"
        - "reversed"
        - "random"
        - "balance_max"
    """
    C = len(class_order_names)

    if train_mode == "base":
        targets = avail_by_cid

    elif train_mode in {"balance", "ordered", "reversed", "random"}:
        lt_t, bal_t = make_targets_lt_and_bal(
            avail_by_cid=avail_by_cid,
            gamma=gamma,
            mean_cap=mean_cap,
            seed=42,
            nmax_policy=nmax_policy,
        )

        if train_mode == "balance":
            targets = bal_t
        elif train_mode == "ordered":
            targets = lt_t
        elif train_mode == "reversed":
            targets = list(reversed(lt_t))
        elif train_mode == "random":
            targets = permute_targets(lt_t, random_seed)

    elif train_mode == "balance_max":
        targets = [min(balance_max_per_class, avail_by_cid[i]) for i in range(C)]

    else:
        raise ValueError(
            "train_mode must be one of: "
            "base / balance / ordered / reversed / random / balance_max"
        )

    train_dict = sample_train_dict_from_targets(
        train_pool_per_label=train_pool_per_label,
        class_order_names=class_order_names,
        targets_by_cid=targets,
    )
    return train_dict


def build_online_triplet(
    csv_path: str,
    feature_root: str,
    dataset_name: str,
    require_features: str,
    top_k: Optional[int],
    ranking: str,
    val_per_class: int,
    test_per_class: int,
    min_train_per_class: int,
    class_order_seed: int,
    pool_shuffle_seed: int,
    use_val_split: bool,
    train_mode: str = "base",
    gamma: float = 100.0,
    mean_cap: int = 512,
    nmax_policy: str = "min",
    random_seed: int = 101,
    balance_max_per_class: int = 578,
):
    """
    Online replacement for the old offline triplet generation, now supporting:
        - base
        - balance
        - ordered
        - reversed
        - random
        - balance_max
    """
    class_order_names, val_dict, test_dict, train_pool_per_label, avail_by_cid, category_encode_dict = \
        build_online_base_pools(
            csv_path=csv_path,
            feature_root=feature_root,
            dataset_name=dataset_name,
            require_features=require_features,
            top_k=top_k,
            ranking=ranking,
            val_per_class=val_per_class,
            test_per_class=test_per_class,
            min_train_per_class=min_train_per_class,
            class_order_seed=class_order_seed,
            pool_shuffle_seed=pool_shuffle_seed,
            use_val_split=use_val_split,
        )

    train_dict = build_train_dict_by_mode(
        train_pool_per_label=train_pool_per_label,
        class_order_names=class_order_names,
        avail_by_cid=avail_by_cid,
        train_mode=train_mode,
        gamma=gamma,
        mean_cap=mean_cap,
        nmax_policy=nmax_policy,
        random_seed=random_seed,
        balance_max_per_class=balance_max_per_class,
    )

    all_classId_vid_dict = {
        "train": train_dict,
        "val": val_dict,
        "test": test_dict,
    }

    all_id_category_dict = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        for cid_str, vids in all_classId_vid_dict[split].items():
            cid = int(cid_str)
            label_name = class_order_names[cid]
            for vid in vids:
                all_id_category_dict[split][vid] = label_name

    class_order_ids = list(range(len(class_order_names)))

    return all_classId_vid_dict, category_encode_dict, all_id_category_dict, class_order_names, class_order_ids


def compute_cpertask(num_classes: int, num_tasks: int, nc_first_task: Optional[int]):
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1
    return cpertask


def split_train_to_val_per_class(x, y, ncla, val_per_class):
    """
    Split train -> train + val with fixed val_per_class for each class.
    No reshuffle here, because train pools are already shuffled in build_online_base_pools.
    """
    trn_x, trn_y = list(x), list(y)
    val_x, val_y = [], []

    for cc in range(ncla):
        cls_idx = [i for i, yy in enumerate(trn_y) if yy == cc]
        if len(cls_idx) < val_per_class:
            raise RuntimeError(
                f"class {cc} has only {len(cls_idx)} samples, cannot carve val_per_class={val_per_class}"
            )

        picked = cls_idx[:val_per_class]
        picked.sort(reverse=True)

        for idx in picked:
            val_x.append(trn_x[idx])
            val_y.append(trn_y[idx])
            trn_x.pop(idx)
            trn_y.pop(idx)

    return trn_x, trn_y, val_x, val_y


def get_data(
    csv_path: str,
    feature_root: str,
    num_tasks: int,
    nc_first_task: Optional[int],
    validation: float,
    shuffle_classes=False,
    class_order: Optional[List[int]] = None,
    specific: Optional[Dict[str, Any]] = None,
):
    """
    Online multimodal data construction for FACIL-style incremental learning.
    The final data would be like:
    data = {
    0: {
        "name": "task-0",
        "trn": {"x": [vids...], "y": [labels...]},
        "val": {"x": [vids...], "y": [labels...]},
        "tst": {"x": [vids...], "y": [labels...]]},
        "ncla": num_classes_in_task_0
    },
    """
    specific = specific or {}

    dataset_name = _get_specific(specific, "dataset_name", "VGGSound")
    require_features = _get_specific(specific, "require_features", "both")
    top_k = _get_specific(specific, "top_k", 100)
    ranking = _get_specific(specific, "ranking", "all")
    val_per_class = _get_specific(specific, "val_per_class", 50)
    test_per_class = _get_specific(specific, "test_per_class", 50)
    min_train_per_class = _get_specific(specific, "min_train_per_class", 1)
    class_order_seed = _get_specific(specific, "class_order_seed", 42)
    pool_shuffle_seed = _get_specific(specific, "pool_shuffle_seed", 42)
    use_val_split = _get_specific(specific, "use_val_split", True)
    validation_from_train = _get_specific(specific, "validation_from_train", False)

    # new fields for dataset variant control
    train_mode = _get_specific(specific, "train_mode", "base")
    gamma = _get_specific(specific, "gamma", 100.0)
    mean_cap = _get_specific(specific, "mean_cap", 512)
    nmax_policy = _get_specific(specific, "nmax_policy", "min")
    random_seed = _get_specific(specific, "random_seed", 101)
    balance_max_per_class = _get_specific(specific, "balance_max_per_class", 578)

    all_classId_vid_dict, category_encode_dict, all_id_category_dict, class_order_names, built_class_ids = build_online_triplet(
        csv_path=csv_path,
        feature_root=feature_root,
        dataset_name=dataset_name,
        require_features=require_features,
        top_k=top_k,
        ranking=ranking,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
        min_train_per_class=min_train_per_class,
        class_order_seed=class_order_seed,
        pool_shuffle_seed=pool_shuffle_seed,
        use_val_split=use_val_split,
        train_mode=train_mode,
        gamma=gamma,
        mean_cap=mean_cap,
        nmax_policy=nmax_policy,
        random_seed=random_seed,
        balance_max_per_class=balance_max_per_class,
    )

    # final class order over the online-built ids
    if class_order is None:
        class_order = built_class_ids[:]
        # build_online_triplet() has already fixed class order with class_order_seed
        if shuffle_classes:
            rng = random.Random(class_order_seed)
            rng.shuffle(class_order)
    else:
        class_order = [int(cid) for cid in class_order if int(cid) in set(built_class_ids)]

    num_classes = len(class_order)
    assert num_classes > 0, "No classes in final class_order"

    cpertask = compute_cpertask(num_classes, num_tasks, nc_first_task)
    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"

    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    data = {}
    for tt in range(num_tasks):
        data[tt] = {
            "name": f"task-{tt}",
            "trn": {"x": [], "y": []},
            "val": {"x": [], "y": []},
            "tst": {"x": [], "y": []},
        }

    def append_split(split_name: str, out_key: str):
        for ordered_idx, cid in enumerate(class_order):
            vids = all_classId_vid_dict[split_name][str(cid)]
            this_task = (ordered_idx >= cpertask_cumsum).sum()
            local_y = ordered_idx - init_class[this_task]

            data[this_task][out_key]["x"].extend(vids)
            data[this_task][out_key]["y"].extend([local_y] * len(vids))

    append_split("train", "trn")
    append_split("test", "tst")

    has_online_val = any(len(v) > 0 for v in all_classId_vid_dict["val"].values())
    if has_online_val and not validation_from_train:
        append_split("val", "val")

    # check train classes
    for tt in range(num_tasks):
        present = sorted(set(data[tt]["trn"]["y"]))
        expected = list(range(cpertask[tt]))
        if present != expected:
            raise AssertionError(
                f"Task {tt} training classes mismatch. expected={expected}, present={present}"
            )
        data[tt]["ncla"] = len(present)

    # Only carve val from train if explicitly required and there is no online val.
    # In AV-CIL-consistent use, val is usually already constructed in build_online_base_pools.
    if ((not has_online_val) or validation_from_train) and val_per_class > 0:
        for tt in range(num_tasks):
            trn_x = data[tt]["trn"]["x"]
            trn_y = data[tt]["trn"]["y"]

            new_trn_x, new_trn_y, val_x, val_y = split_train_to_val_per_class(
                trn_x,
                trn_y,
                ncla=data[tt]["ncla"],
                val_per_class=val_per_class,
            )

            data[tt]["trn"]["x"] = new_trn_x
            data[tt]["trn"]["y"] = new_trn_y
            data[tt]["val"]["x"] = val_x
            data[tt]["val"]["y"] = val_y

    # taskcla: [(task_id, num_classes_in_task), ...]
    taskcla = []
    n_total = 0
    for tt in range(num_tasks):
        taskcla.append((tt, data[tt]["ncla"]))
        n_total += data[tt]["ncla"]
    data["ncla"] = n_total

    return data, taskcla, class_order







