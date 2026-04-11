#########################
# reconstruction of triplet npy files to varify the correctness of the online triplet construction, and also to support downstream usage that relies on the triplet npy files.
#########################
import os
import argparse
import numpy as np

from dataset_config import dataset_config
from av_dataset import get_data, build_online_triplet

# the output of get_data() should like this:
# data = {
#     0: {
#         "name": "task-0",
#         "trn": {"x": [...], "y": [...]},
#         "val": {"x": [...], "y": [...]},
#         "tst": {"x": [...], "y": [...]},
#     },
#     1: {
#         "name": "task-1",
#         "trn": {"x": [...], "y": [...]},
#         "val": {"x": [...], "y": [...]},
#         "tst": {"x": [...], "y": [...]},
#     },
#     ...
# }

def reconstruct_triplet_from_task_data(data, taskcla, class_order, category_encode_dict):
    """
    Reconstruct:
        - all_classId_vid_dict
        - all_id_category_dict
        - category_encode_dict
    from:
        - data
        - taskcla
        - class_order
        - category_encode_dict

    Parameters
    ----------
    data : dict
        Returned by get_data(...)

    taskcla : list[(task_id, ncla)]
        Returned by get_data(...)

    class_order : list[int]
        Returned by get_data(...)

    category_encode_dict : dict
        {label_name: cid}

    Returns
    -------
    all_classId_vid_dict : dict
        {
            "train": {"0": [...], "1": [...], ...},
            "val":   {"0": [...], "1": [...], ...},
            "test":  {"0": [...], "1": [...], ...},
        }

    all_id_category_dict : dict
        {
            "train": {vid: label_name, ...},
            "val":   {vid: label_name, ...},
            "test":  {vid: label_name, ...},
        }

    category_encode_dict : dict
        unchanged
    """
    cid_to_label = {int(cid): label for label, cid in category_encode_dict.items()}

    num_classes = len(class_order)
    all_classId_vid_dict = {
        "train": {str(cid): [] for cid in range(num_classes)},
        "val":   {str(cid): [] for cid in range(num_classes)},
        "test":  {str(cid): [] for cid in range(num_classes)},
    }
    all_id_category_dict = {
        "train": {},
        "val": {},
        "test": {},
    }

    # task-local label -> global ordered class index -> cid
    offset = 0
    for task_id, ncla in taskcla:
        task_data = data[task_id]

        for split_key, split_name in [("trn", "train"), ("val", "val"), ("tst", "test")]:
            xs = task_data[split_key]["x"]
            ys = task_data[split_key]["y"]

            for vid, local_y in zip(xs, ys):
                global_ordered_idx = offset + int(local_y)
                cid = int(class_order[global_ordered_idx])

                all_classId_vid_dict[split_name][str(cid)].append(vid)

                label_name = cid_to_label[cid]
                if vid in all_id_category_dict[split_name]:
                    prev = all_id_category_dict[split_name][vid]
                    if prev != label_name:
                        raise RuntimeError(
                            f"[Inconsistent] split={split_name}, vid={vid}, "
                            f"previous={prev}, new={label_name}"
                        )
                all_id_category_dict[split_name][vid] = label_name

        offset += ncla

    return all_classId_vid_dict, all_id_category_dict, category_encode_dict


def save_triplet(out_dir, all_classId_vid_dict, all_id_category_dict, category_encode_dict):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "all_classId_vid_dict.npy"), all_classId_vid_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "all_id_category_dict.npy"), all_id_category_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "category_encode_dict.npy"), category_encode_dict, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser("Rebuild triplet from get_data outputs")
    parser.add_argument("--dataset", type=str, required=True, help="dataset key in dataset_config")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save reconstructed npy files")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--nc_first_task", type=int, default=None)
    parser.add_argument("--validation", type=float, default=0.0)
    parser.add_argument("--shuffle_classes", action="store_true")
    parser.add_argument("--print_summary", action="store_true")
    args = parser.parse_args()

    if args.dataset not in dataset_config:
        raise KeyError(f"{args.dataset} not found in dataset_config")

    dc = dataset_config[args.dataset]

    csv_path = dc["path"]
    feature_root = dc["feature_root"]
    specific = dc.get("specific", {})
    class_order_cfg = dc.get("class_order", None)

    # 1) build FACIL-style outputs
    data, taskcla, class_order = get_data(
        csv_path=csv_path,
        feature_root=feature_root,
        num_tasks=args.num_tasks,
        nc_first_task=args.nc_first_task,
        validation=args.validation,
        shuffle_classes=args.shuffle_classes,
        class_order=class_order_cfg,
        specific=specific,
    )

    # 2) build online triplet again ONLY to recover category_encode_dict
    #    (data/taskcla/class_order do not contain label names)
    dataset_name = specific.get("dataset_name", "VGGSound")
    require_features = specific.get("require_features", "both")
    top_k = specific.get("top_k", 100)
    ranking = specific.get("ranking", "all")
    val_per_class = specific.get("val_per_class", 50)
    test_per_class = specific.get("test_per_class", 50)
    min_train_per_class = specific.get("min_train_per_class", 1)
    class_order_seed = specific.get("class_order_seed", 42)
    pool_shuffle_seed = specific.get("pool_shuffle_seed", 42)
    use_val_split = specific.get("use_val_split", True)

    train_mode = specific.get("train_mode", "base")
    gamma = specific.get("gamma", 100.0)
    mean_cap = specific.get("mean_cap", 512)
    nmax_policy = specific.get("nmax_policy", "min")
    random_seed = specific.get("random_seed", 101)
    balance_max_per_class = specific.get("balance_max_per_class", 578)

    _, category_encode_dict, _, _, _ = build_online_triplet(
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

    # 3) reconstruct triplet from data/taskcla/class_order/category_encode_dict
    all_classId_vid_dict, all_id_category_dict, category_encode_dict = reconstruct_triplet_from_task_data(
        data=data,
        taskcla=taskcla,
        class_order=class_order,
        category_encode_dict=category_encode_dict,
    )

    # 4) save
    save_triplet(
        out_dir=args.out_dir,
        all_classId_vid_dict=all_classId_vid_dict,
        all_id_category_dict=all_id_category_dict,
        category_encode_dict=category_encode_dict,
    )

    print(f"[OK] Saved to: {args.out_dir}")

    if args.print_summary:
        print(f"taskcla = {taskcla}")
        print(f"class_order = {class_order}")
        for split in ["train", "val", "test"]:
            counts = [len(all_classId_vid_dict[split][str(i)]) for i in range(len(class_order))]
            if len(counts) > 0:
                print(
                    f"{split}: "
                    f"min={min(counts)}, mean={sum(counts)/len(counts):.2f}, max={max(counts)}"
                )


if __name__ == "__main__":
    main()