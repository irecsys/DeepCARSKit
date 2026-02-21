import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List

# ----------------- arguments -----------------
FOLDER = "saved"            # CSV folder
DATASET = "depaulmovie"       # dataset name
ALGORITHM = "NeuCMF0w"        # algorithm name
FOLDS = "1-5"                 # fold range, e.g, "1-5" => 5-folds cross validation
# ----------------------------------------------------

def parse_fold_range(fold_str: str) -> List[int]:
    """
    Parse fold range string like '1-5' or '3-10'
    """
    if "-" not in fold_str:
        return [int(fold_str)]
    start, end = fold_str.split("-")
    return list(range(int(start), int(end) + 1))


def aggregate_one_csv(csv_path: Path) -> Dict[str, float]:
    """
    Aggregate one fold CSV following CARSKit logic:
    UC -> user mean -> global mean
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")

    id_cols = {"uid", "cid", "ucid"}
    metric_cols = [c for c in df.columns if c not in id_cols]

    if not metric_cols:
        raise ValueError(f"No metric columns found in {csv_path.name}")

    # ---- Step 1: per-user (average over contexts) ----
    user_level = (
        df
        .groupby("uid")[metric_cols]
        .mean()
    )

    # ---- Step 2: global (average over users) ----
    global_metrics = user_level.mean().to_dict()

    return global_metrics


def aggregate_folds(csv_files: List[Path]) -> Dict[str, float]:
    """
    Aggregate multiple folds by averaging fold-level results
    """
    fold_results = []

    for csv in csv_files:
        print(f"Processing {csv.name}")
        res = aggregate_one_csv(csv)
        fold_results.append(res)

    df = pd.DataFrame(fold_results)
    return df.mean().to_dict()


def compute_f1(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate F1@X
    """
    f1_metrics = {}
    precisions = {k: v for k, v in metrics.items() if k.startswith("precision@")}
    recalls = {k: v for k, v in metrics.items() if k.startswith("recall@")}

    for k in precisions:
        suffix = k.split("@")[1]
        recall_key = f"recall@{suffix}"
        if recall_key in recalls:
            p = precisions[k]
            r = recalls[recall_key]
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
            f1_metrics[f"F1@{suffix}"] = f1

    return f1_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-UC CSV results following CARSKit evaluation logic"
    )

    parser.add_argument("--folder", type=str, help="Folder containing CSV result files")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. depaulmovie)")
    parser.add_argument("--algorithm", type=str, help="Algorithm name (e.g. NeuCMF0w)")
    parser.add_argument("--folds", type=str, help="Fold range, e.g. 1-5 or 3-10")

    args = parser.parse_args()

    # Use the arguments on the top, if not provided via command line
    folder = Path(args.folder) if args.folder else Path(FOLDER)
    dataset = args.dataset if args.dataset else DATASET
    algorithm = args.algorithm if args.algorithm else ALGORITHM
    folds_str = args.folds if args.folds else FOLDS

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    folds = parse_fold_range(folds_str)

    csv_files = []
    for f in folds:
        fname = f"{dataset}_{algorithm}_fold{f}.csv"
        fpath = folder / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fname}")
        csv_files.append(fpath)

    print(f"\nFound {len(csv_files)} CSV files")
    print("-" * 50)

    final_metrics = aggregate_folds(csv_files)

    f1_metrics = compute_f1(final_metrics)
    final_metrics.update(f1_metrics)

    print("\n=== Final CARSKit-style Results ===")
    for k, v in sorted(final_metrics.items()):
        print(f"{k:15s}: {v:.6f}")


if __name__ == "__main__":
    main()
