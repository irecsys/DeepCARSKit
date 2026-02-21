import pandas as pd
from pathlib import Path
from typing import List, Dict
from deepcarskit.utils import logger


def parse_fold_list(config) -> List[int]:
    """
    check fold list from config, if CV exists, return fold list according to CV, otherwise return [0] for hold-out
    """
    split_cfg = config.get('eval_args', {}).get('split', {})
    if 'CV' in split_cfg:
        return list(range(1, int(split_cfg['CV']) + 1))
    else:  # hold-out
        return [0]

def aggregate_one_csv(csv_path: Path, user_col='uid', context_col='ucid') -> Dict[str, float]:
    """
    calculation for one fold, based on CARSKit logic:
    average over contexts (UC) -> average over users -> global average
    """

    df = pd.read_csv(csv_path, sep=None, engine='python')

    id_cols = {user_col, 'cid', context_col}
    metric_cols = [c for c in df.columns if c not in id_cols]

    if not metric_cols:
        raise ValueError(f"No metric columns found in {csv_path}")

    # Step 1: UC avg
    uc_level = df.groupby([user_col, context_col])[metric_cols].mean()

    # Step 2: user avg
    user_level = uc_level.groupby(user_col).mean()

    # Step 3: fold avg
    return user_level.mean().to_dict()


def aggregate_folds(csv_files: List[Path], user_col='uid', context_col='ucid') -> Dict[str, float]:
    """
    aggregation from multiple folds
    """
    fold_results = []
    for csv_file in csv_files:
        fold_res = aggregate_one_csv(csv_file, user_col, context_col)
        fold_results.append(fold_res)

    df = pd.DataFrame(fold_results)
    return df.mean().to_dict()


def output_metrics_to_carskit(config: dict, logger):
    """
    Transform DeepCARSKit per-user metrics CSVs into aggregated CARSKit-style metrics.
    Computes averages across folds and optionally prints and saves results.

    Args:
        config (dict): DeepCARSKit config dictionary.
        logger (logging.Logger): Logger instance for info/warning messages.
    """
    # Check if per-user metrics are saved
    if not config.get('save_per_uc_metrics', False):
        logger.warning("save_per_uc_metrics=False, skipping output.")
        return

    dataset = config['dataset']
    algorithm = config['model']
    save_folder = Path(config.get('save_per_uc_folder', 'saved'))
    save_folder.mkdir(parents=True, exist_ok=True)

    # Determine fold list based on CV or hold-out split
    folds = parse_fold_list(config)  # returns list of ints, e.g., [1,2,3,4,5] or [0]

    # Collect CSV files for each fold
    csv_files = []
    for f in folds:
        fname = f"{dataset}_{algorithm}_fold{f}.csv"
        fpath = save_folder / fname
        if not fpath.exists():
            logger.warning(f"Missing CSV file: {fpath}")
            continue
        csv_files.append(fpath)

    if not csv_files:
        logger.error("No CSV files found. Cannot compute metrics.")
        return

    # Aggregate metrics across folds
    final_metrics = aggregate_folds(csv_files)

    # Compute F1@k from precision@k and recall@k
    topk_list = config.get('topk', [10, 20, 30])
    for k in topk_list:
        prec_key = f"precision@{k}"
        rec_key = f"recall@{k}"
        f1_key = f"f1@{k}"
        if prec_key in final_metrics and rec_key in final_metrics:
            p = final_metrics[prec_key]
            r = final_metrics[rec_key]
            if p + r > 0:
                final_metrics[f1_key] = 2 * p * r / (p + r)
            else:
                final_metrics[f1_key] = 0.0

    # Format decimal places
    metric_decimal_place = config.get('metric_decimal_place', 4)
    formatted_metrics = {k: round(v, metric_decimal_place) for k, v in final_metrics.items()}

    # Determine fold info for log
    if 'CV' in config.get('eval_args', {}).get('split', {}):
        nfold = config['eval_args']['split']['CV']
        fold_info = f"{nfold} CV"
    else:
        fold_info = "hold-out split"

    # Log the final metrics in CARSKit style
    # ANSI codes
    BLUE = "\033[94m"
    RESET = "\033[0m"

    logger.info(
        f"{BLUE}Data: {dataset}, Results on {fold_info} (CARSKit): best valid by {algorithm}:{RESET} {formatted_metrics}"
    )

