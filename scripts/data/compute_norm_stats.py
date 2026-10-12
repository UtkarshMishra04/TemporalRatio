"""Compute state / action normalization statistics for a named dataset config.

The statistics (mean, std, 1st / 99th percentiles) are written to
``assets/<config name>/<primary repo id>/norm_stats.json``, where the trainers read them.

Usage:
    # Fast: reads the parquet files directly (no video decoding)
    python scripts/data/compute_norm_stats.py --config-name videoVLA_abc130k_pp24_short --fast

    # Through the training data pipeline (decodes frames; --max-frames bounds the cost)
    python scripts/data/compute_norm_stats.py --config-name videoVLA_libero_short --max-frames 50000
"""

import os

import numpy as np
import tqdm
import tyro

import temporal_ratio.models.base as _model
import temporal_ratio.data.normalize as normalize
import temporal_ratio.config.registry as _config
import temporal_ratio.data.action_loader as _data_loader
import temporal_ratio.data.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    observation_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, observation_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main_fast(config_name: str, samples_per_task: int = 100):
    """Fast norm stats by reading parquet files directly (no video decode)."""
    import pathlib
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    repo_ids = [data_config.repo_id] + list(getattr(data_config, "extra_repo_ids", ()))
    print(f"Computing fast norm stats for {len(repo_ids)} repos, {samples_per_task} episodes each")

    # Detect state/action column names from first parquet file
    lerobot_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))
    first_repo_path = lerobot_home / repo_ids[0]
    first_parquet = sorted((first_repo_path / "data").rglob("*.parquet"))[0]
    schema_names = pq.read_schema(str(first_parquet)).names
    state_col = next((c for c in schema_names if "state" in c.lower()), None)
    action_col = next((c for c in schema_names if c in ("action", "actions")), None)
    print(f"Detected columns: state={state_col}, action={action_col}")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    def _process_repo(repo_id):
        dataset_path = lerobot_home / repo_id
        parquet_files = sorted((dataset_path / "data").rglob("*.parquet"))
        if not parquet_files:
            print(f"  [WARN] No parquet files in {dataset_path}")
            return None, None
        # Sample evenly across episodes
        indices = np.linspace(0, len(parquet_files) - 1, min(samples_per_task, len(parquet_files)), dtype=int)
        repo_states, repo_actions = [], []
        for idx in indices:
            table = pq.read_table(str(parquet_files[idx]), columns=[state_col, action_col])
            s = np.stack(table[state_col].to_pylist())
            a = np.stack(table[action_col].to_pylist())
            repo_states.append(s)
            repo_actions.append(a)
        return np.concatenate(repo_states), np.concatenate(repo_actions)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(tqdm.tqdm(pool.map(_process_repo, repo_ids), total=len(repo_ids), desc="Reading parquet"))

    for s, a in results:
        if s is not None:
            stats["state"].update(s)
            stats["actions"].update(a)

    norm_stats = {key: st.get_statistics() for key, st in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


def main(config_name: str, max_frames: int | None = None, fast: bool = False, samples_per_task: int = 100):
    if fast:
        return main_fast(config_name, samples_per_task=samples_per_task)

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.model.observation_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
