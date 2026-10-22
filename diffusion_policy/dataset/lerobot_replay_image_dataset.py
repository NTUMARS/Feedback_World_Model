from typing import Dict, List, Optional
import torch
import numpy as np
import os
import copy
import json
import shutil
import zarr
import cv2
import av
import pyarrow.parquet as pq
from tqdm import tqdm
from filelock import FileLock
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    array_to_stats
)


class LeRobotReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        feature_map: dict,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        image_resize: Optional[List[int]] = None,
    ):
        if image_resize is None:
            image_resize = [224, 224]

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path.rstrip('/') + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _convert_lerobot_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            feature_map=feature_map,
                            image_resize=image_resize,
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_lerobot_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                feature_map=feature_map,
                image_resize=image_resize,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print('episode ends ', replay_buffer.episode_ends[:])
        print('action ', replay_buffer['action'].shape)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action: range normalizer for raw joint positions/gripper values
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_range_normalizer_from_stat(stat)

        # low-dim obs: range normalizer for all
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        # image obs
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # T,H,W,C -> T,C,H,W, uint8 -> float32 [0,1]
            obs_dict[key] = np.moveaxis(
                data[key][T_slice], -1, 1
            ).astype(np.float32) / 255.
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad shorter edge with zeros to make image square."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
    return padded


def _decode_video_episode(
    video_path: str,
    target_size: List[int],
) -> np.ndarray:
    """Decode all frames from an MP4 video, pad to square, resize."""
    with av.open(str(video_path)) as container:
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')  # H, W, 3
            img = _pad_to_square(img)
            img = cv2.resize(
                img,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(img)
    return np.stack(frames)  # (T, H, W, 3) uint8


def _read_parquet_columns(
    table: pq.lib.Table,
    columns: List[str],
) -> np.ndarray:
    """Extract and concatenate columns from a parquet table into a float32 array."""
    arrays = []
    for col_name in columns:
        col_data = table.column(col_name).to_pylist()
        arr = np.array(col_data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def _convert_lerobot_to_replay(
    store,
    shape_meta: dict,
    dataset_path: str,
    feature_map: dict,
    image_resize: List[int],
) -> ReplayBuffer:
    """Convert LeRobot v2.1 dataset to ReplayBuffer."""
    # Read metadata
    info_path = os.path.join(dataset_path, 'meta', 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)

    episodes_path = os.path.join(dataset_path, 'meta', 'episodes.jsonl')
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    n_episodes = len(episodes)
    data_path_template = info.get(
        'data_path',
        'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
    )

    # Parse shape_meta for keys
    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        if obs_type == 'rgb':
            rgb_keys.append(key)
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)

    if not episodes:
        raise ValueError(
            f"No episodes found in {episodes_path}. "
            "Check that the dataset path and episodes.jsonl are valid.")

    # Compute episode ends and total steps
    episode_ends = []
    cumulative = 0
    for ep in episodes:
        cumulative += ep['length']
        episode_ends.append(cumulative)
    n_steps = episode_ends[-1]

    # Create zarr structure
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    meta_group.array(
        'episode_ends', episode_ends,
        dtype=np.int64, compressor=None, overwrite=True)

    # Collect low-dim data per episode
    lowdim_data = {key: [] for key in lowdim_keys}
    action_data = []

    action_columns = feature_map['action']['columns']

    print(f'Loading {n_episodes} episodes from LeRobot dataset...')
    for ep_idx in tqdm(range(n_episodes), desc='Loading parquet data'):
        ep = episodes[ep_idx]
        chunk_idx = ep_idx // info.get('chunks_size', 1000)
        parquet_path = os.path.join(
            dataset_path,
            data_path_template.format(
                episode_chunk=chunk_idx, episode_index=ep_idx))
        table = pq.read_table(parquet_path)

        # Extract actions
        action_arr = _read_parquet_columns(table, action_columns)
        action_data.append(action_arr)

        # Extract low-dim observations
        for key in lowdim_keys:
            obs_columns = feature_map['obs'][key]['columns']
            obs_arr = _read_parquet_columns(table, obs_columns)
            lowdim_data[key].append(obs_arr)

    # Write action data
    all_actions = np.concatenate(action_data, axis=0)
    assert all_actions.shape == (n_steps, shape_meta['action']['shape'][0]), \
        f"Action shape mismatch: {all_actions.shape} vs expected ({n_steps}, {shape_meta['action']['shape'][0]})"
    data_group.array(
        name='action',
        data=all_actions,
        shape=all_actions.shape,
        chunks=all_actions.shape,
        compressor=None,
        dtype=all_actions.dtype)

    # Write low-dim obs data
    for key in lowdim_keys:
        all_obs = np.concatenate(lowdim_data[key], axis=0)
        expected_shape = (n_steps,) + tuple(obs_shape_meta[key]['shape'])
        assert all_obs.shape == expected_shape, \
            f"Obs {key} shape mismatch: {all_obs.shape} vs expected {expected_shape}"
        data_group.array(
            name=key,
            data=all_obs,
            shape=all_obs.shape,
            chunks=all_obs.shape,
            compressor=None,
            dtype=all_obs.dtype)

    # Decode and store video data
    video_path_template = info.get(
        'video_path',
        'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
    )

    for key in rgb_keys:
        video_key = feature_map['obs'][key]['video_key']
        c = obs_shape_meta[key]['shape'][0]
        target_size = image_resize

        # Create zarr array for images
        img_arr = data_group.require_dataset(
            name=key,
            shape=(n_steps, target_size[0], target_size[1], c),
            chunks=(1, target_size[0], target_size[1], c),
            compressor=None,
            dtype=np.uint8)

        episode_start = 0
        for ep_idx in tqdm(range(n_episodes), desc=f'Decoding video: {key}'):
            chunk_idx = ep_idx // info.get('chunks_size', 1000)
            video_path = os.path.join(
                dataset_path,
                video_path_template.format(
                    episode_chunk=chunk_idx,
                    video_key=video_key,
                    episode_index=ep_idx))

            frames = _decode_video_episode(video_path, target_size)
            ep_len = episodes[ep_idx]['length']
            assert frames.shape[0] == ep_len, \
                f"Video {key} ep {ep_idx}: got {frames.shape[0]} frames, expected {ep_len}"

            img_arr[episode_start:episode_start + ep_len] = frames
            episode_start += ep_len

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
