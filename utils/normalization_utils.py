import numpy as np
import torch
from typing import Tuple, Dict, Any


class GlobalNormalizer:

    def __init__(self):
        self.mean_train = None
        self.std_train = None
        self.is_fitted = False
        self.dataset_type = None

    def fit(self, train_data: np.ndarray, dataset_type: str = "default") -> 'GlobalNormalizer':
        self.dataset_type = dataset_type


        original_shape = train_data.shape
        if len(original_shape) == 4:
            data_reshaped = train_data.reshape(original_shape[0], original_shape[1], -1)
        elif len(original_shape) == 2:
            data_reshaped = train_data
        elif len(original_shape) == 3:
            data_reshaped = train_data.reshape(original_shape[0], -1)
        else:
            raise ValueError(f"不支持的数据形状: {original_shape}")


        self.mean_train = np.mean(data_reshaped, axis=(0, 2))
        self.std_train = np.std(data_reshaped, axis=(0, 2))


        self.std_train = np.where(self.std_train < 1e-8, 1.0, self.std_train)

        self.is_fitted = True
        print(f"计算完成 {dataset_type} 数据集统计参数:")
        print(f"  形状: {original_shape}")
        print(f"  均值范围: [{self.mean_train.min():.4f}, {self.mean_train.max():.4f}]")
        print(f"  标准差范围: [{self.std_train.min():.4f}, {self.std_train.max():.4f}]")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("必须在使用transform()之前调用fit()计算统计参数")


        if len(data.shape) < 2:
            raise ValueError(f"数据维度过低: {data.shape}")


        normalized_data = (data - self.mean_train.reshape(-1, 1, 1)) / self.std_train.reshape(-1, 1, 1)

        return normalized_data

    def fit_transform(self, train_data: np.ndarray, dataset_type: str = "default") -> np.ndarray:
        return self.fit(train_data, dataset_type).transform(train_data)

    def save_stats(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("没有可保存的统计参数")

        stats_dict = {
            'mean_train': self.mean_train,
            'std_train': self.std_train,
            'dataset_type': self.dataset_type,
            'is_fitted': self.is_fitted
        }

        np.savez(filepath, **stats_dict)
        print(f"统计参数已保存到: {filepath}")

    def load_stats(self, filepath: str):
        loaded_data = np.load(filepath)

        self.mean_train = loaded_data['mean_train']
        self.std_train = loaded_data['std_train']
        self.dataset_type = loaded_data['dataset_type'].item() if 'dataset_type' in loaded_data else "loaded"
        self.is_fitted = loaded_data['is_fitted']

        print(f"统计参数已从 {filepath} 加载")
        print(f"  数据集类型: {self.dataset_type}")
        print(f"  均值形状: {self.mean_train.shape}")
        print(f"  标准差形状: {self.std_train.shape}")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.transform(data)


def calculate_global_stats(train_data: np.ndarray, data_type: str = "dataset") -> Dict[str, np.ndarray]:
    original_shape = train_data.shape

    if len(original_shape) == 4:
        data_reshaped = train_data.reshape(original_shape[0], original_shape[1], -1)
    elif len(original_shape) == 2:
        data_reshaped = train_data
    elif len(original_shape) == 3:
        data_reshaped = train_data.reshape(original_shape[0], -1)
    else:
        raise ValueError(f"不支持的数据形状: {original_shape}")

    mean = np.mean(data_reshaped, axis=(0, 2))
    std = np.std(data_reshaped, axis=(0, 2))
    std = np.where(std < 1e-8, 1.0, std)

    print(f"{data_type} 统计参数计算完成:")
    print(f"  原始形状: {original_shape}")
    print(f"  均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  标准差范围: [{std.min():.4f}, {std.max():.4f}]")

    return {'mean': mean, 'std': std}


def normalize_with_stats(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
