import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import multivariate_normal as jax_mvn
import math
import time
from jax import random
import sys 
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from utils.data_collect import export_map_to_jsonl, load_map_history_jsonl
# 提取 opt_args 子字典
class TargetDistribution(object):
    def __init__(self, size = 2, peak = None) -> None:
        # 空间定义
        self.n = 2
        self.size = size
        self.grids_points = math.ceil(100 * self.size)
        # 创建空间网格
        self.domain = jnp.meshgrid(
            *[jnp.linspace(0, self.size, self.grids_points)]*self.n
        )
        self._s = jnp.stack([X.ravel() for X in self.domain]).T
        self.file_path = PACKAGE_ROOT / "datas" / "config" / "random_map_history.jsonl"
        self.history = load_map_history_jsonl(self.file_path)
        self._history_index = 0 
        self._dist_params = {
            'means': [],
            'covs': [],
        }
        if peak is not None:
            self._dist_params = peak
        # 初始计算分布值
        self.evals = self._compute_distribution()
    
    def _compute_distribution(self):
        """计算当前参数下的分布值"""
        def p(x):
            length = len(self._dist_params['means'])
            total = 0
            for i in range(length):
                mean = self._dist_params['means'][i]
                cov = self._dist_params['covs'][i]
                total += jax_mvn.pdf(x, mean, cov) / length
            if length == 0:
                total = 1  
            return total
        unnormalized = vmap(p)(self._s)
        normalized = unnormalized / jnp.sum(unnormalized)
        return normalized, self._s
    
    @property
    def grids_x(self):
        return self.domain[0]
    
    @property
    def grids_y(self):
        return self.domain[1]
    
    def get_grids(self):
        return self.grids_x, self.grids_y 
    
    # def plot(self):
    #     plt.contour(self.domain[0], self.domain[1], 
    #                self.evals[0].reshape(self.domain[0].shape))

    def update_map(self, update_times, mode="reset", w_or_r = 'write',perturb_scale=0.3):
        """
        更新地图参数，生成有界的新means。
        mode:
            "perturb"       —— 在当前means附近随机扰动
            "reset"         —— 完全随机生成新的means
        bounds: (min_val, max_val)，每个mean分量的取值范围
        perturb_scale: 控制扰动强度（仅用于perturb模式）
        """
        if w_or_r == 'write':
            key = random.PRNGKey(int(time.time() * 1e6) % (2**32 - 1))
            old_means = self._dist_params["means"]
            num_means = len(old_means)
            new_means = []
            for i in range(num_means):
                key, subkey = random.split(key)
                mean = old_means[i]
                if mode == "perturb":
                    # 在原均值附近添加随机扰动, 均值为0, 方差为perturb_scale的正态分布
                    delta = random.normal(subkey, shape=(2,)) * perturb_scale
                    new_mean = mean + delta
                elif mode == "reset":
                    # 直接在范围内随机采样新的均值
                    new_mean = random.uniform(
                        subkey, shape=(2,), minval = self.size / 6, maxval=self.size * 5 / 6
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                # 强制有界
                new_mean = jnp.clip(new_mean, self.size / 6, self.size * 5 / 6)
                new_means.append(new_mean)
            # 更新地图信息
            self._dist_params["means"] = new_means
            self._dist_params['timestamps'] = [update_times] * num_means
            self.evals = self._compute_distribution()
            export_map_to_jsonl(self._dist_params, log_file=self.file_path)
        elif w_or_r == 'read':
            record = self.history[self._history_index]
            # 关键：还原为 jnp.array 列表
            self._dist_params["means"] = [jnp.array(m) for m in record["means"]]
            self._dist_params["covs"] = [jnp.array(c) for c in record.get("covs", self._dist_params["covs"])]
            self.evals = self._compute_distribution()
            self._history_index += 1  # 自动前进到下一条
        else:
            raise ValueError("w_or_r must be 'write' or 'read'")

    def bayes_filter_reset(self, p_observe, u_t, decay_factor=2.0, sigma_d=0.5):
        """
        将外部地图分布 p_new 融合进当前分布，依据 UAV 位置 u_t 的距离加权。
        Args:
            p_new: jnp.array, shape (N,)，在 self._s 网格上评估的新分布密度
            u_t: jnp.array, shape (2,)，UAV当前位置
            sigma_d: float，距离衰减尺度
        """
        # 1. 计算距离
        diff_to_uav = self._s - u_t  # (N, 2)
        d = jnp.sum(diff_to_uav**2, axis=1)  # (N,)
        # 2. 计算可信度权重（高斯衰减）
        weight_raw = jnp.exp(-d / (2*sigma_d**2))  # (N,)
        weight = weight_raw / (jnp.max(weight_raw) * decay_factor)
        # 3. 加权融合
        p_self = self.evals[0]  # (N,)
        p_fused = (1 - weight) * p_self + weight * p_observe  # (N,)
        p_fused = p_fused / jnp.sum(p_fused) 
        # 4. 更新
        self.evals = (p_fused, self._s)
