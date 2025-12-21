import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import multivariate_normal as jax_mvn
import matplotlib.pyplot as plt


class TargetDistribution(object):
    def __init__(self) -> None:
        # 空间定义
        self.n = 2
        self.grids_points = 100
        
        # 创建空间网格
        self.domain = jnp.meshgrid(
            *[jnp.linspace(0, 1, self.grids_points)]*self.n
        )
        self._s = jnp.stack([X.ravel() for X in self.domain]).T
        
        # 初始化分布参数
        self._dist_params = {
            'means': [
                jnp.array([0.35, 0.85]),
                jnp.array([0.15, 0.55])
            ],
            'covs': [
                jnp.array([[0.010, 0.0], [0.0, 0.004]]),
                jnp.array([[0.010, 0.0], [0.0, 0.004]]),
            ],
            'weights': jnp.array([1.0, 1.0])
        }
        
        # 初始计算分布值
        self.evals = self._compute_distribution()
    
    def _compute_distribution(self):
        """计算当前参数下的分布值"""
        def p(x):
            total = 0.0
            for i in range(len(self._dist_params['weights'])):
                mean = self._dist_params['means'][i]
                cov = self._dist_params['covs'][i]
                weight = self._dist_params['weights'][i]
                total += weight * jax_mvn.pdf(x, mean, cov)
            return total
        return vmap(p)(self._s), self._s
    
    @property
    def grids_x(self):
        return self.domain[0]
    
    @property
    def grids_y(self):
        return self.domain[1]
    
    def get_grids(self):
        return self.grids_x, self.grids_y 
    
    def plot(self):
        plt.contour(self.domain[0], self.domain[1], 
                   self.evals[0].reshape(self.domain[0].shape))
    
    def update_map(self, new_params=None):
        """
        更新分布参数并重新计算分布值
        
        参数:
        new_params: 字典，包含要更新的参数:
            - 'means': 新的均值列表
            - 'covs': 新的协方差矩阵列表
            - 'weights': 新的权重数组
        """
        if new_params:
            # 更新指定参数
            for key in new_params:
                if key in self._dist_params:
                    self._dist_params[key] = new_params[key]
        
        # 重新计算分布值
        self.evals = self._compute_distribution()
        return self.evals