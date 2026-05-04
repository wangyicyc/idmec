import jax.numpy as jnp
from jax import vmap
from functools import partial
import numpy as np


# 本文件实现了遍历度指标相关的傅里叶基底工具，包括：
# - get_hk：计算每个基底的归一化因子，保证基底正交归一。
# - get_ck：计算轨迹的傅里叶系数，用于经验分布展开。
# - get_phik：计算目标分布的傅里叶系数，用于目标分布展开。
# - recon_from_fourier：根据傅里叶系数重构空间分布，可选归一化。
# - BasisFunc：傅里叶基底类，支持多维基底生成、基底函数矢量化计算等。
# 这些工具为遍历度指标的高效计算和轨迹优化提供基础
def get_hk(k): # normalizing factor for basis function
    # 计算傅里叶基底的归一化因子
    _hk = (2. * k + np.sin(2 * k))/(4. * k)
    _hk = _hk.at[np.isnan(_hk)].set(1.)
    return np.sqrt(np.prod(_hk))

def get_ck(trajectory, beta, basis):
    # 计算轨迹的傅里叶系数
    fk_values = vmap(basis.fk_vmap)(trajectory[:, :2])
    weighted_fk = fk_values * beta[:, jnp.newaxis]
    # 沿时间轴（axis=0）求和
    ck = jnp.sum(weighted_fk, axis=0)
    # 归一化
    ck = ck / basis.hk_list
    return ck
def get_ck_avg(trajectory, beta, basis):
    # 
    traj_reshape = trajectory.reshape(trajectory.shape[0], -1, 4)
    traj_reshape = traj_reshape.transpose(1, 0, 2)[
        :, :, :2
    ]
    ck = vmap(get_ck, in_axes=(0, None, None))(traj_reshape, beta, basis)
    return jnp.average(ck, axis=0)
    # return jnp.sum(ck, axis=0)

def get_ck_sum(trajectory, beta, basis):
    # 
    traj_reshape = trajectory.reshape(trajectory.shape[0], -1, 4)
    traj_reshape = traj_reshape.transpose(1, 0, 2)[
        :, :, :2
    ]
    beta_reshape = beta.transpose(1, 0)
    ck = vmap(get_ck, in_axes=(0, 0, None))(traj_reshape, beta_reshape, basis)
    # return jnp.average(ck, axis=0)
    return jnp.sum(ck, axis=0)

def get_phik(vals, basis):
    # 计算目标分布的傅里叶系数
    _phi, _x = vals 
    phik = jnp.dot(_phi, vmap(basis.fk_vmap)(_x))
    phik = phik/phik[0]
    phik = phik/basis.hk_list
    return phik

def recon_from_fourier(basis_coef, basis, k_list, x_vals, normalize=False):
    # 用傅里叶系数重构空间分布
    fk_vmap = partial(basis.fk_kvmap, k_list)
    phi = jnp.dot(vmap(fk_vmap)(x_vals), basis_coef)
    if normalize:
        min_phi = jnp.min(phi)
        phi = phi-min_phi+0.1
        phi = phi/jnp.sum(phi)
    return phi

class BasisFunc(object):
    def __init__(self, n_basis, emap=None) -> None:
        # 初始化傅里叶基底
        kmesh = jnp.meshgrid(
                    *[jnp.arange(0,n_max, step=1) for n_max in n_basis]
                )
        self.n = len(n_basis)
        self.k_list = jnp.stack([
                _k.ravel() for _k in kmesh
        ]).T * jnp.pi 

        self.hk_list = jnp.array([
            get_hk(_k) for _k in self.k_list
        ])

        self._fk = lambda k, x: jnp.prod(jnp.cos(x[: self.n] * k))  # 只使用位置部分
        # 如果有映射函数 emap，则用 emap(x) 替换 x
        if emap is not None:
            self._fk = lambda k, x: jnp.prod(jnp.cos(emap(x)*k))
        self.fk_kvmap = vmap(self._fk, in_axes=(0, None))
        self.fk_xvmap = vmap(self._fk, in_axes=(None, 0))
        # self.fk_vmap = partial(self.fk_kvmap, jnp.array([0.1,0.2]))
        self.fk_vmap = partial(self.fk_kvmap, self.k_list)