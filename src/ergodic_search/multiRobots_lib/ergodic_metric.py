import jax.numpy as np
class ErgodicMetric(object):
    def __init__(self, basis) -> None:
        self.basis = basis
        # lambda_k weighting term for each Fourier basis, shape: (num_basis,)
        # 每个傅里叶基底的权重项 lambda_k，形状为 (num_basis,)
        self.lamk = (1.+np.linalg.norm(basis.k_list/np.pi,axis=1)**2)**(-(basis.n+1)/2.)
        # self.lamk = 1.0
        # lamk = np.exp(-0.8 * np.linalg.norm(k, axis=1))
        # lamk = np.ones((len(k), 1))

    def __call__(self, ck, phik):
        # return (self.lamk * (ck - phik)**2).flatten()
        # 计算遍历度指标：加权的系数差平方和
        # ck: 轨迹的傅里叶系数，phik: 目标分布的傅里叶系数
        # 返回遍历度损失（一个标量），用于衡量轨迹与目标分布的匹配程度
        return np.sum(self.lamk * (ck - phik)**2)

# 定义类用于计算遍历度指标（ergodic metric）。
# 初始化时，依据傅里叶基底的索引向量 k_list 和空间维度 n，计算每个基底的权重 lambda_k（Sobolev范数加权）。
# 调用时，输入轨迹的傅里叶系数 ck 和目标分布的系数 phik，输出加权的系数差平方和作为遍历度损失。
# 损失越小，说明轨迹越“遍历”目标分布，空间覆盖性越好