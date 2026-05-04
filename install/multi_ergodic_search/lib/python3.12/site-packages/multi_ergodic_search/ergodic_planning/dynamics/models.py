import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd, vmap
# 本文件定义了多种动力学模型，包括单积分器、双积分器、自行车模型和三维飞行器模型。
# 每个模型都实现了连续时间状态导数（dfdt）和离散时间状态更新（f），用于轨迹仿真和优化。
# 这些动力学模型可用于遍历搜索和iLQR控制等轨迹优化任务，支持不同类型的机器人

# 单积分器模型
class SingleIntegrator(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 2
        self.m = 2
        B = np.array([
            [1.,0.],
            [0.,1.]
        ])
        def dxdt(x, u):
            return B@u
            # 单积分器的状态导数，线性控制输入
        def f(x, u):
            # B = np.array([
            #     [np.cos(x[2]), 0.,],
            #     [np.sin(x[2]), 0.],
            #     [0., 1.]
            # ])
            return x + self.dt*B@u
            # 单积分器的离散动力学，欧拉积分
        self.f = f
        self.dxdt = dxdt
# 双积分器模型
class DoubleIntegrator:
    def __init__(self, dim=2) -> None:
        self.nx = dim * 2
        self.nu = dim
        A = jnp.eye(dim * 2, dim * 2, k=dim)
        B = jnp.eye(dim * 2, dim, -dim)
        def dxdt(x, u):
            u = jnp.clip(u, -0.5, 0.5)
            return A @ x + B @ u

        self.dxdt = jit(dxdt)
        self.getAt = jit(lambda x, u: A)
        self.getBt = jit(lambda x, u: B)

class HomoDynamics:
    def __init__(self, robot_number, dynamics) -> None:
        self.nx = dynamics.nx
        self.nu = dynamics.nu
        self.dynamics = dynamics
        self.Nx = robot_number * self.nx
        self.Nu = robot_number * self.nu

        def dxdt(x, u):
            # x: horizonal, stack
            xi = x.reshape(-1, self.nx)
            ui = u.reshape(-1, self.nu)
            x_dot = vmap(dynamics.dxdt)(x=xi, u=ui)
            return x_dot.flatten()

        self.dxdt = jit(dxdt)
        self.getAt = jit(jacfwd(self.dxdt, argnums=0))
        self.getBt = jit(jacfwd(self.dxdt, argnums=1))
