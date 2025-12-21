import jax
import numpy as np  
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
import jax.numpy as jnp
from multiRobots_lib.integrator import rk4 as int_func
from jax.lax import scan
from multiRobots_lib.class_types import *
import yaml
import logging
logging.basicConfig(
    filename='../datas/logs/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
with open("../datas/config/config.yaml", "r") as f:
    loaded = yaml.safe_load(f)
# 提取 opt_args 子字典
config = loaded["opt_args"]
robot_number = config["robot_number"]
state_dim = 4
control_dim = 2
# see https://github.com/MurpheyLab/ergodic-control-sandbox/blob/main/notebooks/ilqr_ergodic_control.ipynb
# I speed up the algorithm
class iLQR_template:
    def __init__(self, dt, tsteps, Q_z, R_v, dynamics: callable) -> None:
        self.dt = dt
        self.tsteps = tsteps
        self.tf = dt * tsteps
        self.x_dim = getattr(dynamics, "Nx", getattr(dynamics, "nx", None))
        self.u_dim = getattr(dynamics, "Nu", getattr(dynamics, "nu", None))
        self.dynamics = dynamics
        self.Q_z = Q_z
        self.Q_z_inv = jnp.linalg.inv(Q_z)
        self.R_v = R_v
        self.R_v_inv = jnp.linalg.inv(R_v)
        self.dyn_step = jit(partial(int_func, dxdt=self.dynamics.dxdt, dt=self.dt))
        def dyn_step_fn(x, u):
            return self.dyn_step(xt=x, u=u), x
        self.dyn_step_fn = jit(dyn_step_fn)
        # the following functions are utilities for solving the Riccati equation
        # P
        def P_dyn_rev(Pt, At, Bt, at, bt):
            return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_v_inv @ Bt.T @ Pt + self.Q_z
        self.P_dyn_step = jit(partial(int_func, dxdt=P_dyn_rev, dt=self.dt))
        def P_dyn_step_fn(Pt, inputs: LinearizedDynamics):
            At, Bt, at, bt = inputs.At, inputs.Bt, inputs.at, inputs.bt
            return self.P_dyn_step(xt=Pt, At=At, Bt=Bt, at=at, bt=bt), Pt
        self.P_dyn_step_fn = jit(P_dyn_step_fn)
        # r
        def r_dyn_rev(rt, Pt, At, Bt, at, bt):
            return (
                (At - Bt @ self.R_v_inv @ Bt.T @ Pt).T @ rt
                + at
                - Pt @ Bt @ self.R_v_inv @ bt
            )
        self.r_dyn_step = jit(partial(int_func, dxdt=r_dyn_rev, dt=self.dt))
        
        def r_dyn_step_fn(rt, inputs: RiccatiInput):
            return self.r_dyn_step(xt=rt, Pt=inputs.Pt, At=inputs.At, Bt=inputs.Bt, at=inputs.at, bt=inputs.bt), rt
        self.r_dyn_step_fn = jit(r_dyn_step_fn)

        # z /delta
        def z2v(zt, Pt, rt, Bt, bt):
            return (
                -self.R_v_inv @ Bt.T @ Pt @ zt
                - self.R_v_inv @ Bt.T @ rt
                - self.R_v_inv @ bt
            )

        self.z2v = jit(z2v)

        def z_dyn(zt, Pt, rt, At, Bt, bt):
            return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)

        self.z_dyn_step = jit(partial(int_func, dxdt=z_dyn, dt=self.dt))

        def z_dyn_step_fn(zt, inputs: ZDynamicsInput):
            return self.z_dyn_step(xt=zt, Pt=inputs.Pt,
                rt=inputs.rt,   # ⚠ 如果你把 rt 存在 at，这里注意检查
                At=inputs.At, Bt=inputs.Bt, bt=inputs.bt), zt

        self.z_dyn_step_fn = jit(z_dyn_step_fn)

        # self.temp = {'A_traj':[], 'B_traj':[], 'a_traj':[], 'b_traj':[], 'P_traj':[], 'r_traj':[], 'z_traj':[], 'v_traj':[], 'x_traj':[], 'u_traj':[]}

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_bt_vec(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def get_at_bt_traj(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

    def traj_sim(self, x0, u_traj):
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        return x_traj

    def get_descent(self, x0, u_traj, past_traj, target_distr, dual_solution, r_penalty, R_cost):  # 添加参数
        # forward simulate the trajectory
        xN, x_traj = scan(self.dyn_step_fn, x0, u_traj)
        # sovle the Riccati equation backward in time
        A_traj = vmap(self.dynamics.getAt)(x_traj, u_traj)
        B_traj = vmap(self.dynamics.getBt)(x_traj, u_traj)
        a_traj, b_traj = self.get_at_bt_traj(
            TrajectorySolution(x=x_traj, u=u_traj, px=past_traj),
            target_distr, dual_solution, r_penalty, R_cost
        )
        PN = jnp.zeros((self.x_dim, self.x_dim))
        P0, P_traj = scan(
            f=self.P_dyn_step_fn,
            init=PN,
            reverse=True,
            xs = LinearizedDynamics(At=A_traj, Bt=B_traj, at=a_traj, bt=b_traj)
        )
        P_traj = jnp.vstack([P0[jnp.newaxis, :], P_traj])[:-1]
        rN = jnp.zeros(self.x_dim)
        r0, r_traj = scan(
            f=self.r_dyn_step_fn,
            init=rN,
            reverse=True,
            xs = RiccatiInput(Pt=P_traj, At=A_traj, Bt=B_traj, at=a_traj, bt=b_traj)
        )
        r_traj = jnp.vstack([r0[jnp.newaxis, :], r_traj])[:-1]
        z0 = jnp.zeros(self.x_dim)
        zN, z_traj = scan(
            f=self.z_dyn_step_fn,
            init=z0,
            xs = ZDynamicsInput(Pt=P_traj, At=A_traj, Bt=B_traj, rt=r_traj, bt=b_traj)
        )
        # compute the descent direction
        v_traj = vmap(self.z2v)(z_traj, P_traj, r_traj, B_traj, b_traj)
        return v_traj
    def solve(self, *args, **kwargs):
        raise NotImplementedError("Not implemented.")

class al_iLQR(iLQR_template):
    def __init__(
        self, args: dict, objective: callable, dynamics: callable, inequality: callable
    , equality: callable, target_distr, robot_id: int) -> None:
        super().__init__(
            dt=args["dt"],
            tsteps=args["tsteps"],
            Q_z = jnp.diag(args["Q_z"]),
            R_v=jnp.diag(args["R_v"]),
            dynamics=dynamics,
        )
        self.args = args
        self.objective = jit(objective)#, static_argnames=['robot_id'])
        self.inequality = jit(inequality)
        self.equality = jit(equality)
        self.R = jnp.diag(args["R"])
        self.robot_id = robot_id
        self.target_distr = target_distr 
        self.U_min = args["U_min"]
        self.U_max = args["U_max"]

        self.r_penalty = 1.0
        self.dual_solution = None
        self.init_state = None
        self.solution = None
        self.beta = None
        def lagrangian(solution:TrajectorySolution, dual_solution, r, target_distr, R_cost):
            mu = dual_solution.mu
            lam = dual_solution.lam
            _objective = self.objective(solution, self.beta, target_distr, R_cost)
            _ineq_constr = self.inequality(solution)
            _eq_constr = self.equality(solution)
            return _objective + (0.5 / r) * jnp.sum(
                jnp.maximum(0.0, mu + r * _ineq_constr) ** 2 - mu**2
                + jnp.sum(lam * _eq_constr + r*0.5 * (_eq_constr)**2)
            )
            # + jnp.sum(lam * _eq_constr + r*0.5 * (_eq_constr)**2)
        self.lagrangian = jit(lagrangian)
        self.lagrangian_grad = jit(grad(lagrangian, argnums=0))

        def _loss_func(_step, solution, _u_direct, dual, penalty, target_distr, R_cost):
            ctrl = solution.u + _step * _u_direct
            x_traj = self.traj_sim(self.init_state, ctrl)
            new_sol = TrajectorySolution(x=x_traj, u=ctrl, px=solution.px)  # ✅ 构造 NamedTuple
            return self.lagrangian(new_sol, dual, penalty, target_distr, R_cost)
        self.loss_func4linesearch = jit(_loss_func)

    def get_at_bt_traj(self, solution, target_distr, dual_solution, r_penalty, R_cost):  # 添加参数
        grad_val = self.lagrangian_grad(solution, dual_solution, r_penalty, target_distr, R_cost)  # 使用参数
        return grad_val.x, grad_val.u

    def update_multipliers(self):
        new_mu = jnp.maximum(0, self.dual_solution.mu + self.r_penalty * self.inequality(self.solution))
        new_lam = self.dual_solution.lam + self.r_penalty * self.equality(self.solution)

        self.dual_solution = self.dual_solution._replace(mu=new_mu, lam = new_lam)
        
    def linesearch(
        self, solution, dual_solution, u_direct, target_distr, r_penalty, R_cost, max_iter=50, initial_step=1.0, gamma=0.8
    ):
        steps_arr = jnp.array([initial_step * gamma**i for i in range(max_iter)])
        loss_arr = vmap(
        jit(
                partial(
                    self.loss_func4linesearch,
                    solution=solution,
                    _u_direct=u_direct,
                    dual=dual_solution,
                    penalty=r_penalty,
                    target_distr=target_distr,
                    R_cost = R_cost
                )
            )
        )(_step=steps_arr)
        min_loss_idx = jnp.argmin(loss_arr)
        min_step = steps_arr[min_loss_idx]
        min_loss = loss_arr[min_loss_idx]
        ctrl = jax.lax.cond(
            min_loss <= self.lagrangian(
            solution, dual_solution, r_penalty, target_distr, R_cost),
            lambda _: jnp.clip(solution.u + min_step * u_direct, self.U_min, self.U_max),
            # lambda _: solution.u + min_step * u_direct,
            lambda _: solution.u,  # 如果条件不满足，返回原控制输入
            operand=None
        )
        return ctrl
    def update_distribution(self, new_distribution):
        self.target_distr = new_distribution

    def update_dynamics(self, valid_robots):
        self.x_dim = self.dynamics.nx * valid_robots
        self.u_dim = self.dynamics.nu * valid_robots        
        self.Q_z = jnp.diag(jnp.tile(self.args["Q_z"], valid_robots))
        self.Q_z_inv = jnp.linalg.inv(self.Q_z)
        self.R_v = jnp.diag(jnp.tile(self.args["R_v"], valid_robots))    
        self.R_v_inv = jnp.linalg.inv(self.R_v)
        self.dyn_step = jit(partial(int_func, dxdt=self.dynamics.dxdt, dt=self.dt))
        self.R = jnp.diag(jnp.tile(self.args["R"], valid_robots))
        self.U_min = jnp.tile(self.args["U_min"], valid_robots)
        self.U_max = jnp.tile(self.args["U_max"], valid_robots)


    def _extract_valid_robots(self, init_sol):
        x_full = jnp.array(init_sol["x"])          # (T, N * dx)
        u_full = jnp.array(init_sol["u"])          # (T, N * du)
        # 切分为 per-robot arrays: list of (T, dx) or (T, du)
        x_list = [x_full[:, i * state_dim : (i + 1) * state_dim] for i in range(robot_number)]
        u_list = [u_full[:, i * control_dim : (i + 1) * control_dim] for i in range(robot_number)]
        # 判断每个机器人是否有效（只要 x 或 u 不全为 -100 即视为有效）
        valid_robot = []
        for i in range(robot_number):
            # 使用 allclose 容忍浮点误差
            x_is_invalid = jnp.allclose(x_list[i], -100.0, atol=1e-3)
            u_is_invalid = jnp.allclose(u_list[i], -100.0, atol=1e-3)
            if not (x_is_invalid and u_is_invalid):
                valid_robot.append(i)
        if not valid_robot:
            logging.error(f"{self.robot_id}: No valid robot found in init_sol!")
            raise ValueError("No valid robot found in init_sol!")
        # 提取有效机器人的数据
        x_valid = jnp.concatenate([x_list[i] for i in valid_robot], axis=-1)      # (T, N_valid * dx)
        u_valid = jnp.concatenate([u_list[i] for i in valid_robot], axis=-1)      # (T, N_valid * du)
        # logging.info(f"{self.robot_id}: Valid robot: {valid_robot}")
        return x_valid, u_valid, valid_robot
    # 在 return 之前，把 x, u, px 扩展回 robot_number 个机器人
    def _pad_to_full(self, traj_clean, valid_robot, fill_value=-100.0):
        T = traj_clean.shape[0]
        state_dim = traj_clean.shape[1] // len(valid_robot)
        full_traj = jnp.full((T, robot_number * state_dim), fill_value, dtype=traj_clean.dtype)
        for idx_in_valid, global_id in enumerate(valid_robot):
            start = global_id * state_dim
            end = start + state_dim
            clean_start = idx_in_valid * state_dim
            clean_end = clean_start + state_dim
            full_traj = full_traj.at[:, start:end].set(traj_clean[:, clean_start:clean_end])
        return full_traj
    def solve(self, x0, init_sol, beta, init_dual=True, max_iter=100, r_eps = 0.1, loss_eps = 2e-6, decay_eps=0.05, if_print=True):
        x_valid, u_valid, valid_robot  = self._extract_valid_robots(init_sol)
            # --- Step 2: ✅ 裁剪 x0 到有效机器人子集 ---
        x0_clean = jnp.concatenate([
            x0[i * state_dim : (i + 1) * state_dim]
            for i in valid_robot
        ])
        self.init_state = x0_clean
        self.solution = TrajectorySolution(x=x_valid, u=u_valid, px=init_sol["px"])
        beta_x_raw = jnp.asarray(beta["x"] * len(valid_robot)) 
        total_norm = (
            sum(jnp.sum(px) for px in beta["px"]) +
            jnp.sum(beta_x_raw) +
            1e-8
        )
        self.beta = BetaCoefficients(
            x = beta_x_raw / total_norm,
            px = [px / total_norm for px in beta["px"]]
        )
        self.update_dynamics(len(valid_robot))
        if init_dual is True:
            self.get_descent_jit = jit(self.get_descent)
            self.linesearch_jit = jit(self.linesearch)
            # self.dual_solution = {"mu": jnp.zeros_like(self.inequality(self.solution)),\
            #     "lam": jnp.zeros_like(self.equality(self.solution))}
            self.dual_solution = DualVariables(mu=jnp.zeros_like(self.inequality(self.solution)), lam=jnp.zeros_like(self.equality(self.solution)))
            self.update_multipliers()
            self.r_penalty = 1.0
        loss_val = [self.objective(self.solution, self.beta, self.target_distr, self.R)]
        _func_get_violation = jit(
            lambda sol: jnp.maximum(0, self.inequality(sol)).sum()
            + jnp.abs(self.equality(sol)).sum()
            # lambda sol: jnp.abs(self.equality(sol, self.args, self.robot_id)).sum()
        )
        violations = [_func_get_violation(self.solution)]
        # iterative optimization
        for i in range(max_iter):
            # solver LQR Problem
            v_traj = self.get_descent_jit(
                self.init_state, self.solution.u,
                self.solution.px, self.target_distr, 
                self.dual_solution, self.r_penalty, self.R
            )
            # line search
            _u_traj = self.linesearch_jit(self.solution, self.dual_solution, v_traj, self.target_distr, self.r_penalty, self.R)
            self.solution = self.solution._replace(u=_u_traj, x=self.traj_sim(self.init_state, _u_traj))
            # loss_val.append(
            #     self.lagrangian(self.solution, self.dual_solution, self.r_penalty, self.target_distr)
            # )
            loss_val.append(
                self.objective(self.solution, self.beta, self.target_distr, self.R)
            )
            violations.append(_func_get_violation(self.solution))
            if if_print and (i+1) % 40 == 0:
                print(
                    "robot_id:{:d}\titer: {:d}\tobjective: {:.6f}\tlagrangian: {:.6f}\tviolation: {:.6f}\tpenalty: {:.6f}".format(
                        self.robot_id,
                        i,
                        self.objective(self.solution, self.beta, self.target_distr),
                        self.lagrangian(self.solution, self.dual_solution, self.r_penalty, self.target_distr),
                        violations[-1],
                        self.r_penalty,
                    )
                )
            # if (i+1) % 80 == 0:
                # logging.info(f"all violations:{violations}")
                # logging.info(f"id = {self.robot_id}, mu mean  = {jnp.mean(self.dual_solution.mu)}")
                # logging.info(f"id = {self.robot_id}, mu norm  = {jnp.linalg.norm(self.dual_solution.mu)}")
                # logging.info(f"id = {self.robot_id}, violations  = {violations[-1]}, r = {self.r_penalty}")
                # logging.info("id : {:d}| r:{:6f} and violations:{:.6f} and loss{:.6f}".format(self.robot_id, self.r_penalty, violations[-1], loss_val[-1]))
            self.update_multipliers()
            if (loss_val[-2] - loss_val[-1]) < decay_eps and jnp.abs(violations[-1]) > r_eps:
                self.r_penalty = jnp.clip(self.r_penalty * 1.05, 1e-10, 1e5)
                decay_eps *= 0.95            
            if (jnp.abs(loss_val[-1] - loss_val[-2]) < loss_eps) and jnp.abs(
                violations[-1]) < r_eps:
                logging.info("iter:{:d}, id:{:d}, r:{:.3f} and violateion:{:.3f}".format(i, self.robot_id, self.r_penalty, violations[-1]))
                x_full_return = self._pad_to_full(self.solution.x, valid_robot, fill_value=-100.0)
                u_full_return = self._pad_to_full(self.solution.u, valid_robot, fill_value=-100.0)  # control 可填 0
                return {
                    "x": np.array(x_full_return),
                    "u": np.array(u_full_return),
                    "px": [np.array(arr) for arr in self.solution.px]
                }, True

        if jnp.abs(violations[-1]) > r_eps:
            logging.info("failed to satisfy constraint, id: {:d} and r_penalty: {:.3f} and violations:{:.3f}:".format(self.robot_id, self.r_penalty, violations[-1]))
        else:
            logging.info("satisfy constraint, but not converge, id: {:d} and r_penalty: {:.3f} and violations:{:.3f}:".format(self.robot_id, self.r_penalty, violations[-1]))
        
        x_full_return = self._pad_to_full(self.solution.x, valid_robot, fill_value=-100.0)
        u_full_return = self._pad_to_full(self.solution.u, valid_robot, fill_value=-100.0)  # control 可填 0
        return {
                "x": np.array(x_full_return),
                "u": np.array(u_full_return),
                "px": [np.array(arr) for arr in self.solution.px]
        }, True