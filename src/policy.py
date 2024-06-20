from numpy.core.multiarray import array as array
from regelum.policy import Policy
import numpy as np
from scipy.special import expit
from src.system import (
    InvertedPendulum,
    InvertedPendulumWithFriction,
    InvertedPendulumWithMotor,
)
from typing import Union
from regelum.utils import rg
from regelum import CasadiOptimizerConfig
from regelum.system import System
from regelum.observer import Observer, ObserverTrivial
from src.config import TestScipyOptimizerConfig

from typing import Optional


def soft_switch(signal1, signal2, gate, loc=np.cos(np.pi / 4), scale=10):

    # Soft switch coefficient
    switch_coeff = expit((gate - loc) * scale)

    return (1 - switch_coeff) * signal1 + switch_coeff * signal2


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


def pd_based_on_sin(observation, pd_coeffs=[20, 10]):
    return -pd_coeffs[0] * np.sin(observation[0, 0]) - pd_coeffs[1] * observation[0, 1]


class InvPendulumPolicyPD(Policy):
    def __init__(self, pd_coeffs: np.ndarray, action_min: float, action_max: float):
        super().__init__()

        self.pid_coeffs = np.array(pd_coeffs).reshape(1, -1)
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray):
        action = np.clip(
            (self.pid_coeffs * observation).sum(),
            self.action_min,
            self.action_max,
        )
        return np.array([[action]])


class InvertedPendulumEnergyBased(Policy):
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coeffs: np.ndarray,
        system: Union[InvertedPendulum, InvertedPendulumWithFriction],
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionCompensation(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coeffs: np.ndarray,
        system: InvertedPendulumWithFriction,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length, friction_coeff = (
            params["mass"],
            params["grav_const"],
            params["length"],
            params["friction_coeff"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + friction_coeff * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionAdaptive(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        sampling_time: float,
        gain_adaptive: float,
        switch_loc: float,
        pd_coeffs: list,
        system: InvertedPendulumWithFriction,
        friction_coeff_est_init: float = 0,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.friction_coeff_est = friction_coeff_est_init
        self.sampling_time = sampling_time
        self.gain_adaptive = gain_adaptive
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1/2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + self.friction_coeff_est * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        # Parameter adaptation using Euler scheme
        self.friction_coeff_est += (
            -self.gain_adaptive
            * energy_total
            * mass
            * length**2
            * np.abs(angle_vel) ** 3
            * self.sampling_time
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )
        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvertedPendulumBackstepping(Policy):

    def __init__(
        self,
        energy_gain: float,
        backstepping_gain: float,
        switch_loc: float,
        pd_coeffs: list[float],
        action_min: float,
        action_max: float,
        system: InvertedPendulumWithMotor,
    ):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.energy_gain = energy_gain
        self.backstepping_gain = backstepping_gain
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = self.system._parameters

        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        torque = observation[0, 2]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.energy_gain * np.sign(angle_vel * energy_total)
        backstepping_action = torque - self.backstepping_gain * (
            torque - energy_control_action
        )
        action_pd = -self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel

        action = hard_switch(
            signal1=backstepping_action,
            signal2=action_pd,
            condition=(np.cos(angle) - 1) ** 2 + angle_vel**2 >= self.switch_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvertedPendulumWithMotorPD(Policy):

    def __init__(self, pd_coeffs: list, action_min: float, action_max: float):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.pd_coeffs = pd_coeffs

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        action = -self.pd_coeffs[0] * angle - self.pd_coeffs[1] * angle_vel
        return np.array([[np.clip(action, self.action_min, self.action_max)]])


class ThreeWheeledRobotKinematicMinGradCLF(Policy):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        eps=0.01,
    ):
        super().__init__(optimizer_config=optimizer_config)
        self.action_bounds = action_bounds
        # An epsilon for numerical stability
        self.eps = eps
        self.instantiate_optimization_procedure()

    def derivative_of_three_wheeled_robot_kin_lyapunov_function(
        self, x_coord, y_coord, angle, vel, angle_vel
    ):
        x_derivative = vel * rg.cos(angle)
        y_derivative = vel * rg.sin(angle)

        return (
            x_coord * x_derivative
            + y_coord * y_derivative
            + (angle - np.arctan(y_coord / (rg.sign(x_coord) * self.eps + x_coord)))
            * (
                angle_vel
                - (y_derivative * x_coord - x_derivative * y_coord)
                / (x_coord**2 + y_coord**2)
            )
        )

    def instantiate_optimization_procedure(self):
        self.x_coord_var = self.create_variable(1, name="x_coord", is_constant=True)
        self.y_coord_var = self.create_variable(1, name="y_coord", is_constant=True)
        self.angle_var = self.create_variable(1, name="angle", is_constant=True)
        self.vel_var = self.create_variable(
            1, name="vel", is_constant=False, like=np.array([[0]])
        )
        self.angle_vel_var = self.create_variable(
            1, name="angle_vel", is_constant=False, like=np.array([[0]])
        )
        self.register_bounds(self.vel_var, np.array(self.action_bounds[None, 0]))
        self.register_bounds(self.angle_vel_var, np.array(self.action_bounds[None, 1]))

        self.register_objective(
            self.derivative_of_three_wheeled_robot_kin_lyapunov_function,
            variables=[
                self.x_coord_var,
                self.y_coord_var,
                self.angle_var,
                self.vel_var,
                self.angle_vel_var,
            ],
        )

    def get_action(self, observation: np.ndarray):
        x_coord = observation[0, 0]
        y_coord = observation[0, 1]
        angle = observation[0, 2]

        optimized_vel_and_angle_vel = self.optimize(
            x_coord=x_coord, y_coord=y_coord, angle=angle
        )

        # The result of optimization is a dict of casadi tensors, so we convert them to float
        angle_vel = float(optimized_vel_and_angle_vel["angle_vel"][0, 0])
        vel = float(optimized_vel_and_angle_vel["vel"][0, 0])

        return np.array([[vel, angle_vel]])


class ThreeWheeledRobotDynamicMinGradCLF(ThreeWheeledRobotKinematicMinGradCLF):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        gain: float,
        eps: float = 0.01,
    ):
        super().__init__(
            optimizer_config=optimizer_config, eps=eps, action_bounds=action_bounds
        )
        self.gain = gain

    def get_action(self, observation: np.ndarray):
        three_wheeled_robot_kin_action = super().get_action(observation)
        force_and_moment = np.array([[observation[0, 3], observation[0, 4]]])
        action = -self.gain * (force_and_moment - three_wheeled_robot_kin_action)

        return action


class ThreeWheeledRobotNomial(Policy):
    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        kappa_params: list[float] = [2, 15, -1.50],
        eps=0.01,
    ):
        super().__init__(optimizer_config=optimizer_config)
        self.action_bounds = action_bounds
        # An epsilon for numerical stability
        self.eps = eps
        self.update_kappa(*kappa_params)

    def update_kappa(self, k_rho, k_alpha, k_beta):
        # Parameters for gazebo
        self.k_rho = k_rho
        self.k_alpha = k_alpha  
        self.k_beta = k_beta

    def get_action(self, observation: np.ndarray):
        x_robot = observation[0, 0]
        y_robot = observation[0, 1]
        theta = observation[0, 2]

        x_goal = 0
        y_goal = 0
        theta_goal = 0

        error_x = x_goal - x_robot
        error_y = y_goal - y_robot
        error_theta = theta_goal - theta

        rho = np.sqrt(error_x**2 + error_y**2)
        alpha = -theta + np.arctan2(error_y, error_x)
        beta = error_theta - alpha

        w = self.k_alpha*alpha + self.k_beta*beta
        v = self.k_rho*rho

        while alpha > np.pi:
            alpha -= 2* np.pi

        while alpha < -np.pi:
            alpha += 2* np.pi

        if -np.pi < alpha <= -np.pi / 2 or np.pi / 2 < alpha <= np.pi:
            v = -v
        
        return np.array([[v, w]])


class ThreeWheeledRobotSimpleMPC(Policy):

    def __init__(
        self,
        optimizer_config: TestScipyOptimizerConfig,
        action_bounds: list[list[float]],
        system: System,
        observer: Optional[Observer] = None,
        R1_diag: list = [],
        Nactor: int = 6,
        gamma: float = 1,
        **kwargs
    ):
        super().__init__(optimizer_config=optimizer_config)
        self.action_bounds = action_bounds

        self.system = system
        self.observer = observer if observer is not None else ObserverTrivial()

        # An R1 for numerical stability
        self.R1 = np.diag(np.array(R1_diag))

        self.current_observation = None
        self.Nactor = Nactor
        self.gamma = gamma
        self.dim_input = system.dim_inputs
        self.dim_output = system.dim_observation
        self.instantiate_optimization_procedure()

    def quadratic_cost_function(self, action_sequence):
        my_action_sqn = np.reshape(action_sequence, [self.Nactor, self.dim_input])
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System observation prediction
        observation_sqn[0, :] = self.current_observation
        state = self.state_sys
        for k in range(1, self.Nactor):
            state = state + self.pred_step_size * self.compute_state_dynamics(0., state, my_action_sqn[k-1, :], _native_dim=True)  # Euler scheme
            observation_sqn[k, :] = self.observer.get_state_estimation(
                None, self.observation, self.action
            )
        
        J = 0
        for k in range(self.Nactor):
            J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])

        return J
    
    def run_obj(self, observation, action):
        if self.run_obj_struct == "quadratic":
            chi = np.concatenate([observation, action])
            cost = chi.T @ self.R1 @ chi
        else:
            cost = 1

        if len(self.obstacle_pos):
            obstacle_gain = 1000
            obs_cost = self.rv.pdf(observation[:2])
            cost += obstacle_gain * obs_cost
        
        return cost

    def instantiate_optimization_procedure(self):
        # numeric optimizer only support 1 variable
        self.action_sequence = self.create_variable(
            2, 
            name="action_sequence", 
            is_constant=False, 
            like=np.zeros((1, self.Nactor*self.dim_input))
        )

        (
            self.action_bounds_tiled,
            self.action_initial_guess,
            self.action_min,
            self.action_max,
        ) = self.handle_bounds(
            self.action_bounds,
            self.dim_action,
            tile_parameter=self.Nactor,
        )
        self.register_bounds(self.action_sequence, self.action_bounds_tiled)
    
        self.register_objective(
            self.quadratic_cost_function,
            variables=[
                self.action_sequence
            ],
        )

    def get_action(self, observation: np.ndarray):
        if len(observation[0]) > 3:
            self.current_observation = observation[0, :3]
        else:
            self.current_observation = observation[0]
        optimized_vel_and_angle_vel = self.optimize(
            action_sequence=self.action_initial_guess
        )

        # The result of optimization is a dict of casadi tensors, so we convert them to float
        angle_vel = float(optimized_vel_and_angle_vel["angle_vel"][0, 0])
        vel = float(optimized_vel_and_angle_vel["vel"][0, 0])

        return np.array([[vel, angle_vel]])
