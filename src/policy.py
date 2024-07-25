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

import scipy as sp
from scipy.optimize import minimize
from numpy.linalg import norm
from numpy.matlib import repmat
from .utilities import (
    uptria2vec,
    to_row_vec,
    to_scalar,
    push_vec,
    rep_mat
    )

# from regelum.animation import (
#     ObjectiveAnimation
# )


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
        action_bounds: list[list[float]],
        kappa_params: list[float] = [2, 15, -1.50],
        eps=0.01,
    ):
        super().__init__()
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


# @ObjectiveAnimation.attach
class ThreeWheeledRobotCALFQ(Policy):
    def __init__(
        self,
        # action_bounds: list[list[float]],
        system: Union[System],
        # R1_diag: list = [],
    ):
        super().__init__()
        action_bounds = np.array([[-0.22, 0.22], [-2.84, 2.84]])
        R1_diag = [1, 1, 1e-1, 0, 0]

        # Critic
        self.critic_learn_rate = 0.1
        self.critic_num_grad_steps = 20
        self.critic_struct = "quad-mix"

        critic_big_number = 1e3

        self.action_sampling_time = 0.1  # Taken from common/inv_pendulum config
        self.step_size_multiplier = 1
        self.discount_factor = 0.9
        self.buffer_size = 20

        # CALFQ
        self.obstacle_x = -0.5
        self.obstacle_y = -0.5
        self.obstacle_sigma = 0.2

        # Probability to take CALF action even when CALF constraints are not satisfied
        self.relax_probability = 0.0
        self.critic_init_fading_factor = 0.8
        
        self.score = 0
        self.score_safe = np.inf
        self.calf_penalty_coeff = 0.5

        self.critic_low_kappa_coeff = 1e-1
        self.critic_up_kappa_coeff = 1e3
        # Nominal desired step-wise decay of critic
        self.critic_desired_decay = 1e-5 * self.action_sampling_time
        # Maximal desired step-wise decay of critic
        self.critic_max_desired_decay = 1e-1 * self.action_sampling_time
        self.critic_weight_change_penalty_coeff = 1.0

        self.nominal_ctrl = ThreeWheeledRobotNomial(action_bounds=action_bounds)

        self.action_min = np.array( action_bounds[:,0] )
        self.action_max = np.array( action_bounds[:,1] )

        self.system = system

        # rc
        # Initialization of CALFQ

        self.dim_state = system.dim_observation
        self.dim_action = system.dim_inputs
        self.dim_observation = self.dim_state
        self.dim_observation = self.dim_state

        # Taken from initial_conditions config
        # self.state_init = np.expand_dims(self.system.state, axis=0)
        self.state_init = self.system.state
        self.observation_init = self.state_init
        self.action_init = self.system.apply_action_bounds(self.get_safe_action(self.state_init))
        self.action_curr = self.action_init

        self.run_obj_param_tensor = np.diag(R1_diag)

        self.action_buffer = repmat(self.action_init, self.buffer_size, 1)
        self.observation_buffer = repmat(self.observation_init, self.buffer_size, 1)

        if self.critic_struct == "quad-lin":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
                + (self.dim_observation + self.dim_action)
            )
            self.critic_weight_min = -critic_big_number
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quadratic":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
            )
            self.critic_weight_min = 0
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quad-nomix":
            self.dim_critic = self.dim_observation + self.dim_action
            self.critic_weight_min = 0
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quad-mix":
            self.dim_critic = int(
                self.dim_observation
                + self.dim_observation * self.dim_action
                + self.dim_action
            )
            self.critic_weight_min = -critic_big_number
            self.critic_weight_max = critic_big_number

        self.critic_weight_tensor_init = to_row_vec(
            np.random.uniform(10, critic_big_number, size=self.dim_critic)
        )
        self.critic_weight_tensor = self.critic_weight_tensor_init
        self.critic_buffer_safe = []

        self.critic_weight_tensor_safe = self.critic_weight_tensor_init
        self.observation_safe = self.observation_init
        self.action_safe = self.action_init

        self.action_buffer_safe = np.zeros([self.buffer_size, self.dim_action])
        self.observation_buffer_safe = np.zeros(
            [self.buffer_size, self.dim_observation]
        )

        self.calf_count = 0
        self.safe_count = 0
        self.log_params = {
                "use_calf": 0,
                "critic_new": 0,
                "critic_safe": 0
            }

        # Debugging
        self.debug_print_counter = 0
        # /rc

    # rc
    def obstacle_penalty(self, observation, penalty_factor):
        """
        Calculates the value of probability density function of a bivariate normal distribution at a given point.
        Arguments:
        x, y : float
            Coordinates of the point at which to calculate the probability density value.
        mu_x, mu_y : float
            Mean values (expectations) along the X and Y axes, respectively.
        sigma_x, sigma_y : float
            Standard deviations along the X and Y axes, respectively.
        rho : float
            Correlation coefficient between X and Y.

        Returns:
        float
            Value of the probability density function of a bivariate normal distribution at the given point (x, y).
        """
        mu_x = self.obstacle_x
        sigma_x = self.obstacle_sigma

        mu_y = self.obstacle_y
        sigma_y = self.obstacle_sigma
        rho = 0
        x = observation[0, 0]
        y = observation[0, 1]
        z = ((x - mu_x) ** 2) / (sigma_x ** 2) + ((y - mu_y) ** 2) / (sigma_y ** 2) - (2 * rho * (x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        denom = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)
        return np.exp(-z / (2 * (1 - rho ** 2))) / denom * penalty_factor

    def run_obj(self, observation, action):
        observation_action = np.hstack(
            [to_row_vec(observation), to_row_vec(action)]
        )

        penalty = self.obstacle_penalty(to_row_vec(observation), penalty_factor=1e1)
        result = observation_action @ self.run_obj_param_tensor @ observation_action.T + penalty

        return to_scalar(result)

    def critic_model(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        result = critic_weight_tensor @ feature_tensor.T

        return to_scalar(result) 

    def critic_model_grad(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        return feature_tensor

    def critic_obj(self, critic_weight_tensor_change):
        """
        Objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor = (
            self.critic_weight_tensor_safe + critic_weight_tensor_change
        )

        result = 0

        for k in range(self.buffer_size - 1, 0, -1):
            # Python's array slicing may return 1D arrays, but we don't care here
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # observation_prev_safe = self.observation_buffer_safe[k-1, :]
            # observation_next_safe = self.observation_buffer_safe[k, :]
            # action_prev_safe = self.action_buffer_safe[k-1, :]
            # action_next_safe = self.action_buffer_safe[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += 1 / 2 * temporal_error**2

        # result += (
        #     1
        #     / 2
        #     * self.critic_weight_change_penalty_coeff
        #     * norm(critic_weight_tensor_change) ** 2
        # )

        return result
    
    def critic_obj_2(self, critic_weight_tensor):
        """
        Objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        ############################################################ Only focus on this
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe

        result = 0

        for k in range(self.buffer_size - 1, 0, -1):
            # Python's array slicing may return 1D arrays, but we don't care here
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # observation_prev_safe = self.observation_buffer_safe[k-1, :]
            # observation_next_safe = self.observation_buffer_safe[k, :]
            # action_prev_safe = self.action_buffer_safe[k-1, :]
            # action_next_safe = self.action_buffer_safe[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += 1 / 2 * temporal_error**2

        # result += (
        #     1
        #     / 2
        #     * self.critic_weight_change_penalty_coeff
        #     * norm(critic_weight_tensor_change) ** 2
        # )

        ############################################################

        return result

    def critic_obj_grad(self, critic_weight_tensor):
        """
        Gradient of the objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor_change = critic_weight_tensor_pivot - critic_weight_tensor

        result = to_row_vec(np.zeros(self.dim_critic))

        for k in range(self.buffer_size - 1, 0, -1):

            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # observation_prev_safe = self.observation_buffer_safe[k-1, :]
            # observation_next_safe = self.observation_buffer_safe[k, :]
            # action_prev_safe = self.action_buffer_safe[k-1, :]
            # action_next_safe = self.action_buffer_safe[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += temporal_error * self.critic_model_grad(
                critic_weight_tensor, observation_prev, action_prev
            )

        result += self.critic_weight_change_penalty_coeff * critic_weight_tensor_change

        return result

    def calf_diff(self, critic_weight_tensor, observation, action, debug=False):
        # Q^w  (s_t, a_t)
        critic_new = self.critic_model(critic_weight_tensor, observation, action)
        # Q^w† (s†, a†)
        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )
        if debug:
            return critic_new - critic_safe, critic_new, critic_safe    

        return critic_new - critic_safe

    def calf_decay_constraint_penalty_grad(
        self, critic_weight_tensor, observation, action
    ):
        # This one is handy for explicit gradient-descent optimization.
        # We take a ReLU here

        critic_new = self.critic_model(critic_weight_tensor, observation, action)

        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )

        if critic_new - critic_safe <= -self.critic_desired_decay:
            relu_grad = 0
        else:
            relu_grad = 1

        return (
            self.calf_penalty_coeff
            * self.critic_model_grad(critic_weight_tensor, observation, action)
            * relu_grad
        )

        # Quadratic penalty
        # return (
        #     self.calf_penalty_coeff
        #     * self.critic_model_grad(critic_weight_tensor, observation, action)
        #     * (critic_new - critic_safe + self.critic_desired_decay)
        # )

    def get_optimized_critic_weights(
        self,
        observation,
        action,
        use_grad_descent=True,
        use_calf_constraints=True,
        use_kappa_constraint=False,
        check_persistence_of_excitation=False,
    ):
        # Optimization method of critic. Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        critic_opt_method = "SLSQP"
        if critic_opt_method == "trust-constr":
            # 'disp': True, 'verbose': 2}
            critic_opt_options = {"maxiter": 40, "disp": False}
        else:
            critic_opt_options = {
                "maxiter": 40,
                "maxfev": 80,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-3,
                "fatol": 1e-3,
            }  # 'disp': True, 'verbose': 2}

        critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
        critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2

        constraints = []
        constraints.append(
            sp.optimize.NonlinearConstraint(
                lambda critic_weight_tensor: self.calf_diff(
                    critic_weight_tensor=critic_weight_tensor,
                    observation=observation,
                    action=action,
                ),
                -np.inf,
                -self.critic_desired_decay,
            )
        )
        if use_kappa_constraint:

            constraints.append(
                sp.optimize.NonlinearConstraint(
                    lambda critic_weight_tensor: self.critic_model(
                        critic_weight_tensor=critic_weight_tensor,
                        observation=observation,
                        action=action,
                    ),
                    critic_low_kappa,
                    critic_up_kappa,
                )
            )

        bounds = sp.optimize.Bounds(
            self.critic_weight_min, self.critic_weight_max, keep_feasible=True
        )

        critic_weight_tensor_change_start_guess = to_row_vec(np.zeros(self.dim_critic))
        critic_weight_tensor_start_guess = to_row_vec(np.zeros(self.dim_critic))

        if use_calf_constraints:
            if use_grad_descent:

                critic_weight_tensor = self.critic_weight_tensor

                for _ in range(self.critic_num_grad_steps):

                    critic = self.critic_model(
                        critic_weight_tensor, observation, action
                    )

                    # Simple ReLU penalties for bounding kappas
                    if critic <= critic_up_kappa:
                        relu_kappa_up_grad = 0
                    else:
                        relu_kappa_up_grad = 1

                    if critic >= critic_low_kappa:
                        relu_kappa_low_grad = 0
                    else:
                        relu_kappa_low_grad = 1

                    critic_weight_tensor_change = -self.critic_learn_rate * (
                        self.critic_obj_grad(critic_weight_tensor)
                        + self.calf_decay_constraint_penalty_grad(
                            self.critic_weight_tensor, observation, action
                        )
                        + self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_low_grad
                        + self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_up_grad
                    )
                    critic_weight_tensor += critic_weight_tensor_change

            else:
                ############################################################ Only focus on this
                critic_weight_tensor = minimize(
                    self.critic_obj_2,
                    to_row_vec(self.critic_weight_tensor_safe)[0],
                    method=critic_opt_method,
                    tol=1e-3,
                    bounds=bounds,
                    constraints=constraints,
                    options=critic_opt_options,
                ).x
                ############################################################
        else:
            if use_grad_descent:
                critic_weight_tensor_change = (
                    -self.critic_learn_rate
                    * self.critic_obj_grad(self.critic_weight_tensor)
                )
            else:
                critic_weight_tensor_change = minimize(
                    self.critic_obj(critic_weight_tensor_change),
                    critic_weight_tensor_change_start_guess,
                    method=critic_opt_method,
                    tol=1e-3,
                    bounds=bounds,
                    options=critic_opt_options,
                ).x

        if check_persistence_of_excitation:
            # Adjust the weight change by the replay condition number
            critic_weight_tensor_change *= (
                1
                / np.linalg.cond(self.observation_buffer)
                * 1
                / np.linalg.cond(self.action_buffer)
            )

        return np.clip(
            critic_weight_tensor,
            self.critic_weight_min,
            self.critic_weight_max,
        )

    def actor_obj(self, action_change, critic_weight_tensor, observation):
        """
        Objective function for actor learning.

        """

        return self.critic_model(
            critic_weight_tensor, observation, self.action_curr + action_change
        )
    
    def actor_obj_2(self, action, critic_weight_tensor, observation):
        """
        Objective function for actor learning.

        """        
        ############################################################ Only focus on this
        # System observation prediction
        state = observation
        next_state = state[0] + self.action_sampling_time * self.step_size_multiplier * self.system.compute_state_dynamics(0., state[0], action, _native_dim=True)  # Euler scheme
        _observation = np.expand_dims(next_state, axis=0)

        Q = self.critic_model(
                critic_weight_tensor, _observation, action
            )

        ############################################################
        return Q

    def get_optimized_action(self, critic_weight_tensor, observation):

        actor_opt_method = "SLSQP"
        if actor_opt_method == "trust-constr":
            actor_opt_options = {
                "maxiter": 40,
                "disp": False,
            }  #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {
                "maxiter": 40,
                "maxfev": 60,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-3,
                "fatol": 1e-3,
            }  # 'disp': True, 'verbose': 2}

        
        if False:
            action_change_start_guess = np.zeros(self.dim_action)
            bounds = sp.optimize.Bounds(
                self.action_min, self.action_max, keep_feasible=True
            )

            action_change = minimize(
                lambda action_change: self.actor_obj(
                    action_change, critic_weight_tensor, observation
                ),
                action_change_start_guess,
                method=actor_opt_method,
                tol=1e-3,
                bounds=bounds,
                options=actor_opt_options,
            ).x

            return self.action_curr + action_change
        else:
            ############################################################ Only focus on this
            action_start_guess = np.zeros(self.dim_action)

            bounds = sp.optimize.Bounds(
                self.action_min, self.action_max, keep_feasible=True
            )

            action = minimize(
                lambda action: self.actor_obj_2(
                    action, critic_weight_tensor, observation
                ),
                action_start_guess,
                method=actor_opt_method,
                tol=1e-3,
                bounds=bounds,
                options=actor_opt_options,
            ).x

            ############################################################ Only focus on this
            return action


    def calf_filter(self, critic_weight_tensor, observation, action):
        """
        If CALF constraints are satisfied, put the specified action through and update the CALF's state
        (safe weights, observation and action).
        Otherwise, return a safe action, do not update the CALF's state.

        """

        critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
        critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2

        sample = np.random.rand()
        # condition_1 = -self.critic_max_desired_decay \
        #     <= self.calf_diff(critic_weight_tensor, observation, action) \
        #     <= -self.critic_desired_decay
        
        calf_diff, critic_new, critic_safe = self.calf_diff(critic_weight_tensor, observation, action, debug=True)

        condition_1 = calf_diff <= -self.critic_desired_decay
        
        condition_2 = critic_low_kappa \
                        <= self.critic_model(
                            critic_weight_tensor,
                            observation,
                            action,
                        ) \
                        <= critic_up_kappa
        
        self.log_params["critic_new"] = critic_new
        self.log_params["critic_safe"] = critic_safe

        if (
            (condition_1
             and condition_2
             and norm(observation[0, :2]) > 0.2)
            or sample <= self.relax_probability
        ):
            print("\033[93m") # Color it to indicate CALF
            print("new", critic_weight_tensor, observation, action)
            print("LG", self.critic_weight_tensor_safe, self.observation_safe, self.action_safe)
            print("calf_diff", critic_new, critic_safe)
            print("Condition 1: {} - value: {} - LSL: {} - USL: {}".format(
                condition_1, 
                self.calf_diff(critic_weight_tensor, observation, action),
                -self.critic_max_desired_decay,
                -self.critic_desired_decay
                ))
            print("Condition 2: {} - value: {}".format(
                condition_2, 
                self.critic_model(critic_weight_tensor, observation, action)
                ))
            print("critic_weight_tensor: {} - observation: {}, action: {}".format(critic_weight_tensor, observation, action))
            
            self.critic_weight_tensor_safe = critic_weight_tensor
            self.observation_safe = observation
            self.action_safe = action

            self.critic_buffer_safe.append(critic_weight_tensor)
            self.observation_buffer_safe = push_vec(
                self.observation_buffer_safe, observation
            )
            self.action_buffer_safe = push_vec(self.action_buffer_safe, action)

            self.use_calf = 1
            self.log_params["use_calf"] = 0
            return action

        else:
            print("\033[0m")
            print("new", critic_weight_tensor, observation, action)
            print("LG", self.critic_weight_tensor_safe, self.observation_safe, self.action_safe)
            print("calf_diff", critic_new, critic_safe)
            print("Condition 1: {} - value: {} - LSL: {} - USL: {}".format(
                condition_1, 
                self.calf_diff(critic_weight_tensor, observation, action),
                -self.critic_max_desired_decay,
                -self.critic_desired_decay
                )) 
            print("Condition 2: {} - value: {} - LKappa: {} - UKappa: {}".format(
                condition_2, 
                self.critic_model(critic_weight_tensor, observation, action),
                critic_low_kappa,
                critic_up_kappa
                ))
            print("critic_weight_tensor: {} - observation: {}, action: {}".format(critic_weight_tensor, observation, action))

            self.safe_count += 1
            self.log_params["use_calf"] = 0
            return self.get_safe_action(observation)

    def get_safe_action(self, observation: np.ndarray) -> np.ndarray:
        return self.nominal_ctrl.get_action(observation)

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        # Update replay buffers
        self.action_buffer = push_vec(self.action_buffer, self.action_curr)
        self.observation_buffer = push_vec(self.observation_buffer, observation)

        # Update score (cumulative objective)

        self.current_score = self.run_obj(observation, self.action_curr)
        self.score += (
            self.current_score * self.action_sampling_time
        )

        # Update action
        new_action = self.get_optimized_action(self.critic_weight_tensor, observation)

        # Compute new critic weights
        self.critic_weight_tensor = self.get_optimized_critic_weights(
            observation,
            new_action,
            use_calf_constraints=True,
            use_grad_descent=False,
            use_kappa_constraint=True
        )

        # Apply CALF filter that checks constraint satisfaction and updates the CALF's state
        action = self.calf_filter(self.critic_weight_tensor, observation, new_action)

        # DEBUG
        # action = self.get_safe_action(observation)
        # /DEBUG

        # Apply action bounds
        action = np.clip(
            action,
            self.action_min,
            self.action_max,
        )

        # Force proper dimensionsing according to the convention
        action = to_row_vec(action)

        # Update current action
        self.action_curr = action

        # DEBUG
        np.set_printoptions(precision=3)

        if self.debug_print_counter % 10 == 0:
            print(
                "--DEBUG-- reward: %4.2f score: %4.2f"
                % (-self.run_obj(observation, action), -self.score)
            )
            print("--DEBUG-- critic weights:", self.critic_weight_tensor)
            print("--DEBUG-- CALF counter:", self.calf_count)
            print("--DEBUG-- Safe counter:", self.safe_count)

        self.debug_print_counter += 1

        # /DEBUG

        return action

    def reset(self):
        ############################################################ Only focus on this
        self.action_init = self.system.apply_action_bounds(self.get_safe_action(self.state_init))
        self.action_curr = self.action_init

        self.action_buffer = repmat(self.action_init, self.buffer_size, 1)
        self.observation_buffer = repmat(self.observation_init, self.buffer_size, 1)

        self.observation_safe = self.observation_init
        self.action_safe = self.action_init

        ############################################################  Reset last good buffers
        self.action_buffer_safe = np.zeros([self.buffer_size, self.dim_action])
        self.observation_buffer_safe = np.zeros([self.buffer_size, self.dim_observation])
        ############################################################

        self.calf_count = 0
        self.safe_count = 0

        total_sum = 0
        weight_list = []

        N = len(self.critic_buffer_safe)  
        for i, w_i in enumerate(self.critic_buffer_safe, start=0):
            weight_list.append(self.critic_init_fading_factor ** i)
            total_sum += weight_list[-1] * w_i
            print(f"weight {i}:", w_i)
        if N != 0:
            # weighted_average = total_sum / N
            weighted_average = total_sum / np.sum(weight_list)
        else:
            weighted_average = self.critic_weight_tensor

        Delta_w =  weighted_average - self.critic_weight_tensor_init

        if self.score <= self.score_safe:
            print(f"Final cost:    {self.score:.2f}" + f"    Best cost:    {self.score_safe:.2f}")
            self.score_safe = self.score
            self.critic_weight_tensor_safe_init = self.critic_weight_tensor_init.copy()
            self.critic_weight_tensor_init = self.critic_weight_tensor_init + self.calf_penalty_coeff * Delta_w
        else:
            self.critic_weight_tensor_init = self.critic_weight_tensor_safe_init + \
                self.calf_penalty_coeff * np.clip(self.critic_weight_max/5 * np.random.normal(size=self.dim_critic), 
                                                  self.critic_weight_min, self.critic_weight_max)
            
        print("critic_weight_tensor_init: ", self.critic_weight_tensor_init)

        self.critic_weight_tensor_safe = self.critic_weight_tensor_init
        self.score = 0
        self.critic_buffer_safe.clear()
        ############################################################
