import numpy as np

from typing import Optional

from regelum import RegelumBase
from regelum.policy import Policy, RLPolicy
from regelum.system import System
from regelum.constraint_parser import ConstraintParser, ConstraintParserTrivial
from regelum.objective import RunningObjective
from regelum.observer import Observer, ObserverTrivial
from regelum.utils import Clock, AwaitedParameter
from regelum.simulator import Simulator
from regelum.event import Event
from regelum.predictor import Predictor, EulerPredictor
from regelum.critic import CriticTrivial
from regelum.model import ModelWeightContainer
from regelum.optimizable.core.configs import CasadiOptimizerConfig
from regelum.scenario import RLScenario, Scenario

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist

import transformations as tftr 
import threading
import math

import traceback 


class ROSScenario(RegelumBase):
    def __init__(
        self,
        policy: Policy,
        system: System,
        state_goal: np.ndarray,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        discount_factor: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.time_old = 0
        self.delta_time = 0
        self.value: float = 0.0

        self.system = system
        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.value_threshold = value_threshold
        self.discount_factor = discount_factor
        self.is_episode_ended = False
        self.constraint_parser = (
            ConstraintParserTrivial()
            if constraint_parser is None
            else constraint_parser
        )
        self.observer = observer if observer is not None else ObserverTrivial()

        self.state = np.zeros(3)

        self.action_init = np.zeros(2)
        self.action = self.action_init.copy()
        self.observation = AwaitedParameter(
            "observation", awaited_from=self.system.get_observation.__name__
        )

        self.policy = policy
        self.sampling_time = sampling_time
        self.clock = Clock(period=sampling_time)
        self.iteration_counter: int = 0
        self.episode_counter: int = 0
        self.step_counter: int = 0
        self.action_old = AwaitedParameter(
            "action_old", awaited_from=self.compute_action.__name__
        )
        self.running_objective = (lambda observation, action: 0)
        self.time_final = kwargs["time_final"]

        print("type of state_goal", type(state_goal))
        self.state_goal = state_goal
        self.rotation_counter = 0
        self.prev_theta = 0
        self.new_state = None
        # ROS 
        self.RATE = rospy.get_param('/rate', 100)
        self.lock = threading.Lock()

        # Topics
        rospy.init_node('ros_preset_node')
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)

        self.reset()

    def get_velocity(self, msg):
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z

    def odometry_callback(self, msg):
        self.lock.acquire()
        
        self.get_velocity(msg)
        # Read current robot state
        x = msg.pose.pose.position.x

        # Complete for y and orientation
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
          
        # Transform quat2euler using tf transformations: complete!
        current_rpy = tftr.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Extract Theta (orientation about Z)
        theta = current_rpy[0]
        
        self.state = [x, y, theta]
        
        # Make transform matrix from 'robot body' frame to 'goal' frame
        
        theta_goal = self.state_goal[2]
        
        # Complete rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta_goal), -np.sin(theta_goal), 0],
            [np.sin(theta_goal), np.cos(theta_goal), 0],
            [0, 0, 1]
        ])
            
        state_matrix = np.vstack([self.state_goal[0], self.state_goal[1], 0])  # [x, y, 0] -- column   
        
        # Compute Transformation matrix 
        self.t_matrix = np.block([
            [rotation_matrix, state_matrix],
            [np.array([0, 0, 0, 1])]
            ])
        
        # Complete rotation counter for turtlebot3
        ''' Your solution goes here (rotation_counter) '''        
        if math.copysign(1, self.prev_theta) != math.copysign(1, theta) and \
            abs(self.prev_theta) > np.pi:
            if math.copysign(1, self.prev_theta) == -1:
                self.rotation_counter -= 1
            
            else:
                self.rotation_counter += 1
        
        self.prev_theta = theta
        theta = theta + 2 * np.pi * self.rotation_counter
        
        # Orientation transform
        new_theta = theta - theta_goal
        
        # Do position transform
        
        ''' 
        Your solution goes here 
        self.new_state = using tranformations :) Have fun!
        '''
        temp = np.array([x, y , 0, 1])
        self.new_state = np.linalg.inv(self.t_matrix) @ temp.T
        self.new_state = [self.new_state[0], self.new_state[1], new_theta]
        
        self.lock.release()

    def run(self):
        try:
            for iteration_counter in range(1, self.N_iterations + 1):
                for episode_counter in range(1, self.N_episodes + 1):
                    self.run_episode(
                        episode_counter=episode_counter, iteration_counter=iteration_counter
                    )
                    self.reload_scenario()

                self.reset_iteration()
                if self.sim_status == "simulation_ended":
                    break
        except Exception as err:
            print("Error:", err)
            print("traceback:", traceback.print_exc())

    def get_action_from_policy(self):
        return self.system.apply_action_bounds(self.policy.action)

    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        rate = rospy.Rate(self.RATE)

        self.episode_start = rospy.get_time()

        while self.sim_status != "episode_ended" and not rospy.is_shutdown():
            self.sim_status = self.step()
            rate.sleep()

    def step(self):
        if self.new_state is not None and \
            (self.value <= self.value_threshold):

            self.time = rospy.get_time() - self.episode_start
            self.observation = np.expand_dims(self.new_state, axis=0)
            self.state = self.new_state
            
            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )

            if np.linalg.norm(self.new_state[:2]) < 0.05:
                self.action = np.zeros_like(self.action)
            
            self.system.receive_action(self.action)

            # Publish action 
            velocity = Twist()

            # Generate ROSmsg from action
            velocity.linear.x = self.action[0, 0]
            velocity.angular.z = self.action[0, 1]
            self.pub_cmd_vel.publish(velocity)

        return "episode_continues"

    @apply_callbacks()
    def reset_iteration(self):
        pass

    @apply_callbacks()
    def reload_scenario(self):
        self.is_episode_ended = False
        self.recent_value = self.value
        self.observation = np.expand_dims(self.new_state, axis=0)
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.action = self.action_init.copy()
        self.reset()
        self.sim_status = 0
        return self.recent_value

    @apply_callbacks()
    def post_compute_action(self, observation, estimated_state):
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
        }

    def compute_action_sampled(self, time, estimated_state, observation):
        self.is_time_for_new_sample = self.clock.check_time(time)
        if self.is_time_for_new_sample:
            self.on_observation_received(time, estimated_state, observation)
            action = self.system.apply_action_bounds(
                self.compute_action(
                    time=time,
                    estimated_state=estimated_state,
                    observation=observation,
                )
            )
            self.post_compute_action(observation, estimated_state)
            self.step_counter += 1
            self.action_old = action
        else:
            action = self.action_old
        return action

    def compute_action(self, time, estimated_state, observation):
        self.issue_action(observation)
        return self.get_action_from_policy()

    def issue_action(self, observation):
        self.policy.update_action(observation)

    def __getattribute__(self, name):
        if name == "issue_action":
            return self._issue_action
        else:
            return object.__getattribute__(self, name)

    def _issue_action(self, observation, *args, **kwargs):
        object.__getattribute__(self, "issue_action")(observation, *args, **kwargs)
        self.on_action_issued(observation)

    def on_action_issued(self, observation):
        self.current_running_objective = self.running_objective(
            observation, self.get_action_from_policy()
        )
        self.value = self.calculate_value(self.current_running_objective, self.time)
        observation_action = np.concatenate(
            (observation, self.get_action_from_policy()), axis=1
        )
        return {
            "action": self.get_action_from_policy(),
            "running_objective": self.current_running_objective,
            "current_value": self.value,
            "observation_action": observation_action,
        }

    def on_observation_received(self, time, estimated_state, observation):
        self.time = time
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
        }

    def substitute_constraint_parameters(self, **kwargs):
        self.policy.substitute_parameters(**kwargs)

    def calculate_value(self, running_objective: float, time: float):
        value = (
            self.value
            + running_objective * self.discount_factor**time * self.sampling_time
        )
        return value

    def reset(self):
        """Reset agent for use in multi-episode simulation.

        Only __internal clock and current actions are reset.
        All the learned parameters are retained.

        """
        self.clock.reset()
        self.value = 0.0
        self.is_first_compute_action_call = True

class RosMPC(RLScenario):
    """Leverages the Model Predictive Control Scenario.

    The MPCScenario leverages the Model Predictive Control (MPC) approach within the reinforcement learning scenario,
    utilizing a prediction model to plan and apply sequences of actions that optimize the desired objectives over a time horizon.
    """

    def __init__(
        self,
        running_objective: RunningObjective,
        simulator: Simulator,
        prediction_horizon: int,
        prediction_step_size: int,
        predictor: Optional[Predictor] = None,
        sampling_time: float = 0.1,
        observer: Optional[Observer] = None,
        constraint_parser: Optional[ConstraintParser] = None,
        discount_factor: float = 1.0,
    ):
        """Initialize the MPC agent, setting up the required structures for MPC.

        Args:
            running_objective (RunningObjective): The objective function
                to assess the costs over the prediction horizon.
            simulator (Simulator): The environment simulation for
                applying and testing the agent.
            prediction_horizon (int): The number of steps into the
                future over which predictions are made.
            predictor (Optional[Predictor]): The prediction model used
                for forecasting future states.
            sampling_time (float): The time step interval for scenario.
            observer (Observer | None): The component for estimating the
                system's current state. Defaults to None.
            constraint_parser (Optional[ConstraintParser]): The
                mechanism for enforcing operational constraints.
                Defaults to None.
            discount_factor (float): The factor for discounting the
                value of future costs. Defaults to 1.0.
        """
        system = simulator.system
        super().__init__(
            N_episodes=1,
            N_iterations=1,
            simulator=simulator,
            policy_optimization_event=Event.compute_action,
            critic=CriticTrivial(),
            running_objective=running_objective,
            observer=observer,
            sampling_time=sampling_time,
            policy=RLPolicy(
                action_bounds=system.action_bounds,
                model=ModelWeightContainer(
                    weights_init=np.zeros(
                        (prediction_horizon + 1, system.dim_inputs), dtype=np.float64
                    ),
                    dim_output=system.dim_inputs,
                ),
                constraint_parser=constraint_parser,
                system=system,
                running_objective=running_objective,
                prediction_horizon=prediction_horizon,
                algorithm="mpc",
                critic=CriticTrivial(),
                predictor=(
                    predictor
                    if predictor is not None
                    else EulerPredictor(system=system, 
                                        pred_step_size=prediction_step_size)
                ),
                discount_factor=discount_factor,
                optimizer_config=CasadiOptimizerConfig(),
            ),
        )

class MyScenario(Scenario):
    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        while self.sim_status != "episode_ended":
            self.sim_status = self.step()
            
            if np.linalg.norm(self.observation[0,:2]) < 0.01:
                break
    
    def calculate_value(self, running_objective: float, time: float):
        if hasattr(self.policy, "score"):
            return self.policy.score
        else:
            return 0
    
    def reset_iteration(self):
        self.policy.reset()
        return super().reset_iteration()
        

