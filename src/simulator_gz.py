from regelum.simulator import Simulator
from regelum.system import ComposedSystem, System, ThreeWheeledRobotDynamic

import time
import numpy as np
from numpy import ndarray

from src.environment import EnvironmentConfig, EnvironmentManager, TwistAction, Task
import logging


class Robot3Pi(Simulator):
    def __init__(self, 
                 system: System | ComposedSystem,
                 state_init: ndarray | None = None, 
                 action_init: ndarray | None = None, 
                 time_final: float | None = 1, 
                 max_step: float | None = 0.001, 
                 first_step: float | None = 0.000001, 
                 atol: float | None = 0.00001, 
                 rtol: float | None = 0.001
                 ):

        # env_config = EnvironmentConfig(observation_shape=[20,20,3],
        #                                tasks=tasks,
        #                                actions=actions,
        #                                robot_name='3pi_front_cam_robot',
        #                                vehicle_prefix='/vehicle',
        #                                world_name='/world/pushing_objects_world',
        #                                camera_topic='/vehicle/camera')
        # self.set_manager(env_config)

        super().__init__(system, state_init, action_init, time_final, max_step, first_step, atol, rtol)
        self._action = np.expand_dims(self.initialize_init_action(), axis=0)
        self.loginfo = logging.getLogger("regelum").info

    def set_manager(self, env_config:EnvironmentConfig):
        self.manager = EnvironmentManager(env_config)

    # Override this reset
    def reset(self, current_task:Task):
        self.time = 0.0
        self.state = self.state_init
        self.observation = self.observation_init
        self.observation = self.get_observation()

        self.appox_num_step = np.ceil(self.time_final/self.max_step)
        self.episode_start = None

        self.receive_action(np.zeros_like(self._action))
        ###############################################
        self.current_name = current_task.task_name
        self.current_mass = current_task.get('mass')
        self.starting_transform = current_task.get_random_start()

        # stop and wait until robot has stopped
        self.manager.gz_perform_action_stop()
        time.sleep(0.2)
        response = self.manager.get_data()
        # re-place robot
        self.manager.perform_reset(self.starting_transform.position, self.starting_transform.orientation) ;
        time.sleep(0.5)
        response = self.manager.get_data()
        # print("response:", response)
        state = self.manager.convert_image_msg(response)
        self.state = state[::4,::4,:].flatten()

    # Publish action to gazebo
    def publish_action(self, action):
        if not hasattr(self, "is_time_for_new_sample") or not self.is_time_for_new_sample:
            return
        
        try:
            self.manager.gz_perform_action(TwistAction("go", action[0]))
        except Exception as err:
            print("publish_action got Error:", err)

    def update_time(self):
        current_time = self.manager.get_last_obs_time()

        if self.episode_start is None:
            self.episode_start = current_time
        
        self.time = current_time - self.episode_start
        time.sleep(0.005)

    # Stop condition
    # update time, new_state
    def do_sim_step(self):
        '''
        Return: -1: episode ended
                otherwise: episode continues
        '''        
        self.update_time()

        if self.time >= self.time_final:
            return -1
        
        # if rospy.is_shutdown():
        #     raise RuntimeError("Ros shutdowns")

        response = self.manager.get_data()
        state = self.manager.convert_image_msg(response)
        self.observation = self.state = state[::4,::4,:]

        self.publish_action(self.system.inputs)

    def get_observation(self, **kwargs):
        if not hasattr(self, "observation"):
            self.observation = self.state
            # response = self.manager.get_data()
            # self.observation = self.manager.convert_image_msg(response)
        
        return self.observation
