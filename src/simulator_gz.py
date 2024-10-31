from regelum.simulator import Simulator
from regelum.system import ComposedSystem, System

import time
import numpy as np
from numpy import ndarray

from src.environment import EnvironmentConfig, EnvironmentManager, TwistAction, Task, CatchingRobotManager
import logging
from typing import Tuple


class Robot3Pi(Simulator):
    def __init__(self, 
                 system: System | ComposedSystem,
                 state_init: ndarray | None = None, 
                 action_init: ndarray | None = None, 
                 max_step_per_episode: int = 30,
                 time_final: float | None = 1, 
                 obs_per_sec_sim_time: int | None = 15, 
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
        self.step_count = 0
        self.max_step_per_episode = max_step_per_episode
        
        max_step = 1 / obs_per_sec_sim_time

        super().__init__(system, state_init, action_init, time_final, max_step, first_step, atol, rtol)
        self._action = np.expand_dims(self.initialize_init_action(), axis=0)
        self.loginfo = logging.getLogger("regelum").info
        self.time_final = max_step_per_episode * max_step

    def set_manager(self, env_config:EnvironmentConfig):
        self.manager = EnvironmentManager(env_config)

    # Override this reset
    def reset(self, current_task:Task):
        self.time = 0.0
        self.step_count = 0

        self.episode_start = None

        self.receive_action(np.zeros_like(self._action))
        ###############################################
        self.current_name = current_task.task_name
        self.current_mass = current_task.get('mass')
        self.starting_transform = current_task.get_random_start()

        # stop and wait until robot has stopped
        self.manager.trigger_pause(False)
        try:
            self.manager.gz_perform_action_stop()
            time.sleep(0.2)
            response = self.manager.get_data()
            # re-place robot
            self.manager.perform_reset(self.starting_transform.position, self.starting_transform.orientation) ;
            time.sleep(0.5)
        finally:
            self.manager.trigger_pause(True)

        response = self.manager.get_data()
        # print("response:", response)
        state = self.manager.convert_image_msg(response)
        self.state = self.preprocess(state)
        self.observation = self.get_observation()

    # Publish action to gazebo
    def publish_action(self, action):
        try:
            self.manager.gz_perform_action(TwistAction("go", action[0]))
        except Exception as err:
            print("publish_action got Error:", err)
            print("action:", action)

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
        # print("[do_sim_step] time:", self.time)
        self.update_time()
        self.step_count += 1

        if self.step_count >= self.max_step_per_episode:
            return -1
        
        # if rospy.is_shutdown():
        #     raise RuntimeError("Ros shutdowns")

        self.manager.trigger_pause(False)
        try:
            self.publish_action(self.system.inputs)
            response = self.get_observation_response()
        finally:
            self.manager.trigger_pause(True)

        state = self.manager.convert_image_msg(response)
        self.state = self.preprocess(state)

    def preprocess(self, image):
        return image[::4,::4,:]

    def get_observation_response(self, nsec=0.15):
        if nsec is None:
            nsec = self.step_duration_nsec
        t0 = self.manager.get_last_obs_time()
        i = 0
        last = t0
        while ((self.manager.get_last_obs_time() - t0) < nsec):
            time.sleep(0.001)
            if self.manager.get_last_obs_time() != last: 
              last = self.manager.get_last_obs_time()
            i += 1

            if i > int(nsec/0.001):
                print(f"Counter overflow with t0: {t0} and last moment: {last}")
                break
            pass
        response = self.manager.get_data()
        return response

    def get_observation(self, **kwargs):
        # if not hasattr(self, "observation"):
        self.observation = self.state
            # response = self.manager.get_data()
            # self.observation = self.manager.convert_image_msg(response)
        
        return self.observation


class Robot3PiLineFollowing(Robot3Pi):
    def preprocess(self, image):
        return image[1:3,::2,:]
    
    def glue_images(self, img):
        if self.step_count == 0:
          self.img0 = self.img1 = img ;
        elif self.step_count == 1:
          self.img0 = self.img1 ;
          self.img1 = img ;
        ret = np.concatenate([self.img0,self.img1,img]) ;
        self.img0 = self.img1 ; self.img1 = img ;
        return ret ;
    
    def get_observation(self, **kwargs):
        return self.glue_images(self.state)


class Robot3PiRobotPursuit(Robot3Pi):
    def set_manager(self, env_config):
        self.manager = CatchingRobotManager(env_config)

    # Override this reset
    def reset(self, current_task:Task):
        self.time = 0.0
        self.step_count = 0

        self.episode_start = None

        self.receive_action(np.zeros_like(self._action))
        ###############################################
        self.current_name = current_task.task_name
        self.current_mass = current_task.get('mass')
        self.starting_transform = current_task.get_random_start()

        # stop and wait until robot has stopped
        self.manager.trigger_pause(False)
        self.manager.gz_stop_runner()
        self.manager.gz_perform_action_stop()
        time.sleep(0.2)
        response = self.manager.get_data()
        # re-place robot
        self.manager.perform_reset(self.starting_transform.position, self.starting_transform.orientation) ;
        response = self.manager.get_data()

        self.manager.gz_start_runner()
        time.sleep(0.5)
        self.manager.trigger_pause(True)
        
        # print("response:", response)
        state = self.manager.convert_image_msg(response)
        self.state = self.preprocess(state)
        self.observation = self.get_observation()

    def do_sim_step(self):
        '''
        Return: -1: episode ended
                otherwise: episode continues
        '''     
        # print("[do_sim_step] time:", self.time)
        self.update_time()
        self.step_count += 1

        if self.step_count >= self.max_step_per_episode:
            return -1
        
        # if rospy.is_shutdown():
        #     raise RuntimeError("Ros shutdowns")

        self.manager.trigger_pause(False)
        self.runner_out_of_bounds(self.manager.get_runner_position())
        self.publish_action(self.system.inputs)
        response = self.get_observation_response()
        self.manager.trigger_pause(True)

        state = self.manager.convert_image_msg(response)
        self.state = self.preprocess(state)

    def set_arena_bounds(self, arena_bounds):
        self.arena_bounds = arena_bounds

    def runner_out_of_bounds(self, runner_position:Tuple[float,float,float]):
        if runner_position[1] < self.arena_bounds[0]: # West of the arena
            self.manager.rotate_runner(self.cardinal_directions['east'])

        elif runner_position[1] > self.arena_bounds[1]: # East of the arena
            self.manager.rotate_runner(self.cardinal_directions['west'])

        if runner_position[0] < self.arena_bounds[2]: # South of the arena
            self.manager.rotate_runner(self.cardinal_directions['north'])

        elif runner_position[0] > self.arena_bounds[3]: # North of the arena
            self.manager.rotate_runner(self.cardinal_directions['south'])
