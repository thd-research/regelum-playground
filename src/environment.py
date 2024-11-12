import numpy as np ;
from typing import * ;
from scipy.spatial.transform import Rotation
from threading import Lock ;
from src.rgenv import RgEnv

from gz.transport13 import Node ;
from gz.msgs10.empty_pb2 import Empty ;
from gz.msgs10.scene_pb2 import Scene ;
from gz.msgs10.image_pb2 import Image ;
from gz.msgs10.twist_pb2 import Twist ;
from gz.msgs10.world_control_pb2 import WorldControl ;
from gz.msgs10.boolean_pb2 import Boolean ;
from gz.msgs10.pose_pb2 import Pose ;
from gz.msgs10.pose_v_pb2 import Pose_V ;
import time
import copy
import traceback


class Task():
    class Transform():
        def __init__(self,position:Tuple[float,float,float],euler_rotation:Tuple[int,int,int]):
            self.position = position
            orientation = Rotation.from_euler('xyz',euler_rotation,degrees=True).as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;
        def add_rotation(self, euler_rotation:Tuple[int,int,int]):
            rot_modifier = Rotation.from_euler('xyz',euler_rotation,degrees=True)
            current_orientation = Rotation.from_quat(self.orientation)
            orientation = current_orientation * rot_modifier
            orientation = orientation.as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;

    def __init__(self, task_name:str, start_points:List[Transform], **kwargs) -> None:
        self.task_name = task_name
        self.starting_points = start_points
        self.settings = kwargs
    def get_random_start(self)->Transform:
        indices = np.arange(len(self.starting_points))
        random_index = np.random.choice(indices)
        return self.starting_points[random_index]
    def get(self,key:str):
        return self.settings.get(key,None)
    
class TwistAction():
    def __init__(self, name, action):
        self.name = name
        self.raw_action = action
        self.action = Twist()
        self.action.linear.x = action[0]
        self.action.angular.z = action[1]

    def return_instruction(self):
        return self.action

    def to_string(self):
        return self.__str__()

    def __str__(self):
      return f"LinearVelocity={self.raw_action[0]}; AngularVelocity={self.raw_action[1]}"


class EnvironmentConfig():
    def __init__(self,
                 observation_shape:Tuple[int,int,int],
                 tasks:Dict[str,Task],
                 actions:List[TwistAction],
                 robot_name:str,
                 vehicle_prefix:str,
                 world_name:str,
                 camera_topic:str, 
                 debug = False) -> None:
        self.debug = debug

        ### ENVIRONMENT WRAPPER
        self.observation_shape = observation_shape
        self.tasks = tasks
        self.actions = actions
        
        ### ENVIRONMENT MANAGER
        self.robot_name = robot_name
        self.vehicle_prefix = vehicle_prefix
        self.world_name = world_name
        self.camera_topic = camera_topic


class CatchingRobotConfig(EnvironmentConfig):
    def __init__(self, 
                 observation_shape: Tuple[int, int, int], 
                 tasks: Dict[str, Task], 
                 actions: List[TwistAction], 
                 robot_name: str, 
                 vehicle_prefix: str, 
                 world_name: str, 
                 camera_topic: str, 
                 runner_action:TwistAction,
                 runner_start_positions:Task, 
                 debug = False) -> None:
        super().__init__(observation_shape, tasks, actions, robot_name, vehicle_prefix, world_name, camera_topic, debug)

        ### RUNNER SPECIFIC
        self.runner_action = runner_action
        self.runner_start_positions = runner_start_positions

    
class EnvironmentManager(Node):
    def __init__(self,env_config):
        self.init_node()
        self.mutex = Lock()

        self.env_config = env_config

        self.robot_name = self.env_config.robot_name
        self.vehicle_prefix = self.env_config.vehicle_prefix
        self.world_name = self.env_config.world_name

        self.step = 0
        self.last_obs_time = 0

        if self.subscribe(Image,f'{self.vehicle_prefix}/camera',self.gz_handle_observation_callback):
            print("subscribed to Camera!")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info!")

        self.gz_action = self.advertise(f'{self.vehicle_prefix}/motor',Twist)

        self.wait_for_simulation()

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()
        self.set_pose_service = f'{self.world_name}/set_pose'
        self.pos_req = Pose()
        self.observation_shape = None

    def init_node(self):
        super().__init__()

    def wait_for_simulation(self):
        response = self.request_scene()
        for m in response.model:
          print("Model in scene", m.name) ;
    
        while not hasattr(self, "data") or not hasattr(self, "position"):
            time.sleep(0.005)
        print('\nObservation and position received!')

    def request_scene(self):
        result = False;
        start_time = time.perf_counter()
        while result is False:
            # Request the scene information
            result, response = self.request(f'{self.world_name}/scene/info', Empty(), Empty, Scene, 1)
            print(f'\rWaiting for simulator... {(time.perf_counter() - start_time):.2f} sec', end='')
            time.sleep(0.005)
        print('\nScene received!')
        return response

    def get_step(self):
        return self.step

    def get_data(self):
        return self.data

    def get_position(self):
        return self.position

    def get_last_obs_time(self):
        return self.last_obs_time
    
    def convert_image_msg(self, msg):
        if self.observation_shape is None:
            self.observation_shape = (msg.height, msg.width, 3)
        return np.frombuffer(msg.data,dtype=np.uint8).astype(np.float32).reshape(self.observation_shape) / 255. ;

    def gz_handle_observation_callback(self,msg):
        with self.mutex:
            # print("OBS") ;
            self.data = msg
            self.last_obs_time = msg.header.stamp.sec + round(msg.header.stamp.nsec / 1e9, 3)

    def gz_handle_dynamic_pose_callback(self,msg):
        with self.mutex:
            for p in msg.pose:
                if p.name == self.robot_name:
                    self.position = [p.position.x,p.position.y,p.position.z];
                    return;
            print(f"THERE WAS NO\033[92m {self.robot_name}\033[0m IN THE SIMULATION!")

    def gz_perform_action(self, action:TwistAction):
        self.step += 1
        self.gz_action.publish(action.return_instruction())
        if self.env_config.debug=="yes": print(f'action published: ', action.to_string())

    def gz_perform_action_stop(self):
        action = TwistAction('stop', [0.0, 0.0])
        self.gz_action.publish(action.return_instruction())

    def gz_publish_new_scene(self):
        self.scene

    def perform_switch(self, task_index:str):
        pass

    def perform_reset(self, position, orientation):
        self.position = position
        self.set_entity_pose_request(self.robot_name,self.position,orientation)

    def set_entity_pose_request(self, name, position, orientation):
        self.pos_req.name = name
        self.pos_req.position.x = position[0]
        self.pos_req.position.y = position[1]
        self.pos_req.position.z = position[2]
        self.pos_req.orientation.x = orientation[0]
        self.pos_req.orientation.y = orientation[1]
        self.pos_req.orientation.z = orientation[2]
        self.pos_req.orientation.w = orientation[3]

        result = False ;
        while result == False:
          result, response = self.request(self.set_pose_service, self.pos_req, Pose, Boolean, 1) ;
          time.sleep(0.01)
          if self.env_config.debug=="yes": print(result, response.data)
          if response.data == True: break ;

    def trigger_pause(self, pause):
        self.res_req.pause = pause
        if self.env_config.debug=="yes": print(f'pause={pause} request !')

        result = False ;
        while result == False:
          result, response = self.request(self.world_control_service, self.res_req, WorldControl, Boolean, 1) ;
          time.sleep(0.01)
          if response.data == True: break ;
        if self.env_config.debug=="yes": print(f'pause={pause} request done!')


class CatchingRobotManager(EnvironmentManager):
    def __init__(self,env_config:CatchingRobotConfig):
        self.init_node()
        self.mutex = Lock()

        self.env_config = env_config

        self.robot_name = self.env_config.robot_name
        self.vehicle_prefix = self.env_config.vehicle_prefix
        self.world_name = self.env_config.world_name

        self.runner_action = env_config.runner_action
        self.runner_start_positions = env_config.runner_start_positions
        self.runner_name = next(iter(self.env_config.tasks.values())).task_name

        self.step = 0
        self.last_obs_time = 0

        if self.subscribe(Image,f'{self.vehicle_prefix}/camera',self.gz_handle_observation_callback):
            print("Subscribed to Camera!")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info! for catching robot")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_another_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info! for running robot")

        self.gz_action = self.advertise(f'{self.vehicle_prefix}/motor',Twist)

        self.gz_runner_actions = {}
        for task in env_config.tasks.values():
            self.gz_runner_actions[task.task_name] = self.advertise(f'/{task.task_name}/motor',Twist)

        #self.scene_publisher = self.advertise(f'{self.world_name}/default/scene', Scene) # possibly .../default/scene

        self.wait_for_simulation()

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()
        self.set_pose_service = f'{self.world_name}/set_pose'
        self.pos_req = Pose()
        self.observation_shape = None

    def get_runner_position(self):
        return self.runner_position
    
    def perform_reset(self, position, orientation):
        self.runner_position = self.runner_start_positions.get_random_start()
        self.set_entity_pose_request(self.runner_name,self.runner_position.position,self.runner_position.orientation)
        return super().perform_reset(position, orientation)
    
    def park_runner(self, task_id):
        parking_transform = self.env_config.tasks[task_id].get('parking')
        self.gz_stop_runner()
        self.set_entity_pose_request(self.runner_name,parking_transform.position,parking_transform.orientation)
        print(f"parked runner {self.runner_name} at {parking_transform.position}")
    
    def perform_switch(self,task_id:str):
        ### task_name has to coincide with the name of the robot in question
        if hasattr(self,'task_id'):
            self.park_runner(self.task_id)
        self.task_id = task_id
        self.runner_name = self.env_config.tasks[task_id].task_name
    
    def rotate_runner(self,transform:Task.Transform):
        tmp = copy.deepcopy(transform)
        angle = np.random.uniform(-60,60)
        tmp.add_rotation([0,0,int(angle)])
        self.set_entity_pose_request(self.runner_name,self.runner_position,tmp.orientation)
        print(f"redirected {self.runner_name} at {self.runner_position} by {tmp.orientation}")

    def gz_start_runner(self):
        self.gz_runner_actions[self.runner_name].publish(self.runner_action.return_instruction())
        if self.env_config.debug =="yes": print(f'runner_action published: ', self.runner_action.to_string())

    def gz_stop_runner(self):
        action = TwistAction('stop', [0.0, 0.0])
        self.gz_runner_actions[self.runner_name].publish(action.return_instruction())
        print(f'Stopping {self.runner_name}!')

    def gz_handle_another_dynamic_pose_callback(self, msg):
        with self.mutex:
            for p in msg.pose:
                if p.name == self.runner_name:
                    self.runner_position = [p.position.x,p.position.y,p.position.z];
                    return;
            print(f"THERE WAS NO\033[92m {self.runner_name}\033[0m IN THE SIMULATION!")


class PushingObject(RgEnv):
    OBJ_ID_LOOKUP = {"red":0, "green":1, "blue":2, "yellow":3, "pink":4, "cyan":5}

    def __init__(self, simulator, 
                 running_objective, 
                 action_space = None, 
                 observation_space = None,
                 task_list = ["red", "blue", "green", "yellow"]):
        print("simulator:", simulator)
        assert hasattr(simulator, "set_manager")

        super().__init__(simulator, 
                         running_objective, 
                         action_space, 
                         observation_space)
        tasks = {}
        tasks['red'] = Task('red',[Task.Transform(position=[0.4,0.0,0.05],
                                                  euler_rotation=[0.0,0.0,15.0]),
                                   Task.Transform(position=[0.4,0.0,0.05],
                                                  euler_rotation=[0.0,0.0,-15.0])],
                            mass=20)
        tasks['green'] = Task('green',[Task.Transform(position=[0.4,4.0,0.05],euler_rotation=[0.0,0.0,15.0]),
                                       Task.Transform(position=[0.4,4.0,0.05],euler_rotation=[0.0,0.0,-15.0])],
                              mass=20)
        tasks['blue'] = Task('blue',[Task.Transform(position=[0.4,8.0,0.05],euler_rotation=[0.0,0.0,15.0]),
                                     Task.Transform(position=[0.4,8.0,0.05],euler_rotation=[0.0,0.0,-15.0])],
                              mass=0)
        tasks['yellow'] = Task('yellow',[Task.Transform(position=[0.4,-4.0,0.05],euler_rotation=[0.0,0.0,15.0]),
                                         Task.Transform(position=[0.4,-4.0,0.05],euler_rotation=[0.0,0.0,-15.0])],
                               mass=0)
        tasks['pink'] = Task('pink',[Task.Transform(position=[0.4,-8.0,0.05],euler_rotation=[0.0,0.0,15.0]),
                                     Task.Transform(position=[0.4,-8.0,0.05],euler_rotation=[0.0,0.0,-15.0])],
                             mass=0)
        tasks['cyan'] = Task('cyan',[Task.Transform(position=[0.4,-12.0,0.05],euler_rotation=[0.0,0.0,15.0]),
                                     Task.Transform(position=[0.4,-12.0,0.05],euler_rotation=[0.0,0.0,-15.0])],
                             mass=20)

        env_config = EnvironmentConfig(observation_shape=[20,20,3], # simulator.system.dim_observation
                                       tasks=tasks,
                                       actions=self.action_space,
                                       robot_name='3pi_front_cam_robot',
                                       vehicle_prefix='/vehicle',
                                       world_name='/world/pushing_objects_world',
                                       camera_topic='/vehicle/camera',
                                       debug="no")
        self.task_list = task_list
        self.tasks = tasks
        self.info = dict()
        simulator.set_manager(env_config)
        time.sleep(0.01)

    def step(self, action):
        self.simulator.receive_action(
            self.simulator.system.apply_action_bounds(action.reshape(1, -1))
        )

        reward, truncated, terminated = self.running_objective(self._get_obs(), 
                                                              self._get_pos(), 
                                                              action.reshape(-1),
                                                              self.current_mass)
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), reward, truncated, sim_step is not None or terminated, {}

    def get_current_status(self):
        obj_name = self.info['object'][0]
        return (self.info['object'][1], self.OBJ_ID_LOOKUP[obj_name])
    
    def switch_task(self, task_index):
        self.task_id = self.task_list[task_index]
        self.simulator.manager.perform_switch(self.task_id)

    def _get_obs(self):
        return self.simulator.get_observation()
    
    def _get_pos(self):
        return np.array(self.simulator.manager.position)
    
    def _get_distance_to_object(self):
        object_positions = {
            'red':    [0.9,   0.0, 0.05],
            'green':  [0.9,   4.0, 0.05],
            'blue':   [0.9,   8.0, 0.05],
            'yellow': [0.9,  -4.0, 0.05],
            'pink':   [0.9,  -8.0, 0.05],
            'cyan':   [0.9, -12.0, 0.05],
        }

        robot_pos = self._get_pos()[:2]
        object_pos = np.array(object_positions[self.task_id][:2])
        dis = np.linalg.norm(robot_pos - object_pos)
        return dis

    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super(RgEnv, self).reset(seed=seed)
        
        current_task = self.tasks[self.task_id]
        self.current_name = current_task.task_name
        self.current_mass = current_task.get('mass')
        self.starting_transform = current_task.get_random_start()
        self.info['object'] = (self.current_name,self.current_mass)

        self.simulator.reset(current_task)
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), {}


class LineFollowing(RgEnv):
    def __init__(self, simulator, 
                 running_objective, 
                 action_space = None, 
                 observation_space = None,
                 task_list = ["circle_red", "circle_blue", "circle_green", "circle_yellow"]):
        
        print("simulator:", simulator)
        assert hasattr(simulator, "set_manager")

        super().__init__(simulator, 
                         running_objective, 
                         action_space, 
                         observation_space)
        tasks = {}
        tasks['circle_red'] = Task('circle_red',[Task.Transform(position=[-40.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),
                                                 Task.Transform(position=[-40.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_green'] = Task('circle_green',[Task.Transform(position=[-20.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),
                                                     Task.Transform(position=[-20.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_blue'] =  Task('circle_blue',[Task.Transform(position=[0.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),
                                                    Task.Transform(position=[0.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_yellow'] =Task('circle_yellow',[Task.Transform(position=[20.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),
                                                      Task.Transform(position=[20.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])
        tasks['circle_white'] = Task('circle_white',[Task.Transform(position=[40.0,0.0,0.05],euler_rotation=[0.0,0.0,25.0]),
                                                     Task.Transform(position=[40.0,0.0,0.05],euler_rotation=[0.0,0.0,-25.0])])


        env_config = EnvironmentConfig(observation_shape=[3*2, 50, 3], # simulator.system.dim_observation
                                       tasks=tasks,
                                       actions=self.action_space,
                                       robot_name='3pi_robot',
                                       vehicle_prefix='/vehicle',
                                       world_name='/world/race_tracks_world',
                                       camera_topic='/vehicle/camera')
        self.task_list = task_list
        self.tasks = tasks
        self.info = dict()
        simulator.set_manager(env_config)
        time.sleep(0.01)

    def step(self, action):
        self.simulator.receive_action(
            self.simulator.system.apply_action_bounds(action.reshape(1, -1))
        )

        reward, truncated, terminated = self.running_objective(self._get_state())
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), reward, truncated, sim_step is not None or terminated, {}

    def get_current_status(self):
        return (self.info['track'],)
    
    def switch_task(self, task_index):
        self.task_id = self.task_list[task_index]
        self.simulator.manager.perform_switch(self.task_id)
    
    def _get_state(self):
        return self.simulator.state
    
    def _get_obs(self):
        return self.simulator.get_observation()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super(RgEnv, self).reset(seed=seed)
        
        current_task = self.tasks[self.task_id]
        self.current_name = current_task.task_name
        self.starting_transform = current_task.get_random_start()
        self.info['track'] = self.current_name

        self.simulator.reset(current_task)
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), {}


class RobotPursuit(RgEnv):
    def __init__(self, simulator, 
                 running_objective, 
                 action_space = None, 
                 observation_space = None,
                 task_list = ["red_cube", "green_capsule", "blue_sphere", "yellow_cylinder"]):
        
        print("simulator:", simulator)
        assert hasattr(simulator, "set_manager")

        super().__init__(simulator, 
                         running_objective, 
                         action_space, 
                         observation_space)
        ## Possible Tasks (With either 15 or -15 rotation as starting point)
        tasks = {}
        tasks['red_cube']        = Task('red_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),
                                                            Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],
                                        parking=Task.Transform([-30,-1,-0.25],[0,0,0]))
        tasks['green_capsule']   = Task('green_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),
                                                              Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],
                                        parking=Task.Transform([-30,-2,-0.25],[0,0,0]))
        tasks['blue_sphere']     = Task('blue_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),
                                                             Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],
                                        parking=Task.Transform([-30,-3,-0.25],[0,0,0]))
        tasks['yellow_cylinder'] = Task('yellow_runner_robot',[Task.Transform([-1.2,0.0,0.05],[0.0,0.0,15.0]),
                                                               Task.Transform([-1.2,0.0,0.05],[0.0,0.0,-15.0])],
                                        parking=Task.Transform([-30,-4,-0.25],[0,0,0]))
        
        self.cardinal_directions = {
            'north' : Task.Transform([0.0,0.0,0.0],[0,0,0]),
            'east' : Task.Transform([0.0,0.0,0.0],[0,0,90]),
            'south' : Task.Transform([0.0,0.0,0.0],[0,0,180]),
            'west' : Task.Transform([0.0,0.0,0.0],[0,0,270])
        }

        ### Depending on what png the ground_plane uses
        tiny_arena = [-2.5,2.5,-2.5,2.5]
        small_arena = [-5,5,-5,5]
        big_arena = [-10,10,-10,10]

        self.arena_bounds = tiny_arena # TODO: Argparse!
        self.runner_collision_thickness = 0.42 # TODO: Argparse!

        self.task_list = task_list
        self.tasks = tasks
        self.info = dict()

        runner_start_positions = Task('runner_start',[Task.Transform([0.0,0.0,-0.25],[0.0,0.0,60.0]),
                                                      Task.Transform([0.0,0.0,-0.25],[0.0,0.0,-60.0])])

        env_config = CatchingRobotConfig(
            observation_shape=[20, 20, 3],
            tasks=tasks,
            actions=self.action_space,
            robot_name='catcher_robot',
            vehicle_prefix='/vehicle',
            world_name='/world/catching_robot_world',
            camera_topic='/vehicle/camera',
            runner_action=TwistAction('forward',[0.25, 0]),
            runner_start_positions=runner_start_positions,
            debug=False
        )

        simulator.set_cardinal_directions(self.cardinal_directions)
        simulator.set_manager(env_config)
        simulator.set_arena_bounds(self.arena_bounds)

        time.sleep(0.01)

    def step(self, action):
        self.simulator.receive_action(
            self.simulator.system.apply_action_bounds(action.reshape(1, -1))
        )

        reward, truncated, terminated = self.running_objective(
            np.copy(self.simulator.state), 
            self._get_robot_pos(), 
            self._get_runner_pos(), 
            action, 
            self.simulator.step_count, 
            self.simulator.max_step_per_episode,
            self.runner_collision_thickness,
            self.arena_bounds,
            self.info
        )
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), reward, truncated, sim_step is not None or terminated, {}

    def get_current_status(self):
        return (self.info['terminate_cond'],)
    
    def switch_task(self, task_index):
        self.task_id = self.task_list[task_index]
        print("receive self.task_id:", self.task_id)
        self.simulator.manager.perform_switch(self.task_id)
    
    def _get_state(self):
        return self.simulator.state
    
    def _get_obs(self):
        return self.simulator.get_observation()

    def _get_robot_pos(self):
        return np.array(self.simulator.manager.get_position())
    
    def _get_runner_pos(self):
        return np.array(self.simulator.manager.get_runner_position())
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super(RgEnv, self).reset(seed=seed)
        
        current_task = self.tasks[self.task_id]
        self.simulator.reset(current_task)
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs().reshape(-1), {}
