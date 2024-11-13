from src.scenario.sac import SACScenario
from src.scenario.td3 import TD3Scenario
from src.environment import PushingObject, LineFollowing, RobotPursuit

from pathlib import Path
import torch
import os
import numpy as np
from copy import copy


class SACScenarioWrapper(SACScenario):
    def __init__(self, simulator, 
                 running_objective, 
                 device = "cuda:0", 
                 total_timesteps = 1000000, 
                 buffer_size = 1000000, 
                 gamma = 0.99, 
                 tau = 0.005, 
                 batch_size = 256, 
                 learning_starts = 5000, 
                 policy_lr = 0.0003, 
                 q_lr = 0.001, 
                 policy_frequency = 2, 
                 target_network_frequency = 1, 
                 alpha = 0.2, 
                 autotune = True, 
                 reset_rb_each_task = False,
                 checkpoint_dirpath = None,
                 env = ...,
                 **kwargs):
        
        super().__init__(simulator, 
                         running_objective, 
                         device, 
                         total_timesteps, 
                         buffer_size, 
                         gamma, 
                         tau, 
                         batch_size, 
                         learning_starts, 
                         policy_lr, 
                         q_lr, 
                         policy_frequency, 
                         target_network_frequency, 
                         alpha, 
                         autotune, 
                         env)
        self.reset_rb_each_task = reset_rb_each_task
        self.evaluation_episode_number = int(kwargs.get("evaluation_episode_number", "30"))
        self.eval_only = bool(int(kwargs.get("evaluation_only", False)))

        if checkpoint_dirpath is not None:
            self.checkpoint_dirpath = checkpoint_dirpath

    def run(self):
        if hasattr(self, "checkpoint_dirpath"):
            print("Model Loaded", self.checkpoint_dirpath)
            self.load_checkpoint(self.checkpoint_dirpath)
        
        task_list = self.envs.envs[0].env.task_list
        if self.eval_only:
            self.phase = "eval"
            for eval_id, eval_task_info in enumerate(task_list):
                print("Eval task_info:", eval_task_info)

                self.task_name = eval_task_info
                self.envs.envs[0].env.switch_task(eval_id)

                # check_learning_start=True -> use policy to update action at the beginning
                # set total_timesteps and learning_start as inf to prevent actor from gradient descent
                self.learning_starts = self.total_timesteps = int(1e6)
                self.next_iter_max = self.evaluation_episode_number + self.iteration_id - 1
                super().run(check_learning_start=False, buffer_update=False)
        else:
            
            for train_id, task_name in enumerate(task_list):
                self.phase = "train"
                # reset replay buffer
                if self.reset_rb_each_task:
                    self.rb.reset()

                self.task_name = task_name
                self.envs.envs[0].env.switch_task(train_id)
                super().run()

                self.save_checkpoint(train_id)

                self.phase = "eval"
                for eval_id, eval_task_info in enumerate(self.envs.envs[0].env.task_list):
                    if eval_id > train_id:
                        break

                    print("Eval task_info:", eval_task_info, eval_id, train_id)

                    self.task_name = eval_task_info
                    self.envs.envs[0].env.switch_task(eval_id)

                    # check_learning_start=True -> use policy to update action at the beginning
                    # set total_timesteps and learning_start as inf to prevent actor from gradient descent
                    learning_starts_backup = copy(self.learning_starts)
                    total_timesteps_backup = copy(self.total_timesteps)
                    self.learning_starts = self.total_timesteps = int(1e6)
                    self.next_iter_max = self.evaluation_episode_number + self.iteration_id - 1
                    super().run(check_learning_start=False, buffer_update=False)

                    self.learning_starts = learning_starts_backup
                    self.total_timesteps = total_timesteps_backup

    def meet_stop_condition(self):
        if self.phase == "eval":
            return self.iteration_id > self.next_iter_max if hasattr(self, "next_iter_max") else True
        else:
            return False
    
    def load_checkpoint(self, experiment_path):
        load_nn_model(self.actor, "actor", experiment_path)
        load_nn_model(self.qf1, "qf1", experiment_path)
        load_nn_model(self.qf2, "qf2", experiment_path)
        load_nn_model(self.qf1_target, "qf1_target", experiment_path)
        load_nn_model(self.qf2_target, "qf2_target", experiment_path)

    def save_checkpoint(self, id):
        save_nn_model(self.actor, f"actor_{id}")
        save_nn_model(self.qf1, f"qf1_{id}")
        save_nn_model(self.qf2, f"qf2_{id}")
        save_nn_model(self.qf1_target, f"qf1_target_{id}")
        save_nn_model(self.qf2_target, f"qf2_target_{id}")
    
        save_nn_model(self.actor, "actor")
        save_nn_model(self.qf1, "qf1")
        save_nn_model(self.qf2, "qf2")
        save_nn_model(self.qf1_target, "qf1_target")
        save_nn_model(self.qf2_target, "qf2_target")

    @apply_callbacks()
    def post_compute_action(self, state, obs, action, reward, time, global_step):
        self.current_running_objective = reward
        self.value += reward
        return {
            "estimated_state": state,
            "observation": obs,
            "time": time,
            "episode_id": self.episode_id,
            "iteration_id": self.iteration_id,
            "step_id": global_step,
            "action": action,
            "running_objective": reward,
            "current_value": None,
            "current_undiscounted_value": self.value,
            "task_name": self.task_name if hasattr(self, "task_name") else "",
            "phase": self.phase,
            "exploration": self.exploration if hasattr(self, "exploration") else False,
            "robot_position": self.simulator.manager.get_position(),
        }

class TD3ScenarioWrapper(TD3Scenario):
    def __init__(self, 
                 simulator, 
                 running_objective, 
                 device = "cuda:0", 
                 total_timesteps = 1000000, 
                 buffer_size = 1000000, 
                 gamma = 0.99, 
                 tau = 0.005, 
                 batch_size = 256, 
                 learning_starts = 25000, 
                 policy_frequency = 2, 
                 noise_clip = 0.5, 
                 exploration_noise = 0.1, 
                 learning_rate = 0.0003, 
                 policy_noise = 0.2,
                 reset_rb_each_task = False,
                 checkpoint_dirpath = None,
                 env=...,
                 **kwargs):
        super().__init__(simulator, 
                         running_objective, 
                         device, 
                         total_timesteps, 
                         buffer_size, 
                         gamma, 
                         tau, 
                         batch_size, 
                         learning_starts, 
                         policy_frequency, 
                         noise_clip, 
                         exploration_noise, 
                         learning_rate, 
                         policy_noise,
                         env,
                         **kwargs)
        
        self.reset_rb_each_task = reset_rb_each_task
        self.evaluation_episode_number = int(kwargs.get("evaluation_episode_number", "3"))
        self.eval_only = bool(int(kwargs.get("evaluation_only", False)))

        if checkpoint_dirpath is not None:
            self.checkpoint_dirpath = checkpoint_dirpath

    def meet_stop_condition(self):
        if self.phase == "eval":
            return self.iteration_id > self.next_iter_max if hasattr(self, "next_iter_max") else True
        else:
            return False

    def run(self):
        if hasattr(self, "checkpoint_dirpath"):
            self.load_checkpoint(self.checkpoint_dirpath)
        
        if not self.eval_only:
            self.phase = "train"

            for id, task_info in enumerate(self.envs.envs[0].env.task_list):
                print("task_info:", task_info)

                # reset replay buffer
                if self.reset_rb_each_task:
                    self.rb.reset()

                self.task_name = task_info
                self.envs.envs[0].env.switch_task(id)
                super().run()

            self.save_checkpoint()

        self.phase = "eval"
        for id, task_info in enumerate(self.envs.envs[0].env.task_list):
            print("task_info:", task_info)

            self.task_name = task_info
            self.envs.envs[0].env.switch_task(id)

            # check_learning_start=True -> use policy to update action at the beginning
            # set total_timesteps and learning_start as inf to prevent actor from gradient descent
            self.learning_starts = self.total_timesteps = int(1e6)
            self.next_iter_max = self.evaluation_episode_number + self.iteration_id
            super().run(check_learning_start=False)

    def load_checkpoint(self, experiment_path):
        load_nn_model(self.actor, "actor", experiment_path)
        load_nn_model(self.actor_target, "actor_target", experiment_path)
        load_nn_model(self.qf1, "qf1", experiment_path)
        load_nn_model(self.qf2, "qf2", experiment_path)
        load_nn_model(self.qf1_target, "qf1_target", experiment_path)
        load_nn_model(self.qf2_target, "qf2_target", experiment_path)

    def save_checkpoint(self):
        save_nn_model(self.actor, "actor")
        save_nn_model(self.actor_target, "actor_target")
        save_nn_model(self.qf1, "qf1")
        save_nn_model(self.qf2, "qf2")
        save_nn_model(self.qf1_target, "qf1_target")
        save_nn_model(self.qf2_target, "qf2_target")

    @apply_callbacks()
    def post_compute_action(self, state, obs, action, reward, time, global_step):
        self.current_running_objective = reward
        self.value += reward
        return {
            "estimated_state": state,
            "observation": obs,
            "time": time,
            "episode_id": self.episode_id,
            "iteration_id": self.iteration_id,
            "step_id": global_step,
            "action": action,
            "running_objective": reward,
            "current_value": None,
            "current_undiscounted_value": self.value,
            "task_name": self.task_name if hasattr(self, "task_name") else "",
            "phase": self.phase,
            "exploration": self.exploration if hasattr(self, "exploration") else False,
        }

class PushingObjectSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, 
                 env=..., **kwargs):
        super().__init__(simulator, 
                         running_objective, 
                         device, 
                         total_timesteps, 
                         buffer_size, 
                         gamma, 
                         tau, 
                         batch_size, 
                         learning_starts, 
                         policy_lr, 
                         q_lr, 
                         policy_frequency, 
                         target_network_frequency, 
                         alpha, 
                         autotune, 
                         reset_rb_each_task, 
                         checkpoint_dirpath, 
                         PushingObject,
                         **kwargs)


class LineFollowingSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, 
                 env=..., **kwargs):
        super().__init__(simulator, 
                         running_objective, 
                         device, 
                         total_timesteps, 
                         buffer_size, 
                         gamma, 
                         tau, 
                         batch_size, 
                         learning_starts, 
                         policy_lr, 
                         q_lr, 
                         policy_frequency, 
                         target_network_frequency, 
                         alpha, 
                         autotune, 
                         reset_rb_each_task, 
                         checkpoint_dirpath, 
                         LineFollowing,
                         **kwargs)


class RobotPursuitSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, 
                 env=..., **kwargs):
        super().__init__(simulator, 
                        running_objective, 
                        device, 
                        total_timesteps, 
                        buffer_size, 
                        gamma, 
                        tau, 
                        batch_size, 
                        learning_starts, 
                        policy_lr, 
                        q_lr, 
                        policy_frequency, 
                        target_network_frequency, 
                        alpha, 
                        autotune, 
                        reset_rb_each_task, 
                        checkpoint_dirpath, 
                        RobotPursuit,
                        **kwargs)


class PushingObjectTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...,
                 **kwargs):
        super().__init__(
            simulator, 
            running_objective, 
            device, 
            total_timesteps, 
            buffer_size, 
            gamma, 
            tau, 
            batch_size, 
            learning_starts, 
            policy_frequency, 
            noise_clip, 
            exploration_noise, 
            learning_rate, 
            policy_noise, 
            reset_rb_each_task,
            checkpoint_dirpath,
            PushingObject,
            **kwargs
        )


class LineFollowingTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...,
                 **kwargs):
        super().__init__(
            simulator, 
            running_objective, 
            device, 
            total_timesteps, 
            buffer_size, 
            gamma, 
            tau, 
            batch_size, 
            learning_starts, 
            policy_frequency, 
            noise_clip, 
            exploration_noise, 
            learning_rate, 
            policy_noise, 
            reset_rb_each_task, 
            checkpoint_dirpath,
            LineFollowing,
            **kwargs
        )


class RobotPursuitTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...,
                 **kwargs):
        super().__init__(
            simulator, 
            running_objective, 
            device, 
            total_timesteps, 
            buffer_size, 
            gamma, 
            tau, 
            batch_size, 
            learning_starts, 
            policy_frequency, 
            noise_clip, 
            exploration_noise, 
            learning_rate, 
            policy_noise, 
            reset_rb_each_task, 
            checkpoint_dirpath,
            RobotPursuit,
            **kwargs
        )


def save_nn_model(
    torch_nn_module: torch.nn.Module,
    name: str,
) -> None:
    os.makedirs(".checkpoint", exist_ok=True)
    torch.save(
        torch_nn_module.state_dict(),
        Path(".checkpoint")
        / name,
    )

def load_nn_model(
    torch_nn_module: torch.nn.Module,
    name: str,
    experiment_path: str
) -> None:
    checkpoint_path = Path(experiment_path) / ".checkpoint" / name
    torch_nn_module.load_state_dict(torch.load(checkpoint_path))
