from src.scenario.sac import SACScenario
from src.scenario.td3 import TD3Scenario
from src.environment import PushingObject, LineFollowing, RobotPursuit

from pathlib import Path
import torch
import os


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
                 env = ...):
        
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

        if checkpoint_dirpath is not None:
            self.checkpoint_dirpath = checkpoint_dirpath

    def run(self):
        if hasattr(self, "checkpoint_dirpath"):
            self.load_checkpoint(self.checkpoint_dirpath)
        
        for id, task_info in enumerate(self.envs.envs[0].env.task_list):
            print("task_info:", task_info)

            # reset replay buffer
            if self.reset_rb_each_task:
                self.rb.reset()

            self.envs.envs[0].env.switch_task(id)
            super().run()

        self.save_checkpoint()
    
    def load_checkpoint(self, experiment_path):
        load_nn_model(self.actor, "actor", experiment_path)
        load_nn_model(self.qf1, "qf1", experiment_path)
        load_nn_model(self.qf2, "qf2", experiment_path)
        load_nn_model(self.qf1_target, "qf1_target", experiment_path)
        load_nn_model(self.qf2_target, "qf2_target", experiment_path)

    def save_checkpoint(self):
        save_nn_model(self.actor, "actor")
        save_nn_model(self.qf1, "qf1")
        save_nn_model(self.qf2, "qf2")
        save_nn_model(self.qf1_target, "qf1_target")
        save_nn_model(self.qf2_target, "qf2_target")


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
                 env=...):
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
                         env)
        
        self.reset_rb_each_task = reset_rb_each_task

        if checkpoint_dirpath is not None:
            self.checkpoint_dirpath = checkpoint_dirpath

    def run(self):
        if hasattr(self, "checkpoint_dirpath"):
            self.load_checkpoint(self.checkpoint_dirpath)
        
        for id, task_info in enumerate(self.envs.envs[0].env.task_list):
            print("task_info:", task_info)

            # reset replay buffer
            if self.reset_rb_each_task:
                self.rb.reset()

            self.envs.envs[0].env.switch_task(id)
            super().run()

        self.save_checkpoint()

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


class PushingObjectSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, env=...):
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
                         PushingObject)


class LineFollowingSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, env=...):
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
                         LineFollowing)


class RobotPursuitSACScenario(SACScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=5000, policy_lr=0.0003, q_lr=0.001, policy_frequency=2, target_network_frequency=1, alpha=0.2, autotune=True, reset_rb_each_task=False, checkpoint_dirpath=None, env=...):
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
                        RobotPursuit)


class PushingObjectTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...):
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
            PushingObject
        )


class LineFollowingTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...):
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
            LineFollowing
        )


class RobotPursuitTD3Scenario(TD3ScenarioWrapper):
    def __init__(self, simulator, running_objective, device="cuda:0", total_timesteps=1000000, buffer_size=1000000, gamma=0.99, tau=0.005, batch_size=256, learning_starts=25000, policy_frequency=2, noise_clip=0.5, exploration_noise=0.1, learning_rate=0.0003, policy_noise=0.2,
                 reset_rb_each_task=False, 
                 checkpoint_dirpath=None,
                 env=...):
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
            RobotPursuit
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
