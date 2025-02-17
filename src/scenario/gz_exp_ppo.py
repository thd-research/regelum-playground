from src.scenario.ppo import PPOScenario
from src.environment import PushingObject, LineFollowing, RobotPursuit

from pathlib import Path
import torch
import os
import numpy as np
from copy import copy


class PPOScenarioWrapper(PPOScenario):
    def __init__(self, simulator, 
                 running_objective, 
                 device: str = "cuda:0",
                 total_timesteps: int = 1000000,
                 gamma: float = 0.99,
                 policy_lr: float = 3.0e-4,
                 anneal_lr: bool = True,
                 num_steps: int = 2048,
                 gae_lambda: float = 0.95,
                 update_epoch: int = 10,
                 num_minibatches: int = 32,
                 clip_coef: float = 0.2,
                 norm_adv: bool = True,
                 clip_vloss: bool = True,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.0,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 checkpoint_dirpath = None,
                 env = ...,
                 seed = 42,
                 **kwargs):
        
        if env == "PushingObject":
            chosen_env = PushingObject
        elif env == "LineFollowing":
            chosen_env = LineFollowing
        elif env == "RobotPursuit":
            chosen_env = RobotPursuit
        else:
            raise Exception("Environment should be specified.")


        super().__init__(simulator, 
                         running_objective, 
                         device = device,
                         total_timesteps = total_timesteps,
                         gamma = gamma,
                         policy_lr = policy_lr,
                         anneal_lr = anneal_lr,
                         num_steps = num_steps,
                         gae_lambda = gae_lambda,
                         update_epoch = update_epoch,
                         num_minibatches = num_minibatches,
                         clip_coef = clip_coef,
                         norm_adv = norm_adv,
                         clip_vloss = clip_vloss,
                         vf_coef = vf_coef,
                         ent_coef = ent_coef,
                         max_grad_norm = max_grad_norm,
                         target_kl = target_kl,
                         seed=seed,
                         env=chosen_env)
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
                self.total_timesteps = int(1e6)
                self.next_iter_max = self.evaluation_episode_number + self.iteration_id - 1
                super().run(check_learning_start=False, buffer_update=False)
        else:
            for train_id, task_name in enumerate(task_list):
                self.phase = "train"

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
                    total_timesteps_backup = copy(self.total_timesteps)
                    self.next_iter_max = self.evaluation_episode_number + self.iteration_id - 1
                    super().run(check_learning_start=False, buffer_update=False)

                    self.total_timesteps = total_timesteps_backup

    def meet_stop_condition(self):
        if self.phase == "eval":
            return self.iteration_id > self.next_iter_max if hasattr(self, "next_iter_max") else True
        else:
            return False
    
    def load_checkpoint(self, experiment_path):
        load_nn_model(self.agent, "agent", experiment_path)

    def save_checkpoint(self, id):
        save_nn_model(self.agent, f"agent_{id}")

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