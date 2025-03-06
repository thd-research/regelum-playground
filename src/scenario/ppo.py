"""This file contains the implementation of the Soft Actor-Critic (SAC) algorithm.
The implementation is based on the CleanRL repository:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

We have adapted the original implementation to work with the regelum framework.
Key changes include:
1. Using mlflow for logging instead of Weights & Biases and tensorboard.
2. Utilizing regelum's callback features for smooth logging of metrics and trajectories.
3. Integrating with regelum's Simulator and RunningObjective classes.

The file structure is as follows:
1. Import statements
2. SoftQNetwork class definition
3. Actor class definition
4. SACScenario class definition (which inherits from CleanRLScenario)

The SACScenario class contains the main logic for the SAC algorithm, including:
- Initialization of networks, optimizers, and replay buffer
- Training loop
- Evaluation and logging functions

This implementation allows for easy integration with regelum's ecosystem while
maintaining the core SAC algorithm structure from CleanRL.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from src.rgenv import RgEnv
from torch.distributions.normal import Normal
from regelum.simulator import Simulator
from regelum.objective import RunningObjective
import gymnasium as gym
import mlflow
from .base import CleanRLScenario
import time

from copy import copy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(
        self,
        dim_action: int,
        dim_observation: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(dim_observation, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(dim_observation, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, dim_action), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, dim_action))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean
    

class PPOScenario(CleanRLScenario):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        device: str = "cuda:0",
        total_timesteps: int = 1000000,
        gamma: float = 0.99,
        policy_lr: float = 3.0e-4,
        anneal_lr: bool = True,
        num_steps: int = 2048,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        num_minibatches: int = 32,
        clip_coef: float = 0.2,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: float = None,
        seed: int = 42,
        env: RgEnv = RgEnv,
    ):
        """
        Initializes the Proximity Policy Optimization (PPO) scenario.

        Args:
            simulator: The simulator object.
            running_objective: The running objective for the scenario.
            device: The device to run the computations on.
            total_timesteps: Total number of timesteps for the scenario.
            gamma: Discount factor for future rewards.
            policy_lr: Learning rate for the policy network.
            anneal_lr: Toggle learning rate annealing for policy and value networks
            num_steps: The number of steps to run in each environment per policy rollout
            gae_lambda: The lambda for the general advantage estimation
            update_epochs: the K epochs to update the policy
            num_minibatches: the number of mini-batches
            clip_coef: the surrogate clipping coefficient
            norm_adv: Toggles advantages normalization
            clip_vloss: Toggles whether or not to use a clipped loss for the value function, as per the paper.
            vf_coef: coefficient of the value function
            ent_coef: coefficient of the entropy
            max_grad_norm: the maximum norm for the gradient clipping
            target_kl: the target KL divergence threshold
            env: 
        """
        super().__init__(
            simulator=simulator,
            running_objective=running_objective,
            total_timesteps=total_timesteps,
            device=device,
            env=env,
        )
        self.gamma = gamma
        self.policy_lr = policy_lr

        self.num_envs = 1
        # PPO
        self.batch_size = int(self.num_envs * num_steps)
        self.num_steps = num_steps
        self.num_iterations = total_timesteps // self.batch_size
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.anneal_lr = anneal_lr
        
        self.seed = seed

        self.dim_action, self.dim_observation, self.action_bounds = (
            simulator.system._dim_inputs,
            simulator.system._dim_observation,
            np.array(simulator.system._action_bounds),
        )
        
        self.agent = Actor(self.dim_action, self.dim_observation).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters()
                                    , lr=self.policy_lr
                                    , eps=1e-5)

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
            "task_name": self.task_name if hasattr(self, "task_name") else ""
        }
    
    def meet_stop_condition(self):
        return False
    
    def run(self, check_learning_start=True, buffer_update=True):
        start_debug = True
        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + (self.dim_observation,)).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + (self.dim_action,)).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        for iteration in range(1, self.num_iterations + 1):
            if self.meet_stop_condition():
                break

            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.policy_lr
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, action_mean = self.agent.get_action_and_value(next_obs)

                    # if self.phase == "eval":
                    #     action = copy(action_mean)

                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                self.state = self.envs.envs[0].env.state.reshape(1, -1)
                self.time = self.envs.envs[0].env.simulator.time
        
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())

                # before calling the step method
                self.post_compute_action(
                    self.state,
                    next_obs,
                    action,
                    float(reward),
                    self.time,
                    global_step,
                )

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                # We need state and time for logging, so we extracted it 2 lines above
                if "final_info" in infos:
                    is_episode_end = False
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            self.save_episodic_return(
                                global_step=global_step, episodic_return=info["episode"]["r"]
                            )
                            self.reload_scenario()
                            self.reset_episode()
                            self.reset_iteration()
                            is_episode_end = True
                            break
                    
                    if self.phase == "eval" and is_episode_end:
                        break 

            # ALGO LOGIC: training.
            if  self.phase == "train" :
                # bootstrap value if not done
                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                # flatten the batch
                b_obs = obs.reshape((-1,) + (self.dim_observation,))
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + (self.dim_action,))
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(self.batch_size)
                clipfracs = []
                
                for epoch in range(self.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, self.batch_size, self.minibatch_size):
                        end = start + self.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue, action_mean = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if self.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -self.clip_coef,
                                self.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break
                
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                self.save_losses(
                        global_step=global_step,
                        entropy_loss=entropy_loss.mean().item(),
                        v_loss=v_loss.mean().item(),
                        actor_loss=loss.mean().item(),
                        pg_loss=pg_loss.mean().item(),
                        explained_var=explained_var.mean().item(),
                    )
        
        self.reload_scenario()
        self.reset_episode()
        self.reset_iteration()
        self.envs.close()
        print("Env closed")
