from src.scenario.sac import SACScenario
from src.environment import PushingObject

class PushingObjectSACScenario(SACScenario):
    def __init__(self, simulator, running_objective, device = "cuda:0", total_timesteps = 1000000, buffer_size = 1000000, gamma = 0.99, tau = 0.005, batch_size = 256, learning_starts = 5000, policy_lr = 0.0003, q_lr = 0.001, policy_frequency = 2, target_network_frequency = 1, alpha = 0.2, autotune = True, env = ...):
        
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
                         PushingObject)

    def run(self):
        self.envs.envs[0].env.switch_task(0)
        return super().run()
    
    def reset_episode(self):
        
        return super().reset_episode()