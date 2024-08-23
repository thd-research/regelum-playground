if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then  
        python3.10 run.py \
                    +seed=7 \
                    simulator=ros \
                    scenario=ppo_scenario \
                    system=3wrobot_kin_customized \
                    --experiment=ppo_3wrobot_kin \
                    scenario.N_episodes=1 \
                    scenario.N_iterations=1 \
                    scenario.policy_n_epochs=50 \
                    scenario.critic_n_epochs=50 \
                    scenario.policy_opt_method_kwargs.lr=0.005 \
                    scenario.policy_model.n_hidden_layers=2 \
                    scenario.policy_model.dim_hidden=15 \
                    scenario.policy_model.std=0.1 \
                    scenario.critic_model.n_hidden_layers=3 \
                    scenario.critic_model.dim_hidden=15 \
                    scenario.critic_opt_method_kwargs.lr=0.1 \
                    scenario.gae_lambda=0 \
                    scenario.discount_factor=0.7 \
                    scenario.cliprange=0.2 \
                    scenario.critic_td_n=1 \
                    simulator.time_final=50 \
                    common.sampling_time=0.1 \
                    scenario.policy_checkpoint_path="/regelum-playground/regelum_data/outputs/2024-08-19/15-23-09/0/.callbacks/PolicyModelSaver/model_it_00238" \
                    scenario.critic_checkpoint_path="/regelum-playground/regelum_data/outputs/2024-08-19/15-23-09/0/.callbacks/CriticModelSaver/model_it_00238" \
                    simulator.use_phy_robot=false \
                    --interactive
    else
        python3.10 run.py \
                    +seed=7 \
                    simulator=casadi \
                    scenario=ppo_scenario \
                    system=3wrobot_kin_customized \
                    --experiment=ppo_3wrobot_kin \
                    scenario.N_episodes=1 \
                    scenario.N_iterations=280 \
                    scenario.policy_n_epochs=50 \
                    scenario.critic_n_epochs=50 \
                    scenario.policy_opt_method_kwargs.lr=0.005 \
                    scenario.policy_model.n_hidden_layers=2 \
                    scenario.policy_model.dim_hidden=15 \
                    scenario.policy_model.std=0.1 \
                    scenario.critic_model.n_hidden_layers=3 \
                    scenario.critic_model.dim_hidden=15 \
                    scenario.critic_opt_method_kwargs.lr=0.1 \
                    scenario.gae_lambda=0 \
                    scenario.discount_factor=0.7 \
                    scenario.cliprange=0.2 \
                    scenario.critic_td_n=1 \
                    simulator.time_final=50 \
                    common.sampling_time=0.1
fi