for seed in {1..20}; do
    python3.10 run.py \
                +seed=$seed \
                simulator=ros_qcar \
                scenario=ppo_scenario_qcar \
                system=qcar_kin \
                --experiment=ppo_qcar_kin \
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
                scenario.discount_factor=0.99 \
                scenario.cliprange=0.2 \
                scenario.critic_td_n=1 \
                simulator.time_final=30 \
                common.sampling_time=0.1 \
                --interactive
done