run_experiment_with_seed() {
    local seed="$1"
    python3.10 run.py \
    +seed=$seed \
    simulator=casadi \
    scenario=ppo_scenario \
    system=3wrobot_kin_customized \
    --experiment=ppo_3wrobot_kin_stored_weights \
    scenario.N_episodes=10 \
    scenario.N_iterations=180 \
    scenario.policy_n_epochs=30 \
    scenario.critic_n_epochs=30 \
    scenario.policy_opt_method_kwargs.lr=0.0005 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario.policy_model.std=0.1 \
    scenario.critic_model.n_hidden_layers=2 \
    scenario.critic_model.dim_hidden=15 \
    scenario.critic_opt_method_kwargs.lr=0.1 \
    scenario.gae_lambda=0.96 \
    simulator.time_final=10 \
    common.sampling_time=0.1
}

for seed in {1..10}; do
  run_experiment_with_seed "$seed"
done
