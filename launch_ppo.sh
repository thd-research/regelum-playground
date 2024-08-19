run_experiment_with_seed() {
    local seed="$1"
    python3.10 run.py \
    +seed=$seed \
    --single-thread \
    --parallel \
    simulator=casadi \
    scenario=ppo_scenario \
    system=3wrobot_kin_customized \
    --experiment=ppo_3wrobot_kin_test \
    scenario.N_episodes=5 \
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
}

for seed in {1..10}; do
  run_experiment_with_seed "$seed" 
done
