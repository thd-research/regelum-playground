CUDA_VISIBLE_DEVICES="" python3 run.py \
    scenario=sac \
    system=pendulum_with_gym_observation \
    simulator=casadi_random_state_init \
    scenario.autotune=False \
    scenario.policy_lr=0.00079 \
    scenario.q_lr=0.00025 \
    scenario.alpha=0.0085 \
    +seed=4 \
    --interactive \
    --fps=10