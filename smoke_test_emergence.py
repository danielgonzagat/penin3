#!/usr/bin/env python3
import time
from rl_env import NonStationaryBanditEnv
from agent_behavior_learner import AgentBehaviorLearner
from emergence_metrics import EmergenceMetrics


async def main():
    env = NonStationaryBanditEnv(num_arms=5)
    learner = AgentBehaviorLearner(actions=env.action_names)
    metrics = EmergenceMetrics()

    start = time.time()
    steps = 0
    rewards = []

    while time.time() - start < 10:  # 10s smoke test
        state = env.get_state()
        action = learner.choose_action(state)
        reward, next_state = env.step(action)
        learner.learn(state, action, reward, next_state)
        learner.decay_exploration()
        rewards.append(reward)

        # feed patterns to metrics
        metrics.record_pattern(f"t={steps} a={action} r={round(reward,3)} avg20={state['avg_reward_20']}")
        if steps % 25 == 0:
            m = metrics.compute()
        steps += 1

    m = metrics.compute()
    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
    logger.info("SMOKE TEST RESULT")
    logger.info(f"steps={steps} avg_reward={avg_r:.3f}")
    logger.info(f"metrics: novelty={m['novelty_ratio']:.3f} entropy_bits={m['entropy_bits']:.3f} comp_ratio={m['compression_ratio']:.3f} mi={m['structure_mi']:.3f}")


if __name__ == "__main__":
    main()
