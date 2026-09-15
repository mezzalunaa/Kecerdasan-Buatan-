# Example 3.23 CartPole reinforcement learning
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        print('Example 3.23 requires the optional gymnasium or gym package.')
        gym = None

if gym is not None:
    try:
        environment = gym.make('CartPole-v1')
    except Exception as error:
        print('CartPole could not be created:', error)
    else:
        for episode in range(20):
            reset_result = environment.reset()
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            for timestep in range(100):
                action = environment.action_space.sample()
                step_result = environment.step(action)
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    observation, reward, done, info = step_result
                if done:
                    print(f'Episode {episode + 1} finished after {timestep + 1} timesteps')
                    break
        environment.close()
