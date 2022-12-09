import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    '''register custom env'''
    from gym.envs.registration import registry, register, make, spec
    noisy_max_episode_steps= [500,500,500,200,1000]
    gameidx = 0
    for game in ["AcrobotEnv", "CartPoleEnv", "MountainCarEnv", "PendulumEnv","LunarLander" ]:
        
        register(id='noisy-{}-v1'.format(game), 
                entry_point='noisyGymClassicControl:'+game,
                max_episode_steps=noisy_max_episode_steps[gameidx]
                )
        gameidx +=1

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    obs = env.reset()
    print("obs",obs)
    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic
    print("deterministic",deterministic)
    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    nepoch = 10000
    try:
        # custom_obs = th.tensor([[  -0.329,  -0.091]]).to(model.device)
        # custom_obs = th.tensor([[  -0.445,  0.159]]).to(model.device)
        # custom_obs = th.tensor([[  -0.0261,  -0.2984]]).to(model.device)
        # custom_obs = th.tensor([[  -0.0444,  -0.3171]]).to(model.device)
        custom_obs = th.tensor([[  -0.1179,  -0.3024]]).to(model.device)
        
        custom_action = 0*th.ones(1).to(model.device)
        _, log_prob, _ = model.policy.evaluate_actions( custom_obs , custom_action)
        print("prob",th.exp(log_prob).detach())
        target_log_prob = log_prob.detach()
        sign =  1
        advantages = sign* th.ones(1).to(model.device)
        y = th.sign(advantages)
        policy_loss_fn = th.nn.MarginRankingLoss(margin=0.1)
        for epoch in range(nepoch):
            _, log_prob, _ = model.policy.evaluate_actions( custom_obs , custom_action)
            print("prob",th.exp(log_prob).detach())
            x1 = th.exp(log_prob - target_log_prob) # ratio
            x2 = th.ones_like(x1.clone().detach())
            loss = policy_loss_fn( x1 .unsqueeze(0) , x2 .unsqueeze(0) , (y   ) .unsqueeze(0) ) 
            print(loss)
            if loss.item() == 0:
                break
            model.policy.optimizer.zero_grad()
            loss.backward()
            ## Clip grad norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            model.policy.optimizer.step()
            

        import numpy as np
        import matplotlib
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        
        matplotlib.use('TkAgg')
        # import sys
        # xmin, xmax = -1.2 ,0.6
        # ymin, ymax = -0.07 ,0.07
        xmin, xmax = -3 ,3
        ymin, ymax = -3 ,3
        stepsnum = 200
        xs = th.linspace(xmin, xmax, steps=stepsnum)
        ys = th.linspace(ymin, ymax, steps=stepsnum)
        obs = th.cartesian_prod(xs,ys).to(model.device)
        # x,y = torch.meshgrid(xs, ys, indexing='ij')
        # print("obs",obs)
        # actions, values, log_prob = model.policy.forward(obs)
        for tempa in range(3):
            # tempa = 0
            
            # action = th.tensor([0,0,0,0]).to(model.device)
            action = tempa*th.ones(stepsnum*stepsnum).to(model.device)
            _, log_prob, _ = model.policy.evaluate_actions( obs , action)
            # print("s",obs,"pi[s,a]" , th.exp(log_prob))
            prob = th.exp(log_prob).detach()
            prob = th.reshape(prob,(stepsnum,stepsnum)).transpose(0, 1).cpu()
            # print("prob",prob)
            fig = plt.figure(figsize=(10,10))
            np.set_printoptions(threshold=sys.maxsize)
            ax = fig.add_subplot(111)
            ax.set_xlabel('s[0] Car Position(-1.2,0.6) ')
            ax.set_ylabel('s[1] Car Velocity(-0.07,0.07) ')
            # ax.set_aspect('equal', adjustable='box')
            # plt.axis('square')
            plt.imshow( prob , origin='lower', interpolation='none', extent=[xmin,xmax,ymin,ymax])
            # plt.xticks(x)
            # plt.yticks(y)
            # ax.set_aspect(aspect=10)
            plt.colorbar()  
            # plt.show()
            plt.savefig('./plot/mountaincar_policy/afterUpdate_obs{obs}_sign{sign}_action{ac}_algo-{alg}_expid{eid}_loadbest_{loadbest}.png'.format(obs=custom_obs.cpu().numpy()[0][0],sign=sign ,ac=tempa,alg=args.algo,eid=args.exp_id,loadbest=args.load_best) ,format='png' )
        # x = np.arange(xmin, xmax, 0.01)
        # y = np.arange(ymin, ymax, 0.01)
        # x, y = np.meshgrid(x, y)
        # print(x,y)
        # for 
        # for _ in range(args.n_timesteps):
        # print("obs",obs)
        # actions, values, log_prob = model.policy.forward(th.tensor(obs).to(model.device))
        # print("s",obs,"pi[s,a]",obs,actions.item(), th.exp(log_prob).item() )
        # print("s",obs,"pi[s,a]" , th.exp(log_prob).item() )
        # print("s",obs,"pi[s,a]" , th.exp(log_prob))
        

        # action, state = model.predict(obs, state=state, deterministic=deterministic)
        # print("action",action)
        # if args.algo == "hpo" and model.independent_value_net == True:
        #     next_q_values, _ , _ = model.value_policy.evaluate_actions(th.tensor(obs).to(model.device), th.tensor(action).to(model.device))
        #     print("q:",next_q_values )
        # action = []
        # obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render("human")

        # episode_reward += reward[0]
        ep_len += 1

        # if args.n_envs == 1:
        #     # For atari the return reward is not the atari score
        #     # so we have to get it from the infos dict
        #     if is_atari and infos is not None and args.verbose >= 1:
        #         episode_infos = infos[0].get("episode")
        #         if episode_infos is not None:
        #             print(f"Atari Episode Score: {episode_infos['r']:.2f}")
        #             print("Atari Episode Length", episode_infos["l"])

        #     if done and not is_atari and args.verbose > 0:
        #         # NOTE: for env using VecNormalize, the mean reward
        #         # is a normalized reward when `--norm_reward` flag is passed
        #         print(f"Episode Reward: {episode_reward:.2f}")
        #         print("Episode Length", ep_len)
        #         episode_rewards.append(episode_reward)
        #         episode_lengths.append(ep_len)
        #         episode_reward = 0.0
        #         ep_len = 0
        #         state = None

        #     # Reset also when the goal is achieved when using HER
        #     if done and infos[0].get("is_success") is not None:
        #         if args.verbose > 1:
        #             print("Success?", infos[0].get("is_success", False))

        #         if infos[0].get("is_success") is not None:
        #             successes.append(infos[0].get("is_success", False))
        #             episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    # if args.verbose > 0 and len(successes) > 0:
    #     print(f"Success rate: {100 * np.mean(successes):.2f}%")

    # if args.verbose > 0 and len(episode_rewards) > 0:
    #     print(f"{len(episode_rewards)} Episodes")
    #     print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    # if args.verbose > 0 and len(episode_lengths) > 0:
    #     print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    main()