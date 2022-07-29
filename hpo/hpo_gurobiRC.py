import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
import gym
from torch.nn import functional as F
import torch.nn as nn
from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, polyak_update, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

import math

from hpo.policies import ActorCriticPolicy2
# from hpo.buffer import RolloutBuffer2
from stable_baselines3.common.policies import BaseModel, BasePolicy
import random
import time
from gurobipy import *

class HPO(OnPolicyAlgorithm):
    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy2]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        policy_base: Type[BasePolicy] = ActorCriticPolicy2,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        # classifier: int =0,
        classifier: str="AM",
        aece: str="WAE",
        entropy_hpo: bool =False,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        alpha: float = 0.1,
        rgamma: float = 0.0,
    ):

        super(HPO, self).__init__(
            policy,
            env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            # classifier=classifier,#super is for  inherit class ,inherit class does not have classifier
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.classifier = classifier
        self.aece = aece
        self.alpha = alpha
        self.rgamma = rgamma
        self.entropy_hpo = entropy_hpo
        # self.robust_delta_y = self.ROBUSTDELTAY()
        if _init_setup_model:
            self._setup_model()
        
    # class ROBUSTDELTAY(nn.Module):
    #     def __init__(self):
    #         super(ROBUSTDELTAY, self).__init__()
    #         # batch_size = 64
    #         gamma = 10
    #         self.affine1 = nn.Linear( 1 , 32)
    #         self.affine2 = nn.Linear(32, 32)
    #         self.affine3 = nn.Linear(64, 32)
    #         self.deltaY_head = nn.Linear(32, batch_size)
    #     def forward(self, x):
    #         x = self.affine1(x)
    #         x = self.affine2(x)
    #         deltaY = F.softmax(self.deltaY_head(x), dim=-1) * gamma
    #         return deltaY
    
    def _setup_model(self) -> None:
        super(HPO, self)._setup_model()
        
        # self.target_policy = self.policy_class(  # pytype:disable=not-instantiable
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     use_sde=self.use_sde,
        #     **self.policy_kwargs  # pytype:disable=not-instantiable
        # )
        # self.target_policy = self.target_policy.to(self.device)
        # self.target_policy.load_state_dict(self.policy.state_dict())
        
        # self.value_policy = self.policy_class(  # pytype:disable=not-instantiable
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     use_sde=self.use_sde,
        #     **self.policy_kwargs  # pytype:disable=not-instantiable
        # )
        # self.value_policy = self.value_policy.to(self.device)
        # print("self.value_policy",self.value_policy)
        # print("self.target_policy",self.target_policy)
        # print("self.policy",self.policy)
        # envname = self.env.unwrapped.spec.id
        
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        # rollout_buffer: ReplayBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # print("envname",env)
        n_steps = 0
        # exploration_rate = 0.1
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        # dones = False
        # self._last_obs = env.reset()
        rollout_time = []
        computeV_time = []

        t_rollout_start = time.time()
        while n_steps < n_rollout_steps :
            # print("n_steps:",n_steps,"n_rollout_steps: ",n_rollout_steps)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #actions, values, log_probs = self.policy.forward(obs_tensor) # org
                #print("collect rollout forward")
                actions, _, log_probs = self.policy.forward(obs_tensor)
            
            # print("n_steps: ",n_steps,"n_rollout_steps: ",n_rollout_steps)
            # if exploration_rate > random.random():
            #     # print("random choose action")
            #     actions = np.array([self.action_space.sample()])
            # else:
                # actions = actions.cpu().numpy()
            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            ''' wrong advantage from flipped rewards?'''
            # print("infos:",infos)
            # print("state: ",self._last_obs," next state: ",new_obs," rewards: ",rewards," dones: ",dones,"actions",clipped_actions)
            # Compute value
            t_computeV_start = time.time()
            values = th.Tensor(np.zeros_like(actions, dtype=float)).to(self.device)
            batch_actions = np.zeros_like(actions)
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # new_obs_tensor = obs_as_tensor(new_obs, self.device)
                new_obs_tensor = obs_as_tensor(self._last_obs, self.device)
                for a in range(self.action_space.n):
                    batch_actions = np.full(batch_actions.shape, a)
                    #print(batch_actions)
                    # print("collect_rollouts self.policy.evaluate_actions")
                    next_q_values, next_log_probs, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                    # print("print(next_q_values.shape, next_log_probs.shape) ",next_q_values.shape, next_log_probs.shape) 4,1
                    #print(rewards.shape)
                    #print(next_q_values[:,a])
                    exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                    # exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a])
                    #print(exp_q_values)
                    #print(rewards)
                    # values += th.Tensor(rewards).to(self.device) + exp_q_values
                    values += exp_q_values
                    #print(values)
            #values = th.Tensor(rewards).to(self.device) + (th.exp(next_log_probs) * next_q_values)
            #values = th.Tensor(rewards).to(self.device) + (th.exp(next_log_probs) * next_q_values)
            #print(values.shape)
            t_computeV_end = time.time()
            computeV_time.append(t_computeV_end - t_computeV_start)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            # if dones:
            # # if dones.any():
            #     new_obs = env.reset()
            #     print("env reset")
                # continue
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        values = th.Tensor(np.zeros_like(actions.reshape(-1), dtype=float)).to(self.device)
        batch_actions = np.zeros_like(actions.reshape(-1))
        with th.no_grad():
                # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            #_, values, _ = self.policy.forward(obs_tensor)
            for a in range(self.action_space.n):
                batch_actions = np.full(batch_actions.shape, a)
                #print(batch_actions)
                # print("collect_rollouts last timestep self.policy.evaluate_actions")
                next_q_values, next_log_probs, _ = self.policy.evaluate_actions(obs_tensor, th.from_numpy(batch_actions).to(self.device))
                #print(next_q_values.shape, next_log_probs.shape)
                # exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                #print(exp_q_values)
                values += exp_q_values
        t_rollout_end = time.time()
        rollout_time.append(t_rollout_end - t_rollout_start)
        logger.record("Time/collect_rollout/Sum", np.sum(rollout_time))
        logger.record("Time/collect_computeV/Sum", np.sum(computeV_time))
        logger.record("Time/collect_rollout/Mean", np.mean(rollout_time))
        logger.record("Time/collect_computeV/Mean", np.mean(computeV_time))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # for step in reversed(range(rollout_buffer.buffer_size)):
        #     if step == rollout_buffer.buffer_size - 1:
        #         next_non_terminal = 1.0 - self._last_episode_starts 
        #         rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.gamma * values.clone().cpu().numpy().flatten()
        #     else:
        #         next_non_terminal = 1.0 - rollout_buffer.episode_starts[step + 1]
        #         rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.returns[step + 1]
        #         #rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.gamma * rollout_buffer.values[step + 1]
        callback.on_rollout_end()
        # print("collect rollout self.ep_info_buffer",self.ep_info_buffer)
        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # print("train self.ep_info_buffer",self.ep_info_buffer)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        #clip_fractions = []
        margins = []
        #positive_a = []
        #negative_a = []

        positive_p = []
        negative_p = []
        ratio_p = []
        rollout_return = []

        epoch_time = []
        loss_time = []
        computeV_time = []
        t_action_adv_time = []
        #alpha = 0.1
        # if args.algo == 'HPO':
        #     print("args.algo == 'HPO' can use in hpg.py")
        # print("custom_hyperparams: ",self.custom_hyperparams)
        # print("in hpg.py def train: self .classifier",self.classifier)
        print("in hpg.py def train: self .aece",self.aece)
        print("self.rgamma",self.rgamma)
        # Hard update for target policy -6~8
        #polyak_update(self.policy.parameters(), self.target_policy.parameters(), 1.0)
        # print("self.batch_size: ",self.batch_size)
        # train for n_epochs epochs
        t_epoch_start = time.time()
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            # Hard update for target policy - 9
            #polyak_update(self.policy.parameters(), self.target_policy.parameters(), 1.0)
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # print("len(rollout_data): ",len(rollout_data))
                
                actions = rollout_data.actions
                # print("actions: ",actions)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                #print(actions)
                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                
                batch_actions = np.zeros(self.batch_size)
                batch_values = np.zeros(self.batch_size)

                # val_q_values = th.zeros(self.batch_size, requires_grad=True).to(self.device)
                # val_q_values = th.zeros(self.batch_size ).to(self.device)
                val_q_values = []
                #tmp_values = th.zeros(self.batch_size, requires_grad=True).to(self.device)
                #val_log_prob = th.zeros(self.batch_size).to(self.device)
                advantages = th.zeros(self.batch_size).to(self.device)
                #advantages = np.zeros(self.batch_size)

                action_advantages = []
                action_probs = []
                #action_log_probs = []
                #action_q_values = []
                positive_adv_prob = np.zeros(self.batch_size)
                negative_adv_prob = np.zeros(self.batch_size)
                # positive_adv_prob = 0
                # negative_adv_prob = 0
                # Q-value
                minMu = np.ones(self.batch_size)
                epsilon = np.zeros(self.batch_size)
                # old_p = th.exp(rollout_data.old_log_prob).detach()
                # for i in range(len(old_p)):
                #     minMu = min(old_p[i],minMu)
                # print("train self.policy.evaluate_actions real actions")
                # action_q_values, val_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                action_q_values, val_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                t_computeV_start = time.time()
                minMu = th.zeros_like(val_log_prob).cpu().detach()
                gpu_zero_batchsize =  th.zeros_like(val_log_prob)
                gpu_positive_adv_prob = th.zeros_like(val_log_prob)
                gpu_negative_adv_prob = th.zeros_like(val_log_prob)
                gpu_batch_values  = th.zeros_like(val_log_prob)
                gpu_action_advantages = []
                gpu_action_probs = []
                for a in range(self.action_space.n):
                    # print("action", a, batch_actions)
                    batch_actions = np.full(batch_actions.shape,a)
                    # print("train self.policy.evaluate_actions batch actions")
                    q_values, a_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    #q_values, a_log_prob, _ = self.target_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    #_, a_log_prob, _ = self.target_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    #q_values, _, _ = self.value_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    gpu_q =  q_values[:,a].flatten()
                    # q = gpu_q.cpu().detach()
                    gpu_p = th.exp(a_log_prob)
                    gpu_action_probs.append(gpu_p)
                    
                    p = gpu_p.cpu().detach()
                    gpu_batch_values += gpu_q*gpu_p
                    # batch_values += (p*q).numpy()
                    minMu = th.minimum(p,minMu)
                    # action_advantages.append(advantages.cpu())
                    gpu_action_advantages.append(gpu_q)
                    # action_advantages.append(q)
                    # action_q_values.append(q)
                    # action_probs.append(p)
                    #action_log_probs.append(a_log_prob)
                    #print("val_values: ", val_values)
                t_computeV_end = time.time()
                computeV_time.append(t_computeV_end - t_computeV_start)   
                # print("batch_values", batch_values)
                # print("action_advantages",action_advantages)
                # action_advantages = action_advantages.cpu()
                t_action_adv_start = time.time()
                # for j in range(self.batch_size):
                #     minMu[j]  = th.min(action_probs,dim=0)
                
                for a in range(self.action_space.n):
                    # print("action_advantages shape",action_advantages[i].shape)
                    #print("Before A", action_advantages[i])
                    # action_advantages[a] -= batch_values
                    gpu_action_advantages[a] -= gpu_batch_values
                    gpu_positive_adv_prob = th.where(gpu_action_advantages[a]>0,gpu_action_probs[a] +gpu_positive_adv_prob,gpu_positive_adv_prob)
                    gpu_negative_adv_prob = th.where(gpu_action_advantages[a]<0,gpu_action_probs[a] +gpu_negative_adv_prob,gpu_negative_adv_prob)
                    
                    # action_advantages.append(gpu_action_advantages[a].cpu().clone().detach().numpy())
                    # print("gpu_action_advantages[a]",gpu_action_advantages[a])
                    
                    #print("After A", action_advantages[i])
                    # for j in range(self.batch_size):
                    #     # minMu[j] = min(action_probs[a][j].clone().detach().numpy(),minMu[j])
                    #     # minMu[j] = min(action_probs[a][j],minMu[j])
                    #     if action_advantages[a][j] > 0:
                    #         positive_adv_prob[j] += action_probs[a][j] 
                    #     elif action_advantages[a][j] < 0:
                    #         negative_adv_prob[j] += action_probs[a][j] 
                        # if a == actions[j]:
                        #     #val_log_prob[j] = action_log_probs[i][j]
                        #     #val_q_values[j] = action_advantages[i][j] + batch_values[j]
                        #     val_q_values[j] = action_q_values[j][a] 
                        #     advantages[j] = action_advantages[a][j]
                    #print("val_log_prob: ", val_log_prob)
                positive_adv_prob = gpu_positive_adv_prob.cpu().clone().detach().numpy()
                negative_adv_prob = gpu_negative_adv_prob.cpu().clone().detach().numpy()
                # print("action_advantages",action_advantages)
                # action_advantages[:] = gpu_action_advantages[:].cpu().clone().detach().numpy()
                for j in range(self.batch_size):
                    # val_q_values[j] = action_q_values[j][ actions[j] ].clone()
                    val_q_values.append(action_q_values[j][ actions[j] ])
                    advantages[j] = gpu_action_advantages[ actions[j] ][j]
                # val_q_values = action_q_values
                t_action_adv_end = time.time()
                t_action_adv_time.append(t_action_adv_end - t_action_adv_start) 
                # HPO: max(0, epsilon - weight_a (ratio - 1))
                #      max(0, margin - y * (x1 - x2))
                if self.classifier == "AM":
                    x1 = th.exp(val_log_prob - rollout_data.old_log_prob.detach()) # ratio
                    x2 = th.ones_like(x1.clone().detach())
                elif self.classifier == "AM-log":# log(pi) - log(mu)
                    x1 = val_log_prob
                    x2 = rollout_data.old_log_prob
                elif self.classifier == "AM-root":# root: (pi/mu)^(1/2) - 1
                    x1 = th.sqrt(th.exp(val_log_prob - rollout_data.old_log_prob.detach())) # ratio
                    x2 = th.ones_like(x1.clone().detach())
                elif self.classifier == "AM-sub":
                    x1 = th.exp(val_log_prob )
                    x2 = th.exp(rollout_data.old_log_prob)
                elif self.classifier == "AM-square":
                    x1 = th.square(th.exp(val_log_prob - rollout_data.old_log_prob.detach())) # ratio
                    x2 = th.ones_like(x1.clone().detach())
                #advantages = rollout_data.advantages.cpu().detach()
                #print("advantages",advantages)
                #abs_adv = np.abs(advantages.cpu())
                advantages = advantages.detach()
                y = th.sign(advantages)
                if self.aece == "WAE" or self.aece == "WCE":
                    # y = advantages
                    abs_adv = th.abs(advantages)
                else:
                    abs_adv = th.abs(y)
                    # y = y
                # abs_adv = y*advantages
                # abs_adv = th.abs(advantages)
                # y = advantages / abs_adv
                #y = rollout_data.advantages / abs_adv
                #y = advantages / self.batch_size
                #y = advantages

                #adv_prob = th.exp(log_prob.clone()).cpu().detach()
                #adv_positive = advantages.cpu() > 0
                #adv_negative = advantages.cpu() < 0
                ## Forward action --> check all actions
                #pos_adv_prob = 0.0
                #neg_adv_prob = 0.0
                #for i in range(len(adv_prob)):
                #    if adv_positive[i]:
                #        pos_adv_prob += adv_prob[i]
                #    if adv_negative[i]:
                #        neg_adv_prob += adv_prob[i]
                #print(pos_adv_prob.cpu().detach())
                positive_p.append(positive_adv_prob)
                negative_p.append(negative_adv_prob)
                
                #epsilon = alpha * min(1, adv_negative / adv_positive) * self.batch_size
                #epsilon = alpha * min(1, adv_negative / adv_positive)
                #epsilon = alpha * min(1, negative_adv_prob / (positive_adv_prob + 1e-8))
                prob_ratio = negative_adv_prob / (positive_adv_prob + 1e-8)
                #print("Prob Ratio", prob_ratio)
                ratio_p.append(prob_ratio)

                #epsilon = np.zeros(self.batch_size)
                # ratio version      max(0, margin - y * (x1 - x2))
                #epsilon = alpha * min(1, prob_ratio)
                # log version 
                #epsilon = math.log(1 + alpha * min(1, prob_ratio))
                # root: (pi/mu)^(1/2) - 1
                # epsilon = math.sqrt(1 + alpha * min(1, prob_ratio)) - 1
                t_loss_start = time.time()
                deltaYnum = int(self.batch_size* self.rgamma ) 
                # print("deltaYnum",deltaYnum)
                policy_losses2 = []
                # for key ,value in od:
                #     # print("k,v",key,value)
                #     if deltaYcounter>= deltaYnum:
                #         break
                #     deltaYs[key] = 1
                #     deltaYcounter+=1
                SVM = Model("robust_hpo")
                NUM_DATA = self.batch_size
                M = 1000000000
                ksi = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS)
                phi = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS)
                q = SVM.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
                r = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS)
                s = SVM.addVars(range(NUM_DATA), vtype=GRB.BINARY )
                t = SVM.addVars(range(NUM_DATA), vtype=GRB.BINARY )
                pitheta = SVM.addVars(range(NUM_DATA), vtype=GRB.CONTINUOUS,lb=0.0,ub=1.0)
                SVM.modelSense = GRB.MINIMIZE
                SVM.Params.outputFlag = 0
                SVM.addConstr(q >=0 )
                # gamma = int(NUM_DATA*args.rgamma )
                old_probs = th.exp(rollout_data.old_log_prob).cpu().clone().detach().numpy()
                ys_grb = y.cpu().clone().detach().numpy()
                for i in range(self.batch_size):
                    if self.classifier == "AM":
                        epsilon[i] = self.alpha * min(1, prob_ratio[i])
                    elif self.classifier == "AM-log":
                        epsilon[i] = math.log(1 + self.alpha * min(1, prob_ratio[i]))
                    elif self.classifier == "AM-root":
                        epsilon[i] = math.sqrt(1 + self.alpha * min(1, prob_ratio[i])) - 1
                    elif self.classifier == "AM-sub":
                        epsilon[i] = minMu[i] * self.alpha * min(1, prob_ratio[i])
                    elif self.classifier == "AM-square":
                        epsilon[i] = ( 1 + self.alpha * min(1, prob_ratio[i]) )** 2
                    if self.aece == "WAE" or self.aece == "AE":
                        policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon[i])
                        epsilon_for_gurobi = epsilon[i]
                    elif self.aece == "WCE" or self.aece == "CE":
                        policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha)
                        epsilon_for_gurobi = self.alpha
                    # policy_loss += abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) )
                    # policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) ))
                    index = i
                    SVM.addConstr(q+r[index]>=phi[index]-ksi[index])
                    
                    if self.classifier == "AM":
                        SVM.addConstr(ksi[index] >= epsilon_for_gurobi - ys_grb[index]* ( pitheta[index]/(old_probs[i]+1e-8) - 1) )
                        SVM.addConstr(ksi[index] <= epsilon_for_gurobi - ys_grb[index]* ( pitheta[index]/(old_probs[i]+1e-8) - 1) +M*(1-s[index]))
                        SVM.addConstr(ksi[index] <= M*(s[index]) )
                        SVM.addConstr(phi[index] >= epsilon_for_gurobi + ys_grb[index]* ( pitheta[index]/(old_probs[i]+1e-8) - 1) )
                        SVM.addConstr(phi[index] <= epsilon_for_gurobi + ys_grb[index]* ( pitheta[index]/(old_probs[i]+1e-8) - 1) +M*(1-t[index]))
                        SVM.addConstr(phi[index] <= M*(t[index]) )
                        SVM.addConstr(r[index] >=0 )
                        SVM.addConstr(ksi[index] >=0 )
                        SVM.addConstr(phi[index] >=0 )
                        SVM.addConstr(pitheta[index] >=0 )
                
                SVM.setObjective( quicksum(ksi[i] for i in range(NUM_DATA) ) + deltaYnum*q + quicksum(r[i] for i in range(NUM_DATA) ) )
                SVM.optimize()
                if SVM.status != GRB.OPTIMAL:
                    print("SVM is not optimal")
                else:
                    
                    for inner_epoch in range( 100 ):
                        mimic_losses = []
                        action_q_values, val_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                        for i in range(self.batch_size):
                            ''' mse loss between th.exp(val_log_prob) - pitheta[index].x with 100 gd '''
                            mloss = th.nn.MSELoss()
                            # print("th.exp(val_log_prob[i])",th.exp(val_log_prob[i]))
                            # print("pitheta[i].x",pitheta[i].x)
                            mimic_loss = mloss(th.exp(val_log_prob[i]), th.tensor(pitheta[i].x ).to(self.device)  )
                            mimic_losses.append( mimic_loss )
                        # self.policy.optimizer.zero_grad()
                        policy_loss = th.stack(mimic_losses).sum() /self.batch_size
                        # policy_loss.backward()
                        # self.policy.optimizer.step()
                        if inner_epoch == 0 :
                            margins.append(epsilon)
                            
                            val_q_values = th.stack(val_q_values)
                            if self.clip_range_vf is None:
                                # No clipping
                                #values_pred = val_values # org version
                                values_pred = val_q_values # org version
                                #values_pred = th.exp(val_log_prob) * val_values
                            else:
                                # Clip the different between old and new value
                                # NOTE: this depends on the reward scaling
                                values_pred = rollout_data.old_values + th.clamp(
                                    val_values - rollout_data.old_values, -clip_range_vf, clip_range_vf # org version
                                    #th.exp(val_log_probs) * val_values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                                )
                            
                            value_loss = F.mse_loss(rollout_data.returns, values_pred )
                            value_losses.append(value_loss.item())

                            rollout_return.append(rollout_data.returns.detach().cpu().numpy())
                        # entropy = None
                        # Entropy loss favor exploration
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -th.mean(-val_log_prob)
                        else:
                            entropy_loss = -th.mean(entropy)

                        entropy_losses.append(entropy_loss.item())

                        # org version
                        # print("policy_loss.requires_grad:",policy_loss.requires_grad)
                        if self.entropy_hpo == True and inner_epoch == 0 :
                            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss ##0810 test ,hope to find 360 HPO-62
                        elif self.entropy_hpo == False and inner_epoch == 0 :
                            loss = policy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                        else:
                            loss =  policy_loss
                        # loss = policy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                        #loss = policy_loss
                        #loss = th.stack(policy_loss).sum() + self.vf_coef * value_loss

                        ## Optimization step
                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        ## Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                    t_loss_end = time.time()
                    loss_time.append(t_loss_end - t_loss_start)
                #approx_kl_divs.append(th.mean(rollout_data.old_log_prob - val_log_prob).detach().cpu().numpy())
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob.detach() - val_log_prob).detach().cpu().numpy())
                
                ## value policy
                #value_loss = self.vf_coef * value_loss
                #self.value_policy.optimizer.zero_grad()
                #value_loss.backward()
                ### Clip grad norm
                #th.nn.utils.clip_grad_norm_(self.value_policy.parameters(), self.max_grad_norm)
                #self.value_policy.optimizer.step()

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break
            #if np.mean(pg_losses) < 1e-4:
            #    print(f"Early stopping at step {epoch} due to reaching ploss: {np.mean(pg_losses):.2f}")
            #    break
            # th.cuda.empty_cache()
        t_epoch_end = time.time()
        epoch_time.append(t_epoch_end - t_epoch_start)
        logger.record("Time/train_loss/Sum", np.sum(loss_time))
        logger.record("Time/train_computeV/Sum", np.sum(computeV_time))
        logger.record("Time/train_epoch/Sum", np.sum(epoch_time))
        logger.record("Time/train_action_adv/Sum", np.sum(t_action_adv_time))
        logger.record("Time/train_loss/Mean", np.mean(loss_time))
        logger.record("Time/train_computeV/Mean", np.mean(computeV_time))
        logger.record("Time/train_epoch/Mean", np.mean(epoch_time))
        logger.record("Time/train_action_adv/Mean", np.mean(t_action_adv_time))
        
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())


        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        #logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

        # HPO
        logger.record("HPO/margin", np.mean(margins))
        #logger.record("HPO/positive_advantage", np.mean(positive_a))
        #logger.record("HPO/negative_advantage", np.mean(negative_a))
        logger.record("HPO/positive_advantage_prob", np.mean(positive_p))
        logger.record("HPO/negative_advantage_prob", np.mean(negative_p))
        logger.record("HPO/prob_ratio", np.mean(ratio_p))
        logger.record("HPO/rollout_return", np.mean(rollout_return))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "HPO":
        
        return super(HPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
