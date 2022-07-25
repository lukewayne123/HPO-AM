import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
import gym
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, polyak_update, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

import math

from nhpo.policies import ActorCriticPolicy2
# from hpo.buffer import RolloutBuffer2
from stable_baselines3.common.policies import BaseModel, BasePolicy
import random

import time

class NeuralHPO(OnPolicyAlgorithm):
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
        #aece: str="WAE",
        aece: str="CE",
        entropy_hpo: bool =False,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        alpha: float = 0.1,
        temperature: float = 1.0,
        #K: int = 10,
        K: int = 1,
        EMDAstep: float = 1e-3,
    ):

        super(NeuralHPO, self).__init__(
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
        self.temperature = temperature
        self.K = K
        self.EMDA_step = EMDAstep
        self.entropy_hpo = entropy_hpo
        if _init_setup_model:
            self._setup_model()
        
        # check time
        #self.rollout_time = []
        #self.computeV_time = []

    def _setup_model(self) -> None:
        super(NeuralHPO, self)._setup_model()
        #self.EMDA_policy = self.policy_class(  # pytype:disable=not-instantiable
        #    self.observation_space,
        #    self.action_space,
        #    self.lr_schedule,
        #    use_sde=self.use_sde,
        #    **self.policy_kwargs  # pytype:disable=not-instantiable
        #)
        #self.EMDA_policy = self.target_policy.to(self.device)
        #self.EMDA_policy.load_state_dict(self.policy.state_dict())
        
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
                actions, _, log_probs, energy_function = self.policy.forward(obs_tensor, temperature=self.temperature)
                #print(log_probs, energy_function)
            #_ = input("")
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
                    next_q_values, next_log_probs, _, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device), temperature=self.temperature)
                    #print("print(next_q_values.shape, next_log_probs.shape) ",next_q_values.shape, next_log_probs.shape)
                    #print(rewards.shape)
                    #print(next_q_values[:,a])
                    #exp_q_values = (th.exp(next_log_probs[:,a]) * next_q_values[:,a]).clone().detach()
                    exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                    # exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a])
                    #print(exp_q_values)
                    #print(rewards)
                    # values += th.Tensor(rewards).to(self.device) + exp_q_values
                    values += exp_q_values
                    #print(values)
            t_computeV_end = time.time()
            computeV_time.append(t_computeV_end - t_computeV_start)
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
            #rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, energy_funtion)
            #rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, energy_function)
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
                next_q_values, next_log_probs, _, _ = self.policy.evaluate_actions(obs_tensor, th.from_numpy(batch_actions).to(self.device), temperature=self.temperature)
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
        energy_losses = []
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
        # print("in hpg.py def train: self .classifier",self.classifier)
        #print("in nhpo.py def train: self .aece",self.aece)
        # train for n_epochs epochs
        t_epoch_start = time.time()
        #random_rollout_actions = [np.random.randint(self.action_space.n) for _ in range(len(self.rollout_buffer.observations))]
        #_, _, _, replay_energy_function_ = self.policy.evaluate_actions(self.rollout_buffer.observations, random_rollout_actions, self.temperature)
        #replay_probs = []
        #for a in range(self.action_space.n):
        #    batch_actions = np.full(len((self.rollout_buffer.observations),),a)
        #    _, rollout_log_prob, _, _ = self.policy.evaluate_actions(th.from_numpy(self.rollout_buffer.observations).to(self.device), th.from_numpy(batch_actions).to(self.device), self.temperature)
        #    replay_probs.append(rollout_log_prob)
        # generate data first
        # record old prob
        epochs_rollout_data = []
        epochs_replay_probs = []
        for _ in range(self.n_epochs):
            epoch_rollout_data = []
            epoch_replay_probs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                epoch_rollout_data.append(rollout_data)
                replay_probs = []
                for a in range(self.action_space.n):
                    batch_actions = np.full(len((rollout_data.observations),), a)
                    _, rollout_log_prob, _, _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device), self.temperature)
                    replay_probs.append(rollout_log_prob)
                epoch_replay_probs.append(replay_probs)
            epochs_rollout_data.append(epoch_rollout_data)
            epochs_replay_probs.append(epoch_replay_probs)
        #print(np.array(epochs_rollout_data).shape)
        #print(np.array(epochs_replay_probs).shape)
        #t_epoch = self._n_updates // self.n_epochs

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            new_temperature = np.sqrt(self.total_timesteps) / float(self.K * self.EMDA_step * (self.num_timesteps))
            #new_temperature = np.sqrt(self.total_timesteps) / float(self.K * self.EMDA_step * (t_epoch))
            #new_temperature = np.sqrt(self.n_epochs) / float(self.K * self.EMDA_step * (t_epoch+1)) # wrong!
            #new_temperature = 100
            sqrt_T = np.sqrt(self.n_epochs)
            #for rollout_data in self.rollout_buffer.get(self.batch_size):
            for rollout_data in epochs_rollout_data[epoch]:
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                
                batch_actions = np.zeros(self.batch_size)
                batch_values = np.zeros(self.batch_size)

                val_q_values = th.zeros(self.batch_size, requires_grad=True).to(self.device)
                #val_log_prob = th.zeros(self.batch_size).to(self.device)
                advantages = th.zeros(self.batch_size).to(self.device)
                #advantages = np.zeros(self.batch_size)

                action_advantages = []
                action_probs = []
                #action_log_probs = []
                #action_q_values = []
                positive_adv_prob = np.zeros(self.batch_size)
                negative_adv_prob = np.zeros(self.batch_size)
                # Q-value
                minMu = np.ones(self.batch_size)
                epsilon = np.zeros(self.batch_size)
                # print("train self.policy.evaluate_actions real actions")
                # action_q_values, val_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                #action_q_values, val_log_prob, entropy, energy_function = self.policy.evaluate_actions(rollout_data.observations, actions, self.temperature)
                action_q_values, rollout_log_prob, entropy, _ = self.policy.evaluate_actions(rollout_data.observations, actions, self.temperature)
                #action_q_values, _, entropy, _ = self.policy.evaluate_actions(rollout_data.observations, actions, self.temperature)

                t_computeV_start = time.time()
                
                gpu_batch_values  = th.zeros_like(rollout_log_prob)
                gpu_uniform_batch_values  = th.zeros_like(rollout_log_prob)
                gpu_action_advantages = [] # EMDA
                gpu_action_probs = []
                #self.EMDA_policy.load_state_dict(self.policy.state_dict())
                for a in range(self.action_space.n):
                    batch_actions = np.full(batch_actions.shape,a)
                    q_values, a_log_prob, _, _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device), self.temperature)
                    gpu_q =  q_values[:,a].flatten()
                    gpu_p = th.exp(a_log_prob)
                    gpu_action_probs.append(gpu_p)
                    ##print(uniform_action_idx)
                    #match_idx = (np.array(uniform_action_idx) == a)
                    ##print(match_idx)
                    #for idx in range(self.batch_size):
                    #    if match_idx[idx]:
                    #        uniform_action_prob[idx] = a_log_prob[a].detach()
                                        
                    p = gpu_p.cpu().detach()
                    gpu_batch_values += gpu_q*gpu_p
                    gpu_uniform_batch_values += gpu_q / self.action_space.n
                    #print("gpu_values:", gpu_batch_values)
                    #print("gpu_uniform_values:", gpu_uniform_batch_values)
                    
                    gpu_action_advantages.append(gpu_q)
                    #print("val_values: ", val_values)
                t_computeV_end = time.time()
                computeV_time.append(t_computeV_end - t_computeV_start)
                
                t_action_adv_start = time.time()
                
                all_advantages = th.zeros((self.batch_size, self.action_space.n)).to(self.device)
                for a in range(self.action_space.n):
                    # action_advantages[a] -= batch_values
                    gpu_action_advantages[a] -= gpu_batch_values # maybe ok? tune in 5
                    all_advantages[:,a] = gpu_action_advantages[a].detach()
                
                for j in range(self.batch_size):
                    val_q_values[j] = action_q_values[j][ actions[j] ]
                t_action_adv_end = time.time()
                t_action_adv_time.append(t_action_adv_end - t_action_adv_start)
                
                # 0624 a0 uniform policy for EMDA and eneryg function
                t_loss_start = time.time()
                c_values = th.tensor([0.] * self.batch_size, requires_grad=False).to(self.device) #6
                #policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha)
                #policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha, reduce=False, reduction='none')
                policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha, reduce=False, reduction='none')
                
                uniform_action_idx = [np.random.randint(self.action_space.n) for _ in range(self.batch_size)]
                uniform_q_value, val_log_prob, _, energy_function = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(np.array(uniform_action_idx)).to(self.device), self.temperature)
                
                old_log_prob = th.zeros_like(val_log_prob)
                for idx in range(self.batch_size):
                    old_log_prob[idx] = epochs_replay_probs[epoch][0][uniform_action_idx[idx]][idx].detach()
                
                init_energy_function = energy_function.clone().detach()
                #print(init_energy_function.shape)
                init_all_prob = th.sum(th.exp(init_energy_function / self.temperature), 1)
                init_log_prob = th.exp(init_energy_function / self.temperature) / init_all_prob.unsqueeze(1).expand(init_energy_function.shape)
                current_energy_function = energy_function.clone().detach() / self.temperature
                #val_log_prob2 = val_log_prob.clone().requires_grad_()
                next_all_prob = th.sum(th.exp(current_energy_function), 1)
                next_prob = th.log(th.exp(current_energy_function) / next_all_prob.unsqueeze(1).expand(current_energy_function.shape))
                #val_log_prob = th.Tensor((next_prob[ i ][ uniform_action_idx[ i ] ] for i in range(self.batch_size))).requires_grad_()
                #print([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)])
                val_log_prob2 = th.from_numpy(np.array([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)]))
                #check_log_prob = val_log_prob.clone().to('cpu').detach()
                #check_log_prob_g = val_log_prob2.clone().to(self.device).requires_grad_()
                #for i in range(self.batch_size):
                #    #if val_log_prob2[i].to('cpu') != check_log_prob[i]:
                #    if check_log_prob_g[i] != val_log_prob[i]:
                #        print(val_log_prob)
                #        print(check_log_prob_g)
                #        print(val_log_prob2)
                #        _ = input("")
                #val_log_prob2 = th.from_numpy(np.array([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)])).requires_grad_()
                #val_log_prob2 = th.from_numpy(np.array([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)])).to(self.device).requires_grad_()
                #print(val_log_prob.shape, old_log_prob.shape)
                #print(val_log_prob - old_log_prob)
                for j in range(self.batch_size):
                    advantages[j] = gpu_action_advantages[ uniform_action_idx[j] ][j]
                advantages = advantages.detach()
                y = th.sign(advantages) # 5 6
                abs_adv = y*advantages
                lr = self.lr_schedule(self._current_progress_remaining)
                for k in range(self.K):
                    val_log_prob_g = val_log_prob2.clone().to(self.device).requires_grad_()
                    #print(k)
                    #uniform_q_value, val_log_prob, _, energy_function = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(np.array(uniform_action_idx)).to(self.device), self.temperature)
                    if self.classifier == "AM":
                        #x1 = th.exp(val_log_prob - old_log_prob) # ratio maybe ok? tune in 5
                        #x1 = th.exp(val_log_prob2.to(self.device) - old_log_prob) # ratio maybe ok? tune in 5
                        #x1 = th.exp(val_log_prob2 - old_log_prob) # ratio maybe ok? tune in 5
                        x1 = th.exp(val_log_prob_g - old_log_prob) # ratio maybe ok? tune in 5
                        x2 = th.ones_like(x1.clone().detach())

                    g = policy_loss_fn(x1, x2, y)
                    #g_loss = -self.EMDA_step * abs_adv * th.exp(old_log_prob) * g
                    g_loss = -self.EMDA_step * abs_adv.clone().detach() * th.exp(old_log_prob) * g
                    self.policy.optimizer.zero_grad()
                    #val_log_prob2.retain_grad()
                    val_log_prob_g.retain_grad()
                    g_loss.sum().backward()
                    #c_values += val_log_prob2.grad.to(self.device)
                    #c_values += val_log_prob2.grad
                    val_log_prob_g.data -= lr * val_log_prob_g.grad
                    c_values += val_log_prob_g.grad
                    #val_log_prob2.grad.zero_()
                    val_log_prob_g.grad.zero_()
                    #c_values -= self.EMDA_step * abs_adv * old_log_prob * g

                    # compute next policy
                    all_prob = th.sum(th.exp(current_energy_function), 1)
                    #print(all_prob.shape)
                    log_prob_a = th.log(th.exp(current_energy_function) / all_prob.unsqueeze(1).expand(current_energy_function.shape)).requires_grad_()
                    all_x1 = th.exp(log_prob_a - init_log_prob)
                    all_x2 = th.ones_like(all_x1.clone().detach())
                    #y = th.sign(th.from_numpy(np.array(gpu_action_advantages)))
                    all_y = th.sign(all_advantages)
                    all_abs_adv = all_y * all_advantages
                    all_g = policy_loss_fn(all_x1, all_x2, all_y)

                    all_g_loss = -self.EMDA_step * all_abs_adv * th.exp(init_log_prob) * all_g
                    self.policy.optimizer.zero_grad()
                    log_prob_a.retain_grad()
                    all_g_loss.sum().backward()
                    log_prob_a.data -= lr * log_prob_a.grad
                    current_energy_function += log_prob_a.grad
                    log_prob_a.grad.zero_()
        

                    #current_energy_function -= self.EMDA_step * abs_adv * log_prob_a * g


                    #for j in range(self.batch_size):
                    #    for a in range(self.action_space.n):
                    #        #print(j, a)
                    #        #print((current_energy_function[j][a]))
                    #        #print(all_prob[j].unsqueeze(1).expand(current_energy_function.shape))
                    #        log_prob_a = th.log(th.exp(current_energy_function[j][a])/all_prob[j])
                    #        x1 = th.exp(log_prob_a - init_log_prob[j][a]) # ratio maybe ok? tune in 5
                    #        x2 = th.ones_like(x1.clone().detach())
                    #        y = th.sign(gpu_action_advantages[a][j])
                    #        g = policy_loss_fn(x1.unsqu, x2, y)
                    #        current_energy_function[j][a] -= self.EMDA_step * g
                    next_all_prob = th.sum(th.exp(current_energy_function), 1)
                    next_prob = th.log(th.exp(current_energy_function) / next_all_prob.unsqueeze(1).expand(current_energy_function.shape))
                    #next_all_prob = th.sum(th.exp(current_energy_function / self.temperature), 1)
                    #next_prob = th.log(th.exp(current_energy_function / self.temperature) / next_all_prob.unsqueeze(1).expand(current_energy_function.shape))
                    #val_log_prob = th.Tensor((next_prob[ i ][ uniform_action_idx[ i ] ] for i in range(self.batch_size))).requires_grad_()
                    #print([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)])
                    #val_log_prob2 = th.from_numpy(np.array([next_prob[ i ][ uniform_action_idx[ i ] ].item() for i in range(self.batch_size)])).requires_grad_()
                    
                    ##val_log_prob = th.Tensor([0.] * self.batch_size, require_grad=True)
                    for j in range(self.batch_size):
                        val_log_prob2[j] = next_prob[j][uniform_action_idx[j]]


                t_loss_end = time.time()
                loss_time.append(t_loss_end - t_loss_start)
                
                #policy_loss /= self.batch_size
                #c_values /= self.batch_size
                #print("Policy loss", policy_loss_data)
                # debug 6
                #policy_loss = th.mean(th.stack(policy_loss_data))
                #policy_loss = th.mean(th.stack(policy_loss_data))
                #print("Policy loss", policy_loss.item())
                #for i in range(self.batch_size):
                #    epsilon[i] = alpha * min(1, prob_ratio[i])
                margins.append(epsilon)
                # policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon)
                # policy_loss = th.mean(abs_adv * policy_loss_fn(x1, x2, y))

                # Logging
                #pg_losses.append(policy_loss.item())
                pg_losses.append(c_values.sum().item())
                #pg_losses.append(policy_loss)
                #clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                #clip_fractions.append(clip_fraction)
                uniform_q_value, val_log_prob, _, energy_function = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(np.array(uniform_action_idx)).to(self.device), self.temperature)

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
                # Value loss using the TD(gae_lambda) target
                #value_loss = F.mse_loss(rollout_data.returns.unsqueeze(1), values_pred )
                value_loss = F.mse_loss(rollout_data.returns, values_pred )
                value_losses.append(value_loss.item())

                rollout_return.append(rollout_data.returns.detach().cpu().numpy())

                # ?? SKIP entropy loss??
                # entropy = None
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-val_log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                energy_loss = th.tensor([0.], requires_grad=True).to(self.device)
                #print(energy_function.shape)
                #print(c_values.shape)
                #print(abs_adv)
                for i in range(self.batch_size):
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i] * advantages[i] + energy_function[i, actions[i]].detach() / self.temperature))
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i] * advantages[i] + energy_function[i, actions[i]] / self.temperature)) #8 --> x
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i] * advantages[i] + energy_function[i, actions[i]].detach() / self.temperature)) #7
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i].detach() * advantages[i] + energy_function[i, actions[i]].detach() / self.temperature)) #8
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i].detach() * abs_adv[i] + energy_function[i, actions[i]].detach() / self.temperature)) #6
                    #energy_loss += F.mse_loss(energy_function[i, actions[i]], new_temperature * (c_values[i].detach() + energy_function[i, actions[i]].detach() / self.temperature)) #8
                    energy_loss += F.mse_loss(energy_function[i, uniform_action_idx[i]], new_temperature * (c_values[i].detach() + energy_function[i, uniform_action_idx[i]].detach() / self.temperature)) #6 7
                    #energy_loss += F.mse_loss(energy_function[i, uniform_action_idx[i]], new_temperature * (c_values[i].detach() * abs_adv[i] + energy_function[i, uniform_action_idx[i]].detach() / self.temperature)) #5
                energy_loss /= self.batch_size
                energy_losses.append(energy_loss.item())

                # org version
                if self.entropy_hpo == True:
                    #loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss ##0810 test ,hope to find 360 HPO-62
                    loss = energy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss ##0810 test ,hope to find 360 HPO-62
                else :
                    #loss = policy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                    loss = energy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                # loss = policy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                #loss = policy_loss
                #loss = th.stack(policy_loss).sum() + self.vf_coef * value_loss

                ## Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                ## Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
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
            
            self.temperature = new_temperature
            #if np.mean(pg_losses) < 1e-4:
            #    print(f"Early stopping at step {epoch} due to reaching ploss: {np.mean(pg_losses):.2f}")
            #    break

        t_epoch_end = time.time()
        epoch_time.append(t_epoch_end - t_epoch_start)
        logger.record("Time/train_loss/Sum", np.sum(loss_time))
        logger.record("Time/train_computeV/Sum", np.sum(computeV_time))
        logger.record("Time/train_epoch/Sum", np.sum(epoch_time))
        logger.record("Time/train_loss/Mean", np.mean(loss_time))
        logger.record("Time/train_computeV/Mean", np.mean(computeV_time))
        logger.record("Time/train_epoch/Mean", np.mean(epoch_time))
        logger.record("Time/train_action_adv/Sum", np.sum(t_action_adv_time))
        logger.record("Time/train_epoch/Mean", np.mean(epoch_time))
        logger.record("Time/train_action_adv/Mean", np.mean(t_action_adv_time))
        
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())


        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/energy_loss", np.mean(energy_losses))
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
        tb_log_name: str = "NeuralHPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "NeuralHPO":
        self.total_timesteps = total_timesteps
        
        return super(NeuralHPO, self).learn(
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
