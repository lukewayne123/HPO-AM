from typing import Any, Dict, Optional, Type, Union

import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

import numpy as np

from stable_baselines3.common import logger

class HPOAC(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (HPOAC)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to HPOAC: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
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
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        classifier: str="AM",
    ):

        super(HPOAC, self).__init__(
            policy,
            env,
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

        self.normalize_advantage = normalize_advantage

        # HPO 
        self.classifier = classifier
        self.logger = logger

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        #print("in hpoac.py def train: self .classifier",self.classifier)
        alpha = 0.1
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            #print("batch size", len(rollout_data))

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            #values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            val_values, val_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            val_values = val_values.flatten()

            # Original version
            ## Normalize advantage (not present in the original implementation)
            #advantages = rollout_data.advantages
            #if self.normalize_advantage:
            #    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
            
            advantages = rollout_data.advantages.detach()
            abs_adv = th.abs(advantages)
            #y = advantages / abs_adv
            y = rollout_data.advantages / abs_adv

            # handle each data
            #print(type(actions))
            np_actions = actions.detach().cpu().numpy()
            batch_actions = np.zeros_like(np_actions)
            batch_values = np.zeros_like(np_actions, dtype=float)
            action_advantages = []
            action_probs = []
            positive_adv_prob = np.zeros_like(np_actions, dtype=float)
            negative_adv_prob = np.zeros_like(np_actions, dtype=float)
            minMu = np.ones_like(np_actions)
            epsilon = np.zeros_like(np_actions, dtype=float)

            for a in range(self.action_space.n):
                batch_actions = np.full(batch_actions.shape,a)
                q_values, a_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                v = q_values.flatten().cpu().detach()
                p = th.exp(a_log_prob).cpu().detach()
                batch_values += (v*p).numpy()
                #v = q_values.flatten()
                #p = th.exp(a_log_prob)
                #batch_values += (v*p).cpu().detach().numpy()

                action_advantages.append(v)
                action_probs.append(p)

            for i in range(self.action_space.n):
                action_advantages[i] -= batch_values
                for j in range(len(rollout_data)):
                    minMu[j] = min(action_probs[i][j].clone().detach().numpy(),minMu[j])
                    if action_advantages[i][j] > 0:
                        positive_adv_prob[j] += action_probs[i][j].float()
                    if action_advantages[i][j] < 0:
                        negative_adv_prob[j] += action_probs[i][j].float()
            #positive_p.append(positive_adv_prob)
            #negative_p.append(negative_adv_prob)

            prob_ratio = negative_adv_prob / (positive_adv_prob + 1e-8)
            #print("Prob Ratio", prob_ratio)
            #ratio_p.append(prob_ratio)

            # epsilon = math.sqrt(1 + alpha * min(1, prob_ratio)) - 1
            policy_loss = th.tensor([0.], requires_grad=True).to(self.device)
            #policy_loss_data = []
            for i in range(len(rollout_data)):
                if self.classifier == "AM":
                    epsilon[i] = alpha * min(1, prob_ratio[i])
                elif self.classifier == "AM-log":
                    epsilon[i] = math.log(1 + alpha * min(1, prob_ratio[i]))
                elif self.classifier == "AM-root":
                    epsilon[i] = math.sqrt(1 + alpha * min(1, prob_ratio[i])) - 1
                elif self.classifier == "AM-sub":
                    epsilon[i] = minMu[i] * alpha * min(1, prob_ratio[i])
                elif self.classifier == "AM-square":
                    epsilon[i] = ( 1+alpha * min(1, prob_ratio[i]) )** 2
                policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon[i])
                # print("th.tensor([x1[i]]) , th.tensor([x2[i]]) , th.tensor([y[i]])",th.tensor([x1[i]]) , th.tensor([x2[i]]) , th.tensor([y[i]]))
                # th.tensor([x1[i]])
                #policy_loss = policy_loss + abs_adv[i] * policy_loss_fn( th.tensor([x1[i]]) , th.tensor([x2[i]]) , th.tensor([y[i]]) )
                #policy_loss += abs_adv[i] * policy_loss_fn( th.tensor([x1[i]]) , th.tensor([x2[i]]) , th.tensor([y[i]]) )
                policy_loss += abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , y[i].unsqueeze(0) )
            policy_loss /= float(len(rollout_data))

            ## Policy gradient loss
            #policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, val_values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-val_log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            #loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            #loss = -policy_loss + self.vf_coef * value_loss
            loss = policy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HPOAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "HPOAC":

        return super(HPOAC, self).learn(
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
