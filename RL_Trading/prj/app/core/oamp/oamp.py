import datetime
import gc
import os
import typing
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import joblib
import timeit
from collections import deque
import pandas as pd
from RL_Trading.prj.app.core.oamp.config import ConfigOAMP
from RL_Trading.prj.app.core.oamp.utils import (
    get_m,
    get_p,
    get_r,
    upd_n,
    upd_w,
)
import seaborn as sns


class OAMP:
    def __init__(
        self,
        agents_count: int,
        episode_length: int,
        args: ConfigOAMP,
        verbose: bool = False,
    ):
        assert args.agents_weights_upd_freq > 0, "Agents' weights update frequency should be greater than 0"
        #assert args.loss_fn_window > 0, "Loss function window should be greater than 0"
        
        # Initializing agents
        self.agents_count = agents_count
        self.episode_length = episode_length
        self.agents_rewards = self.init_agents_losses(args.loss_fn_window * episode_length)
        self.agents_weights_upd_freq = args.agents_weights_upd_freq # Right now it is measured in groups
        self.lr_ub = args.lr_ub
        self.action_threshold = args.action_threshold or 1 / 4
        self.action_aggregation_type = args.action_aggregation_type
        self.agents_sample_freq = args.agents_sample_freq

        self.verbose = verbose
                
        # Initializing OAMP
        self.l_tm1 = np.zeros(agents_count)
        self.n_tm1 = np.ones(agents_count) * 0.25
        self.w_tm1 = np.ones(agents_count) / agents_count
        self.p_tm1 = np.ones(agents_count) / agents_count
        self.cum_err = np.zeros(agents_count)
        
        # Group params
        self.mc_samples = 200
        self.t_step = 0
        self.t_day = 0
        self.group_t = 0
        self.max_agent_idx = np.random.choice(np.arange(agents_count), p=self.p_tm1)
        self.max_agents = [np.random.choice(np.arange(agents_count), p=self.p_tm1, size=self.mc_samples)]
        self.next_agents = np.random.choice(np.arange(agents_count), p=self.p_tm1, size=self.mc_samples)
        self.last_actions = np.zeros(self.mc_samples)
        self.switch_agent = np.ones(self.mc_samples).astype(bool)
        self.rng = np.random.default_rng(42)

        # Initializing OAMP stats
        self.stats = {
            "rewards": [],
            "reward_times": [],
            "losses": [],
            "losses_times": [],
            "weights": [],
            "agents_history": [],
            "probabilities": [],
            "sample_probabilities": [self.p_tm1],
            "sample_times": [],
            'pnl': []
        }
        
    def init_agents_losses(
        self,
        loss_fn_window: int,
    ):
        if loss_fn_window<0:
            return []

        return deque(maxlen=loss_fn_window)
    
    def step(
        self,
        agents_rewards: np.ndarray,
        agents_pnl: np.ndarray,
        time: datetime.datetime,
    ):
        is_last_step_day = False
        if self.t_step > 0 and (self.t_step + 1) % self.episode_length == 0:
            is_last_step_day = True
            self.group_t += 1
            self.t_day += 1
                                    
        # Updating agents' losses
        self.agents_rewards.append(agents_pnl)
        self.stats["rewards"].append(agents_rewards)
        self.stats['pnl'].append(agents_pnl)
        self.stats["reward_times"].append(time)
        self.stats["agents_history"].append(self.max_agent_idx)

        if self.group_t > 0 and is_last_step_day:

            # Updating agent
            if self.group_t%self.agents_sample_freq==0:
                if self.verbose:
                    print(f"Updating agent at t={self.t_step} {self.t_step % self.episode_length}")
                self.next_agents = self.rng.choice(np.arange(self.agents_count), size=self.mc_samples, p=self.p_tm1)
                self.max_agent_idx = self.rng.choice(np.arange(self.agents_count), size=1, p=self.p_tm1)[0]
                self.stats['sample_probabilities'].append(self.p_tm1)
                self.stats['sample_times'].append(time)


            # Updating agents' weights
            if self.group_t%self.agents_weights_upd_freq==0:
                if self.verbose:
                    print(f"Updating agents' weights at t={self.t_step} {self.t_step % self.episode_length}")
                self.update_agents_weights()
                #self.group_t = 0
                self.stats["losses_times"].append(time)

        self.max_agents.append(np.where(self.switch_agent, self.next_agents, self.max_agents[-1]))

        #self.max_agents = np.append(self.max_agents, np.where(self.last_actions == 0, self.next_agents, self.max_agents[:, -1])[:, None], axis=1)
        self.t_step += 1
            
    def update_agents_weights(
        self,
    ):
        # Computing agents' losses
        l_t = self.compute_agents_losses()
        # Computing agents' regrets estimates
        m_t = get_m(
            self.l_tm1,
            self.n_tm1,
            self.w_tm1,
            self.agents_count,
        )
        # Computing agents' selection probabilites
        p_t = get_p(m_t, self.w_tm1, self.n_tm1)
        #p_t = [1/self.agents_count]*self.agents_count
        # Computing agents' regrets
        r_t = get_r(l_t, p_t)
        # Computing agents' regrets estimatation error
        self.cum_err += (r_t - m_t) ** 2
        # Updating agents' learning rates
        n_t = upd_n(self.cum_err, self.agents_count, ub=self.lr_ub)
        # Updating agents' weights

        w_t = upd_w(
            self.w_tm1,
            self.n_tm1,
            n_t,
            r_t,
            m_t,
            self.agents_count,
        )

        #w_t = [1/self.agents_count]*self.agents_count
        self.l_tm1 = l_t
        self.n_tm1 = n_t
        self.w_tm1 = w_t
        self.p_tm1 = p_t
        self.stats["losses"].append(l_t)
        self.stats["weights"].append(w_t)
        self.stats["probabilities"].append(p_t)


        #self.max_agent_idx = np.argmax(self.p_tm1)
        #self.max_agent_idx = self.rng.choice(np.arange(self.agents_count), size=1, p=self.p_tm1)[0]

        return p_t

    def compute_agents_losses(
        self,
    ) -> np.ndarray:
        # Computing agents' losses
        agents_losses: np.ndarray = -np.sum(self.agents_rewards, axis=0)
        # Normalizing agents' losses
        agents_losses_min = agents_losses.min()
        agents_losses_max = agents_losses.max()
        if agents_losses_min != agents_losses_max:
            agents_losses = (agents_losses - agents_losses_min) / (
                agents_losses_max - agents_losses_min
            )
            
        return agents_losses

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'class.joblib'))
    
    @staticmethod
    def load(path: str) -> 'OAMP':
        return joblib.load(os.path.join(path, 'class.joblib'))

    
    # Actions are in [0, 1, 2] range
    def compute_action(
        self,
        agent_actions: np.ndarray,
    ):
        self.switch_agent = (self.last_actions != agent_actions[self.max_agents[-1]]) | (agent_actions[self.max_agents[-1]]==0)
        self.last_actions = agent_actions[self.max_agents[-1]]

        if self.action_aggregation_type == "max":
            #return agent_actions[np.argmax(self.p_tm1)]
            return agent_actions[self.max_agent_idx]
        elif self.action_aggregation_type == "threshold":
            action = np.dot(agent_actions, self.p_tm1)
            action_int = action - 1
            if np.abs(action_int) < self.action_threshold:
                return 1
            else:
                return np.sign(action_int) + 1
        else:
            raise ValueError(f"Unknown aggregation type: {self.action_aggregation_type}")
                        
        
    def plot_stats(
        self,
        figsize=(10, 14),
        labels: typing.Optional[typing.List[str]] = None,
        palette_dict: typing.Optional[dict] = None,
        save_path: typing.Optional[str] = None,
    ):
        labels = [f"Agent {n}" for n in range(self.agents_count)] if labels is None else labels
        if palette_dict is None:
            color_palette = sns.color_palette("tab10", len(labels))
            palette_dict = {label: color for label, color in zip(labels, color_palette)}

        agents_rewards = np.array(self.stats["pnl"])
        agents_rewards_times = np.array(self.stats["reward_times"])
        agents_losses = np.array(self.stats["losses"])
        agents_losses_times = np.array(self.stats["losses_times"])
        agents_weights = np.array(self.stats["weights"])
        agents_probabilities = np.array(self.stats['probabilities'])

        
        fig, axs = plt.subplots(2, 1, figsize=figsize)

        [
            axs[0].plot(agents_rewards_times, agents_rewards[:, i].cumsum(), color=palette_dict[labels[i]])
            for i in range(len(labels))
        ]
        axs[0].set_title("Agents' Rewards")
        #axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        axs[0].grid(True)
        axs[0].tick_params(labelbottom=False) 

        """
        [
            axs[1].plot(agents_losses_times, agents_losses[:, i].cumsum(), color=palette_dict[labels[i]])
            for i in range(len(labels))
        ]
        axs[1].set_title("Agents' Losses")
        axs[1].grid(True)
        axs[1].tick_params(labelbottom=False) 
        """

        colors = [palette_dict[agent] for agent in labels]

        axs[1].stackplot(agents_losses_times, np.transpose(agents_probabilities), colors=colors)
        axs[1].set_title("Agents' Probabilites")
        axs[1].grid(True)
        axs[1].tick_params(axis='x', rotation=45)

        fig.legend(labels=labels, loc="center left", bbox_to_anchor=(0.95, 0.5))

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

        plt.close(fig)
        gc.collect()

    def get_J(self, start_date):
        reward_times = np.array(self.stats['reward_times'])

        idx = 0
        if start_date is not None:
            idx = np.searchsorted(reward_times, start_date, side='right') - 1

        J = np.array(self.max_agents).T[:, idx+1:]

        return J


    def plot_distribution(self, save_path, ql, start_date, labels, phase):
        probabilities = np.array(self.stats['sample_probabilities'])
        probs_normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
        times = np.array([self.stats['reward_times'][0]] + self.stats['sample_times'])
        reward_times = np.array(self.stats['reward_times'])
        rewards = np.array(self.stats['rewards'])
        pnl = np.array(self.stats['pnl'])

        k = self.agents_sample_freq * self.episode_length
        idx = 0
        if start_date is not None:
            idx = np.searchsorted(times, start_date, side='right') -1
            probs_normalized = probs_normalized[idx:, :]

            idx = np.searchsorted(reward_times, start_date, side='right') -1
            rewards = rewards[idx:, :]
            pnl = pnl[idx:, :]
            reward_times = reward_times[idx:]


        n = 20000
        T, K = probs_normalized.shape

        # Precompute CDF for each timestep
        cdf = np.cumsum(probs_normalized, axis=1)

        # Generate uniforms and sample arms
        rng = np.random.default_rng()
        U = rng.random((n, T))
        J = np.array([np.searchsorted(cdf[t], U[:, t]) for t in range(T)]).T  # shape (n, T)
        J = np.repeat(J, k, axis=1)[:, :rewards.shape[0]]
        

        #J = np.array(self.max_agents).T[:, idx+1:]

        labels = [f"Agent {n}" for n in range(self.agents_count)] if labels is None else labels
        color_palette = sns.color_palette("tab10", len(labels))
        palette_dict = {label: color for label, color in zip(labels, color_palette)}
        colors = [palette_dict[agent] for agent in labels]
        out = pd.DataFrame(J).apply(pd.Series.value_counts, normalize=True).reindex(range(14), fill_value=0).fillna(0).sort_index()
        x = out.columns  # length 1000
        series = out.values  # shape (14, 1000)
        plt.figure()
        plt.stackplot(x, *series, colors=colors)  # or add colors=colors
        plt.title("Value Distribution per Column")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        plt.close()

        # Get rewards (deterministic)
        timestep_idx = np.arange(rewards.shape[0])  # (T,)
        rewards_per_round = rewards[timestep_idx, J]  # shape (n, T)
        pnl_per_round = pnl[timestep_idx, J] #20000, 1936; pnl: 1936, 9

        day_labels = reward_times.astype('datetime64[D]')  # (steps,)

        # 2. Turn those day labels into integer bins 0,1,2,… (sorted)
        unique_days, day_idx = np.unique(day_labels, return_inverse=True)
        #   unique_days : (days,)   e.g. array(['2025-08-06', '2025-08-07', ...], dtype='datetime64[D]')
        #   day_idx     : (steps,)  integers telling which day each step belongs to
        n_days = unique_days.size

        # 3. Bin-sum the pnl for every seed in a vectorised way ----------------
        #    np.bincount can sum `weights` that correspond to `day_idx`.
        pnl_per_day = np.vstack([
            np.bincount(day_idx,  # bins along the time axis
                        weights=row,  # the seed’s rewards
                        minlength=n_days)  # keep all days
            for row in pnl_per_round  # loop only over seeds (usually far fewer than steps)
        ])

        """
        pnl_per_day = (
            pnl_per_round
            .reshape(n, pnl_per_round.shape[1]//self.episode_length, self.episode_length)
            .sum(axis=2)
        )
        """

        means = np.mean(pnl_per_day, axis=1)
        stds = np.std(pnl_per_day, axis=1)
        sqrt_ndays = np.sqrt(pnl_per_day.shape[1])
        sharpes = sqrt_ndays*means/stds

        num_bins = 50

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.hist(sharpes, bins=num_bins)
        ax.set_xlabel("Sharpe ratio")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of Sharpe ratios ({phase})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"oamp_sharpe_{phase}.png"))
        plt.close(fig)


        # Compute cumulative rewards
        cum_rewards = rewards_per_round.cumsum(axis=1)  # (n, T)

        ql, q25, median, q75, qh = np.percentile(cum_rewards, [ql, 25, 50, 75, 100-ql], axis=0)


        plt.figure(figsize=(15, 7))
        plt.plot(reward_times, median, label='Median')
        plt.fill_between(reward_times, ql, qh, alpha=0.3, label='IQR 5-95')
        plt.fill_between(reward_times, q25, q75, alpha=0.3, label='IQR 25-75')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Cumulative percent pnl")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, f'oamp_iqr_{phase}.png'))
        #plt.show()
        plt.close()


        labels = [f"Agent {n}" for n in range(self.agents_count)] if labels is None else labels
        color_palette = sns.color_palette("tab10", len(labels))
        palette_dict = {label: color for label, color in zip(labels, color_palette)}



        plt.figure(figsize=(15, 7))
        for i in range(self.agents_count):
            plt.plot(reward_times, rewards[:, i].cumsum(), color=palette_dict[labels[i]], alpha=0.7)

        #plt.plot(times, median, label='oamp median', linewidth=3)
        #plt.plot(times, q25, label='oamp 25th percentile', linewidth=3)
        #plt.plot(times, q75, label='oamp 75th percentile', linewidth=3)

        plt.plot(reward_times, median, label='Median')
        plt.fill_between(reward_times, ql, qh, alpha=0.3, label='IQR 5-95')
        plt.fill_between(reward_times, q25, q75, alpha=0.3, label='IQR 25-75')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        plt.title(f'Cumulative percent pnl ({phase})')
        plt.legend()
        plt.xticks(rotation=45)
        #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.grid()
        plt.savefig(os.path.join(save_path, f'oamp_percentages_{phase}.png'))
        #plt.show()
        plt.close()

        return median[-1]

    def plot_agent_switches(
        self, 
        labels, 
        palette_dict = None, 
        figsize=(20, 10), 
        save_path=None
    ):
        
        labels = [f"Agent {n}" for n in range(self.agents_count)] if labels is None else labels
        if palette_dict is None:
            color_palette = sns.color_palette("tab10", len(labels))
            palette_dict = {label: color for label, color in zip(labels, color_palette)}
                        
        agents_history = self.stats["agents_history"]
        times = self.stats["reward_times"]
        switches = [agents_history[0]]
        switch_times = [times[0]]

        assert len(agents_history) == len(times), "Agents' history and times should have the same length"
        
        last_i = 1
        for i in range(1, len(agents_history)):
            if agents_history[i] != agents_history[i - 1]:
                switches.append(agents_history[i])
                switch_times.append(times[i])
                last_i = i
        
        if len(switches) == 1 or last_i != len(agents_history) - 1:
            switches.append(agents_history[-1])
            switch_times.append(times[-1])

        plt.figure(figsize=figsize)
        plt.step(switch_times, switches, where='post', linewidth=2, color='b', label="Agent Switch")
        plt.scatter(switch_times, switches, color='r', zorder=5, label="Switch Points")

        for time, agent in zip(switch_times, switches):
            label = labels[agent]
            color = palette_dict[label]
            plt.text(time, agent, f"{agent}", fontsize=20, ha='center', va='bottom', color=color)

        plt.title("Agent Switches Over Time")
        plt.xlabel("Time")
        plt.ylabel("Agent")
        plt.grid(True)
        plt.legend()

        plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=15)
        for i, label in enumerate(plt.gca().get_yticklabels()):
            label.set_color(palette_dict[labels[i]])

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()
            
        plt.close()
        gc.collect()

        return agents_history


