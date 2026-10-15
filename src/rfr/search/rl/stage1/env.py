"""Stage-1 policy environment: GELU choices with fixed Softmax parameter 6."""

from __future__ import annotations

import numpy as np
from rfr.search.rl.stage1 import config as stage1_config

class TransformerOptEnv:
    """Stage-1 GELU search environment with fixed Softmax degree 6."""
    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 constraint_limits=None, prev_metrics=None, num_metrics=2):
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        self.num_metrics = num_metrics


        if constraint_limits is None:

            self.constraint_limits = {
                'loss': self.baseline_loss * (1 + stage1_config.REWARD_THRESHOLD),
                'metric1': self.baseline_p * (1 - stage1_config.REWARD_THRESHOLD),
                'metric2': self.baseline_s * (1 - stage1_config.REWARD_THRESHOLD)
            }
        else:
            self.constraint_limits = constraint_limits


        if prev_metrics is None:
            self.prev_episode_metrics = {
                'loss': self.baseline_loss,
                'metric1': self.baseline_p,
                'metric2': self.baseline_s,
                'cost': self.baseline_cost
            }
        else:
            self.prev_episode_metrics = prev_metrics


        self.mid_gelu_cost = stage1_config.GELU_COST[2]
        self.mid_softmax_cost = stage1_config.SOFTMAX_COST[4]
        self.expected_cost_per_layer = self.mid_gelu_cost + self.mid_softmax_cost


        self.max_cost_per_layer = stage1_config.GELU_COST[4] + stage1_config.SOFTMAX_COST[6]


        self.gelu_degree_to_norm = {4: 0.0, 2: 0.5, 1: 1.0, 0: 1.25}

        self.current_episode_metrics = None

        self.reset()

    def reset(self):
        """Reset the Stage 1 search state."""
        self.current_layer = 0
        self.accumulated_cost = 0.0
        self.gelu_config = []
        self.softmax_config = []
        self.prev_gelu_degree = 4


        self.accumulated_dense_reward = 0.0


        self.gelu_history = np.full(self.total_layers, stage1_config.HISTORY_MASK_VALUE, dtype=np.float32)


        return self._get_state()

    def get_gelu_action_mask(self, layer_idx=None):
        """Return the current GELU action mask without mutating the environment."""
        del layer_idx
        return stage1_config.STAGE1_GELU_ACTION_MASK.copy()

    def _get_state(self):
        """Build the state vector from selected degrees and the current layer."""


        position = np.zeros(self.total_layers)
        if self.current_layer < self.total_layers:
            position[self.current_layer] = 1.0


        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0

        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)


        gelu_norm = self.gelu_degree_to_norm.get(self.prev_gelu_degree, 0.0)


        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0

        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)


        progress = self.current_layer / self.total_layers


        prev_loss = self.prev_episode_metrics['loss']
        prev_m1 = self.prev_episode_metrics['metric1']
        prev_m2 = self.prev_episode_metrics['metric2']


        loss_budget = 1.0 - prev_loss / (self.constraint_limits['loss'] + 1e-8)

        m1_budget = prev_m1 / (self.constraint_limits['metric1'] + 1e-8) - 1.0

        if self.num_metrics == 1:
            m2_budget = 0.0
        else:

            m2_budget = prev_m2 / (self.constraint_limits['metric2'] + 1e-8) - 1.0


        loss_budget = np.clip(loss_budget, -1.0, 1.0)
        m1_budget = np.clip(m1_budget, -1.0, 1.0)
        m2_budget = np.clip(m2_budget, -1.0, 1.0)


        self._policy_cont_features = np.array(
            [cost_deviation, complexity_debt, progress,
             loss_budget, m1_budget, m2_budget],
            dtype=np.float32,
        )
        state = np.concatenate([
            position,
            [cost_deviation],
            [gelu_norm],
            [complexity_debt],
            [progress],
            self.gelu_history,
            [loss_budget, m1_budget, m2_budget]
        ])
        return state.astype(np.float32)

    def get_policy_cont_features(self):
        """Return the 6-dim continuous feature vector the active GTrXL policy
        consumes: [cost_deviation, complexity_debt, progress, loss_budget,
        m1_budget, m2_budget]. Refreshed on every ``_get_state()`` call, so it is
        valid for whatever state ``reset()``/``step()`` most recently returned."""
        return self._policy_cont_features.copy()

    def _compute_dense_step_reward(self, gelu_degree):
        """Compute a progress reward from the selected approximation degrees."""
        step_cost = stage1_config.GELU_COST[gelu_degree] + stage1_config.SOFTMAX_COST[stage1_config.FIXED_SOFTMAX_DEGREE]


        cost_saving = (self.max_cost_per_layer - step_cost) / self.max_cost_per_layer
        cost_reward = stage1_config.REWARD_DENSE_SCALE * cost_saving
        return cost_reward

    def step(self, gelu_action_idx):
        """Apply one layer decision and return the next transition."""

        gelu_degree = stage1_config.GELU_MAP[gelu_action_idx]
        softmax_degree = stage1_config.FIXED_SOFTMAX_DEGREE


        self.gelu_config.append(gelu_degree)
        self.softmax_config.append(softmax_degree)


        self.accumulated_cost += stage1_config.GELU_COST[gelu_degree] + stage1_config.SOFTMAX_COST[softmax_degree]


        self.prev_gelu_degree = gelu_degree


        self.gelu_history[self.current_layer] = self.gelu_degree_to_norm[gelu_degree]


        dense_reward = self._compute_dense_step_reward(gelu_degree)
        self.accumulated_dense_reward += dense_reward


        info = {
            'layer_index': self.current_layer,
            'curr_gelu_degree': gelu_degree,
            'curr_softmax_degree': softmax_degree,
            'accumulated_cost': self.accumulated_cost,
            'gelu_config': self.gelu_config.copy(),
            'softmax_config': self.softmax_config.copy(),
            'dense_reward': dense_reward,
            'gelu_history': self.gelu_history.copy(),
        }

        self.current_layer += 1

        if self.current_layer < self.total_layers:

            return self._get_state(), dense_reward, False, info
        else:

            final_reward = self._compute_final_reward()
            info['final_reward'] = final_reward
            info['accumulated_dense_reward'] = self.accumulated_dense_reward

            total_reward = final_reward + dense_reward
            return self._get_state(), total_reward, True, info

    def _compute_final_reward(self):
        """Combine the terminal constraint barrier and cost saving."""

        gelu_arr = np.array(self.gelu_config)
        softmax_arr = np.array(self.softmax_config)


        loss, m1, m2, _ = self.evaluator.evaluate_model(gelu_arr, softmax_arr)


        self.current_episode_metrics = {
            'loss': loss,
            'metric1': m1,
            'metric2': m2,
            'cost': self.accumulated_cost
        }

        def log_barrier_reward(curr_value, limit_value, is_upper_bound=True):
            """Apply a smooth log barrier to a constraint margin."""
            if is_upper_bound:

                margin = limit_value - curr_value
            else:

                margin = curr_value - limit_value

            if margin < 0:

                return -stage1_config.LOG_BARRIER_VIOLATION_SCALE * np.exp(-margin * stage1_config.LOG_BARRIER_VIOLATION_STEEPNESS)
            else:

                return stage1_config.LOG_BARRIER_SATISFACTION_SCALE * np.log(margin + 1e-5)

        r_loss_barrier = log_barrier_reward(loss, self.constraint_limits['loss'], is_upper_bound=True)
        r_m1_barrier = log_barrier_reward(m1, self.constraint_limits['metric1'], is_upper_bound=False)
        r_m2_barrier = log_barrier_reward(m2, self.constraint_limits['metric2'], is_upper_bound=False)


        if self.num_metrics == 1:
            r_constraint = (r_loss_barrier + r_m1_barrier) / 2.0
        else:
            r_constraint = (r_loss_barrier + r_m1_barrier + r_m2_barrier) / 3.0


        cost_saving = (self.baseline_cost - self.accumulated_cost) / self.baseline_cost
        r_cost = cost_saving * stage1_config.REWARD_COST_WEIGHT


        r_accuracy = r_constraint


        raw_reward = r_accuracy + r_cost


        scaled_reward = raw_reward / stage1_config.REWARD_NORMALIZATION_SCALE


        clipped_reward = np.clip(scaled_reward, stage1_config.REWARD_CLIP_MIN, stage1_config.REWARD_CLIP_MAX)


        return clipped_reward
