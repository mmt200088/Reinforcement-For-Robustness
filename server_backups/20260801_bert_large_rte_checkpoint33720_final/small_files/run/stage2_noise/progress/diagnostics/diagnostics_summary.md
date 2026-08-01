# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=33720）

_更新时间: 2026-08-01 20:13:00_  ·  累计用时: **23h34m02s**

**Run meta**：
- `profile` = `rte_large`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692999, 'shared': 676032, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `c58aa0cfe514fda78b4aa14e8dfaecd414ff7e0618688898be0c2759b8c76f28`
- `run_context_hash` = `7635e1458dfb2bb10fdf1069b6d482d87684358b8353e87f880d87b4c6dee91b`
- `cost_model_revision` = `network_weighted_compute_communication_v3`
- `resource_objective` = `{'compute_axis': 'learnable_block4_fusion_count', 'communication_axis': 'layerwise_precision_preset_utility', 'selection': 'network_weighted_sum_then_balance', 'ppo_surrogate': '(compute+rho*communication)/(1+rho)'}`
- `communication_importance_ratio` = `1.0`
- `network_axis_weights` = `[0.5, 0.5]`
- `compute_axis_denominator` = `24`
- `communication_axis_denominator` = `24`
- `resource_credit_mode` = `separable_weighted_per_slot_v1`
- `strict_resource_order` = `['weighted_score', 'balance_tiebreak']`
- `total_episodes_planned` = `50000`
- `rollout_size` = `120`
- `ppo_lr` = `5e-05`
- `gamma` = `1.0`
- `gae_lambda` = `1.0`
- `entropy_regularization` = `{'kind': 'disabled', 'coefficient': 0.0, 'optimization_role': 'monitor_only'}`
- `termination` = `{'mode': 'convergence_or_max_episodes', 'episode_limit': 50000, 'minimum_episodes': 90000, 'patience_updates': 100, 'requires_robust_feasible_candidate': True, 'frontier_stall_update_windows': 100, 'selected_action_stable_update_windows': 100, 'strict_revalidation_required': True, 'strict_revalidation_trials': 15, 'strict_revalidation_diagnostic_probability': 0.95, 'selection_order': 'feasible,weighted_resource_score,balance_tiebreak,confidence_vector,safety_margin_vector,action_lexicographic', 'entropy_role': 'diagnostic_only', 'validation_banks': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'counts_only_finite_ppo_updates': True}`
- `ppo_mode` = `{'factorized_actor_clip': True, 'behavior_log_prob_source': 'sampling_time_per_slot_v1', 'actor_credit_mode': 'shared_constraint_plus_separable_axis_resource', 'actor_advantage_normalization': 'per_slot_center_shared_scale_v1', 'entropy_average_active_slots': True, 'entropy_normalize_active_slots': True}`
- `stage2_k_trials` = `3`
- `baseline_groups` = `5`
- `baseline_trials_per_group` = `3`
- `constraint_bootstrap_samples` = `4096`
- `constraint_probabilities` = `{'online': 0.5, 'promotion': 0.8, 'final': 0.95}`
- `constraint_limits` = `{'loss': 0.5642259006574749, 'metric1': 0.729478125, 'metric2': 0.729478125, 'loss_std': 0.007530118327379561, 'metric1_std': 0.01368117243392052, 'metric2_std': 0.01368117243392052}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.7302083333333333, 'metric2_mean': 0.7302083333333333, 'loss_mean': 0.5636622384190559, 'metric1_std': 0.00684058621696026, 'metric2_std': 0.00684058621696026, 'loss_std': 0.0037650591636897803, 'metric1_threshold': 0.729478125, 'metric2_threshold': 0.729478125, 'loss_threshold': 0.5642259006574749, 'metric1_std_threshold': 0.01368117243392052, 'metric2_std_threshold': 0.01368117243392052, 'loss_std_threshold': 0.007530118327379561, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.5660970509052277, 0.5668774396181107, 0.5569720640778542], 'metric1_trials': [0.74609375, 0.7265625, 0.734375], 'metric2_trials': [0.74609375, 0.7265625, 0.734375]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.5611904039978981, 0.57029028236866, 0.5627498924732208], 'metric1_trials': [0.73046875, 0.73046875, 0.7265625], 'metric2_trials': [0.73046875, 0.73046875, 0.7265625]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.5664645880460739, 0.5640975162386894, 0.5642364472150803], 'metric1_trials': [0.72265625, 0.73828125, 0.73046875], 'metric2_trials': [0.72265625, 0.73828125, 0.73046875]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.5682400092482567, 0.5585056394338608, 0.565181277692318], 'metric1_trials': [0.73046875, 0.734375, 0.71875], 'metric2_trials': [0.73046875, 0.734375, 0.71875]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.5636107325553894, 0.5587103590369225, 0.5617098733782768], 'metric1_trials': [0.7265625, 0.72265625, 0.734375], 'metric2_trials': [0.7265625, 0.72265625, 0.734375]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.5636622384190559, 'metric1_mean': 0.7302083333333333, 'metric2_mean': 0.7302083333333333, 'loss_std': 0.0037650591636897803, 'metric1_std': 0.00684058621696026, 'metric2_std': 0.00684058621696026, 'limits': {'loss': 0.5642259006574749, 'metric1': 0.729478125, 'metric2': 0.729478125, 'loss_std': 0.007530118327379561, 'metric1_std': 0.01368117243392052, 'metric2_std': 0.01368117243392052}}, 'limits': {'loss': 0.5642259006574749, 'metric1': 0.729478125, 'metric2': 0.729478125, 'loss_std': 0.007530118327379561, 'metric1_std': 0.01368117243392052, 'metric2_std': 0.01368117243392052}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.5660970509052277, 0.5668774396181107, 0.5569720640778542], 'metric1_trials': [0.74609375, 0.7265625, 0.734375], 'metric2_trials': [0.74609375, 0.7265625, 0.734375]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.5611904039978981, 0.57029028236866, 0.5627498924732208], 'metric1_trials': [0.73046875, 0.73046875, 0.7265625], 'metric2_trials': [0.73046875, 0.73046875, 0.7265625]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.5664645880460739, 0.5640975162386894, 0.5642364472150803], 'metric1_trials': [0.72265625, 0.73828125, 0.73046875], 'metric2_trials': [0.72265625, 0.73828125, 0.73046875]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.5682400092482567, 0.5585056394338608, 0.565181277692318], 'metric1_trials': [0.73046875, 0.734375, 0.71875], 'metric2_trials': [0.73046875, 0.734375, 0.71875]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.5636107325553894, 0.5587103590369225, 0.5617098733782768], 'metric1_trials': [0.7265625, 0.72265625, 0.734375], 'metric2_trials': [0.7265625, 0.72265625, 0.734375]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.5636622384190559, 'metric1_mean': 0.7302083333333333, 'metric2_mean': 0.7302083333333333, 'loss_std': 0.0037650591636897803, 'metric1_std': 0.00684058621696026, 'metric2_std': 0.00684058621696026, 'limits': {'loss': 0.5642259006574749, 'metric1': 0.729478125, 'metric2': 0.729478125, 'loss_std': 0.007530118327379561, 'metric1_std': 0.01368117243392052, 'metric2_std': 0.01368117243392052}}, 'limits': {'loss': 0.5642259006574749, 'metric1': 0.729478125, 'metric2': 0.729478125, 'loss_std': 0.007530118327379561, 'metric1_std': 0.01368117243392052, 'metric2_std': 0.01368117243392052}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374594418853, 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823], 'loss_trials': [0.5638350975642565, 0.5665445207258424, 0.5712569893912718], 'metric1_trials': [0.7256317702441439, 0.7364620951538913, 0.7256317702441439], 'metric2_trials': [0.7256317702441439, 0.7364620951538913, 0.7256317702441439]}, {'group_index': 1001, 'group_probe_seed': 9223369376165613078, 'trial_seeds': [9223369376165613078, 9223369374455707559, 9223369381474450804], 'loss_trials': [0.5646650793320005, 0.5681355832285829, 0.5681884835343068], 'metric1_trials': [0.7328519868506421, 0.7364620957994289, 0.7220216619408948], 'metric2_trials': [0.7328519868506421, 0.7364620957994289, 0.7220216619408948]}, {'group_index': 1002, 'group_probe_seed': 9223369373444064327, 'trial_seeds': [9223369373444064327, 9223369371801271798, 9223369369079723813], 'loss_trials': [0.5686215616305382, 0.5697759639485218, 0.5681412844020968], 'metric1_trials': [0.7328519868506421, 0.736462094938712, 0.718411553852825], 'metric2_trials': [0.7328519868506421, 0.736462094938712, 0.718411553852825]}, {'group_index': 1003, 'group_probe_seed': 9223369367499062704, 'trial_seeds': [9223369367499062704, 9223369369146835969, 9223369370794643154], 'loss_trials': [0.5706468680705404, 0.5725608586404298, 0.5676208370859442], 'metric1_trials': [0.7220216621560741, 0.7256317704593231, 0.7292418787625723], 'metric2_trials': [0.7220216621560741, 0.7256317704593231, 0.7292418787625723]}, {'group_index': 1004, 'group_probe_seed': 9223369369139590113, 'trial_seeds': [9223369369139590113, 9223369366492400208, 9223369373499408515], 'loss_trials': [0.5650728448203325, 0.569394763195988, 0.5704749570856886], 'metric1_trials': [0.7328519866354629, 0.7220216617257156, 0.7400722032419611], 'metric2_trials': [0.7328519866354629, 0.7220216617257156, 0.7400722032419611]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.5683290461770895, 'metric1_mean': 0.7292418785904289, 'metric2_mean': 0.7292418785904289, 'loss_std': 0.002503922852936918, 'metric1_std': 0.006684621625804724, 'metric2_std': 0.006684621625804724, 'limits': {'loss': 0.5688973752232664, 'metric1': 0.7285126367118384, 'metric2': 0.7285126367118384, 'loss_std': 0.005007845705873836, 'metric1_std': 0.013369243251609448, 'metric2_std': 0.013369243251609448}}, 'limits': {'loss': 0.5688973752232664, 'metric1': 0.7285126367118384, 'metric2': 0.7285126367118384, 'loss_std': 0.005007845705873836, 'metric1_std': 0.013369243251609448, 'metric2_std': 0.013369243251609448}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720425372765, 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903], 'loss_trials': [0.5645714064367411, 0.5714349690757503, 0.5691978365505646], 'metric1_trials': [0.7328519866354629, 0.7184115536376458, 0.729241878547393], 'metric2_trials': [0.7328519866354629, 0.7184115536376458, 0.729241878547393]}, {'group_index': 2001, 'group_probe_seed': 9223366722063803790, 'trial_seeds': [9223366722063803790, 9223366720019946559, 9223366726961026796], 'loss_trials': [0.5696805818846941, 0.567129315667204, 0.5730096383645646], 'metric1_trials': [0.7220216619408948, 0.7292418783322138, 0.7220216621560741], 'metric2_trials': [0.7220216619408948, 0.7292418783322138, 0.7220216621560741]}, {'group_index': 2002, 'group_probe_seed': 9223366719342124031, 'trial_seeds': [9223366719342124031, 9223366717365510734, 9223366714717300893], 'loss_trials': [0.5705414907166243, 0.5673881349580813, 0.5658846768661526], 'metric1_trials': [0.7256317704593231, 0.7220216619408948, 0.7436823117603895], 'metric2_trials': [0.7256317704593231, 0.7220216619408948, 0.7436823117603895]}, {'group_index': 2003, 'group_probe_seed': 9223366712331899176, 'trial_seeds': [9223366712331899176, 9223366714711074969, 9223366717367542346], 'loss_trials': [0.5701925175715009, 0.5721831689672779, 0.5746342264357887], 'metric1_trials': [0.711191337461506, 0.7328519868506421, 0.729241878547393], 'metric2_trials': [0.711191337461506, 0.7328519868506421, 0.729241878547393]}, {'group_index': 2004, 'group_probe_seed': 9223366713905190553, 'trial_seeds': [9223366713905190553, 9223366712056639272, 9223366717992199675], 'loss_trials': [0.5691034699174902, 0.5711051554886443, 0.5779993618008031], 'metric1_trials': [0.7328519868506421, 0.7256317702441439, 0.7220216623712533], 'metric2_trials': [0.7328519868506421, 0.7256317702441439, 0.7220216623712533]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.5702703967134589, 'metric1_mean': 0.7265944658490581, 'metric2_mean': 0.7265944658490581, 'loss_std': 0.003431175334918224, 'metric1_std': 0.00765414304874562, 'metric2_std': 0.00765414304874562, 'limits': {'loss': 0.5708406671101723, 'metric1': 0.725867871383209, 'metric2': 0.725867871383209, 'loss_std': 0.006862350669836448, 'metric1_std': 0.01530828609749124, 'metric2_std': 0.01530828609749124}}, 'limits': {'loss': 0.5708406671101723, 'metric1': 0.725867871383209, 'metric2': 0.725867871383209, 'loss_std': 0.006862350669836448, 'metric1_std': 0.01530828609749124, 'metric2_std': 0.01530828609749124}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066322514933, 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119], 'loss_trials': [0.5691025655192158, 0.5645805263777502, 0.5616015608990665], 'metric1_trials': [0.7328519868506421, 0.7292418787625723, 0.736462094938712], 'metric2_trials': [0.7328519868506421, 0.7292418787625723, 0.736462094938712]}, {'group_index': 3001, 'group_probe_seed': 9223364067895867686, 'trial_seeds': [9223364067895867686, 9223364065584185495, 9223364072532478532], 'loss_trials': [0.5726014595169453, 0.5724266974073885, 0.5681566783237113], 'metric1_trials': [0.7328519868506421, 0.718411553852825, 0.7148014453343966], 'metric2_trials': [0.7328519868506421, 0.718411553852825, 0.7148014453343966]}, {'group_index': 3002, 'group_probe_seed': 9223364065181531799, 'trial_seeds': [9223364065181531799, 9223364062929749798, 9223364060283768309], 'loss_trials': [0.5704062068935766, 0.5711528835313845, 0.5626344971278084], 'metric1_trials': [0.7328519870658213, 0.7220216621560741, 0.7328519872810005], 'metric2_trials': [0.7328519870658213, 0.7220216621560741, 0.7328519872810005]}, {'group_index': 3003, 'group_probe_seed': 9223364058162918592, 'trial_seeds': [9223364058162918592, 9223364060275314033, 9223364062921426850], 'loss_trials': [0.5709217926655435, 0.5638210825111031, 0.5733527780009521], 'metric1_trials': [0.7256317704593231, 0.7220216619408948, 0.7184115536376458], 'metric2_trials': [0.7256317704593231, 0.7220216619408948, 0.7184115536376458]}, {'group_index': 3004, 'group_probe_seed': 9223364059736205873, 'trial_seeds': [9223364059736205873, 9223364057620878208, 9223364063562602835], 'loss_trials': [0.5704809272332312, 0.5693718470390954, 0.5716618815914388], 'metric1_trials': [0.7256317702441439, 0.7220216621560741, 0.7148014455495758], 'metric2_trials': [0.7256317702441439, 0.7220216621560741, 0.7148014455495758]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.5688182256425474, 'metric1_mean': 0.7253910964720228, 'metric2_mean': 0.7253910964720228, 'loss_std': 0.0038286292544674194, 'metric1_std': 0.007151122016037655, 'metric2_std': 0.007151122016037655, 'limits': {'loss': 0.5693870438681898, 'metric1': 0.7246657053755508, 'metric2': 0.7246657053755508, 'loss_std': 0.007657258508934839, 'metric1_std': 0.01430224403207531, 'metric2_std': 0.01430224403207531}}, 'limits': {'loss': 0.5693870438681898, 'metric1': 0.7246657053755508, 'metric2': 0.7246657053755508, 'loss_std': 0.007657258508934839, 'metric1_std': 0.01430224403207531, 'metric2_std': 0.01430224403207531}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.5692997214452741, 'metric1_mean': 0.7279181722197433, 'metric2_mean': 0.7279181722197433, 'loss_std': 0.003112060122312347, 'metric1_std': 0.007187985584066076, 'metric2_std': 0.007187985584066076, 'limits': {'loss': 0.5698690211667193, 'metric1': 0.7271902540475236, 'metric2': 0.7271902540475236, 'loss_std': 0.006224120244624694, 'metric1_std': 0.014375971168132152, 'metric2_std': 0.014375971168132152}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.5691392228443652, 'metric1_mean': 0.72707581363717, 'metric2_mean': 0.72707581363717, 'loss_std': 0.0033316616028384745, 'metric1_std': 0.007195560072850961, 'metric2_std': 0.007195560072850961, 'limits': {'loss': 0.5697083620672095, 'metric1': 0.7263487378235328, 'metric2': 0.7263487378235328, 'loss_std': 0.006663323205676949, 'metric1_std': 0.014391120145701921, 'metric2_std': 0.014391120145701921}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 277, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **33720**
- 最近 50 回合 mean return: **+0.6627** (min=-3.4669, max=+1.3756)
- 最近 50 回合 mean terminal reward: **+0.6627**
- 最近 50 回合 mean invalid 子步数: **0.00** / 95
- 训练期 best reward: **+1.5110**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **281**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 113 | 3 | +0.5104 | +1.5110 | +1.5110 | +0.0000 | 24 | 0 | 0 | 62 |
| 2 | 1019 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 24 | 0 | 0 | 62 |
| 3 | 78 | 3 | +0.4896 | +1.4900 | +1.4900 | +0.0000 | 24 | 0 | 0 | 59 |
| 4 | 4008 | 3 | +0.4792 | +1.4798 | +1.4798 | +0.0000 | 24 | 0 | 0 | 62 |
| 5 | 382 | 3 | +0.4792 | +1.4795 | +1.4795 | +0.0000 | 24 | 0 | 0 | 62 |
| 6 | 368 | 3 | +0.4792 | +1.4794 | +1.4794 | +0.0000 | 24 | 0 | 0 | 61 |
| 7 | 4900 | 3 | +0.4688 | +1.4694 | +1.4694 | +0.0000 | 24 | 0 | 0 | 59 |
| 8 | 4824 | 3 | +0.4688 | +1.4692 | +1.4692 | +0.0000 | 24 | 0 | 0 | 62 |
| 9 | 4686 | 3 | +0.4688 | +1.4692 | +1.4692 | +0.0000 | 24 | 0 | 0 | 61 |
| 10 | 173 | 3 | +0.4688 | +1.4691 | +1.4691 | +0.0000 | 24 | 0 | 0 | 63 |
| 11 | 5393 | 3 | +0.4583 | +1.4590 | +1.4590 | +0.0000 | 24 | 0 | 0 | 60 |
| 12 | 1062 | 3 | +0.4583 | +1.4589 | +1.4589 | +0.0000 | 24 | 0 | 0 | 65 |
| 13 | 649 | 3 | +0.4583 | +1.4589 | +1.4589 | +0.0000 | 24 | 0 | 0 | 58 |
| 14 | 6077 | 3 | +0.4583 | +1.4589 | +1.4589 | +0.0000 | 24 | 0 | 0 | 62 |
| 15 | 4908 | 3 | +0.4583 | +1.4588 | +1.4588 | +0.0000 | 24 | 0 | 0 | 62 |
| 16 | 247 | 3 | +0.4583 | +1.4588 | +1.4588 | +0.0000 | 24 | 0 | 0 | 58 |
| 17 | 3849 | 3 | +0.4479 | +1.4486 | +1.4486 | +0.0000 | 24 | 0 | 0 | 61 |
| 18 | 7853 | 3 | +0.4479 | +1.4486 | +1.4486 | +0.0000 | 24 | 0 | 0 | 60 |
| 19 | 2467 | 3 | +0.4479 | +1.4486 | +1.4486 | +0.0000 | 24 | 0 | 0 | 60 |
| 20 | 1230 | 3 | +0.4479 | +1.4485 | +1.4485 | +0.0000 | 24 | 0 | 0 | 61 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 442 个槽与 baseline 不同_（322 SF + 120 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 7 | -6 |
| `L0.B2.K` | 13 | 6 | -7 |
| `L0.B3.K` | 13 | 6 | -7 |
| `L0.B4.K` | 13 | 8 | -5 |
| `L0.B5.K` | 13 | 7 | -6 |
| `L1.B1.K` | 13 | 9 | -4 |
| `L1.B2.K` | 13 | 8 | -5 |
| `L1.B3.K` | 13 | 8 | -5 |
| `L1.B4.K` | 13 | 10 | -3 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B1.K` | 13 | 11 | -2 |
| `L10.B2.K` | 13 | 10 | -3 |
| `L10.B3.K` | 13 | 10 | -3 |
| `L10.B4.K` | 13 | 12 | -1 |
| `L10.B5.K` | 13 | 11 | -2 |
| `L11.B1.K` | 13 | 11 | -2 |
| `L11.B2.K` | 13 | 10 | -3 |
| `L11.B3.K` | 13 | 10 | -3 |
| `L11.B4.K` | 13 | 12 | -1 |
| `L11.B5.K` | 13 | 11 | -2 |
| `L12.B1.K` | 13 | 9 | -4 |
| `L12.B2.K` | 13 | 8 | -5 |
| `L12.B3.K` | 13 | 8 | -5 |
| `L12.B4.K` | 13 | 10 | -3 |
| `L12.B5.K` | 13 | 9 | -4 |
| `L13.B1.K` | 13 | 7 | -6 |
| `L13.B2.K` | 13 | 6 | -7 |
| `L13.B3.K` | 13 | 6 | -7 |
| `L13.B4.K` | 13 | 8 | -5 |
| `L13.B5.K` | 13 | 7 | -6 |
| `L14.B1.K` | 13 | 9 | -4 |
| `L14.B2.K` | 13 | 8 | -5 |
| `L14.B3.K` | 13 | 8 | -5 |
| `L14.B4.K` | 13 | 10 | -3 |
| `L14.B5.K` | 13 | 9 | -4 |
| `L15.B1.K` | 13 | 7 | -6 |
| `L15.B2.K` | 13 | 6 | -7 |
| `L15.B3.K` | 13 | 6 | -7 |
| `L15.B4.K` | 13 | 8 | -5 |
| `L15.B5.K` | 13 | 7 | -6 |
| `L16.B1.K` | 13 | 11 | -2 |
| `L16.B2.K` | 13 | 10 | -3 |
| `L16.B3.K` | 13 | 10 | -3 |
| `L16.B4.K` | 13 | 12 | -1 |
| `L16.B5.K` | 13 | 11 | -2 |
| `L17.B1.K` | 13 | 11 | -2 |
| `L17.B2.K` | 13 | 10 | -3 |
| `L17.B3.K` | 13 | 10 | -3 |
| `L17.B4.K` | 13 | 12 | -1 |
| `L17.B5.K` | 13 | 11 | -2 |
| `L18.B1.K` | 13 | 11 | -2 |
| `L18.B2.K` | 13 | 10 | -3 |
| `L18.B3.K` | 13 | 10 | -3 |
| `L18.B4.K` | 13 | 12 | -1 |
| `L18.B5.K` | 13 | 11 | -2 |
| `L19.B1.K` | 13 | 7 | -6 |
| `L19.B2.K` | 13 | 6 | -7 |
| `L19.B3.K` | 13 | 6 | -7 |
| `L19.B4.K` | 13 | 8 | -5 |
| `L19.B5.K` | 13 | 7 | -6 |
| `L2.B1.K` | 13 | 9 | -4 |
| `L2.B2.K` | 13 | 8 | -5 |
| `L2.B3.K` | 13 | 8 | -5 |
| `L2.B4.K` | 13 | 10 | -3 |
| `L2.B5.K` | 13 | 9 | -4 |
| `L20.B1.K` | 13 | 11 | -2 |
| `L20.B2.K` | 13 | 10 | -3 |
| `L20.B3.K` | 13 | 10 | -3 |
| `L20.B4.K` | 13 | 12 | -1 |
| `L20.B5.K` | 13 | 11 | -2 |
| `L21.B1.K` | 13 | 11 | -2 |
| `L21.B2.K` | 13 | 10 | -3 |
| `L21.B3.K` | 13 | 10 | -3 |
| `L21.B4.K` | 13 | 12 | -1 |
| `L21.B5.K` | 13 | 11 | -2 |
| `L22.B1.K` | 13 | 11 | -2 |
| `L22.B2.K` | 13 | 10 | -3 |
| `L22.B3.K` | 13 | 10 | -3 |
| `L22.B4.K` | 13 | 12 | -1 |
| `L22.B5.K` | 13 | 11 | -2 |
| `L23.B1.K` | 13 | 7 | -6 |
| `L23.B2.K` | 13 | 6 | -7 |
| `L23.B3.K` | 13 | 6 | -7 |
| `L23.B4.K` | 13 | 8 | -5 |
| `L23.B5.K` | 13 | 7 | -6 |
| `L3.B1.K` | 13 | 7 | -6 |
| `L3.B2.K` | 13 | 6 | -7 |
| `L3.B3.K` | 13 | 6 | -7 |
| `L3.B4.K` | 13 | 8 | -5 |
| `L3.B5.K` | 13 | 7 | -6 |
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 8 | -5 |
| `L4.B3.K` | 13 | 8 | -5 |
| `L4.B4.K` | 13 | 10 | -3 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 11 | -2 |
| `L5.B2.K` | 13 | 10 | -3 |
| `L5.B3.K` | 13 | 10 | -3 |
| `L5.B4.K` | 13 | 12 | -1 |
| `L5.B5.K` | 13 | 11 | -2 |
| `L6.B1.K` | 13 | 9 | -4 |
| `L6.B2.K` | 13 | 8 | -5 |
| `L6.B3.K` | 13 | 8 | -5 |
| `L6.B4.K` | 13 | 10 | -3 |
| `L6.B5.K` | 13 | 9 | -4 |
| `L7.B1.K` | 13 | 11 | -2 |
| `L7.B2.K` | 13 | 10 | -3 |
| `L7.B3.K` | 13 | 10 | -3 |
| `L7.B4.K` | 13 | 12 | -1 |
| `L7.B5.K` | 13 | 11 | -2 |
| `L8.B1.K` | 13 | 7 | -6 |
| `L8.B2.K` | 13 | 6 | -7 |
| `L8.B3.K` | 13 | 6 | -7 |
| `L8.B4.K` | 13 | 8 | -5 |
| `L8.B5.K` | 13 | 7 | -6 |
| `L9.B1.K` | 13 | 9 | -4 |
| `L9.B2.K` | 13 | 8 | -5 |
| `L9.B3.K` | 13 | 8 | -5 |
| `L9.B4.K` | 13 | 10 | -3 |
| `L9.B5.K` | 13 | 9 | -4 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L1.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L9.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L12.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L14.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L15.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L16.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L17.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L18.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L21.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L0.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L0.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L0.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L1.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L1.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L1.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 272 | 32640 | -0.0018 | +0.1746 | +0.6103 | 0.025 | 0.00000 | 0.00092 | 1.000 | 0.00000 | +0.8568 | 0.00 |
| 273 | 32760 | -0.0030 | +0.2624 | +0.6146 | 0.027 | 0.00000 | 0.00313 | 1.000 | 0.00000 | +0.6253 | 0.00 |
| 274 | 32880 | -0.0028 | +0.2288 | +0.6046 | 0.028 | 0.00000 | 0.00085 | 1.000 | 0.00000 | +0.7196 | 0.00 |
| 275 | 33000 | -0.0018 | +0.2612 | +0.6155 | 0.019 | 0.00000 | 0.00155 | 1.000 | 0.00000 | +0.6362 | 0.00 |
| 276 | 33120 | -0.0018 | +0.3414 | +0.6297 | 0.020 | 0.00000 | 0.00196 | 1.000 | 0.00000 | +0.4105 | 0.00 |
| 277 | 33240 | -0.0026 | +0.3334 | +0.6333 | 0.026 | 0.00000 | 0.00203 | 1.000 | 0.00000 | +0.4352 | 0.00 |
| 278 | 33360 | -0.0040 | +0.1386 | +0.6291 | 0.043 | 0.00000 | 0.00349 | 1.000 | 0.00000 | +0.9678 | 0.00 |
| 279 | 33480 | -0.0021 | +0.3896 | +0.6283 | 0.024 | 0.00000 | 0.00152 | 1.000 | 0.00000 | +0.2820 | 0.00 |
| 280 | 33600 | -0.0023 | +0.3222 | +0.6188 | 0.022 | 0.00000 | 0.00171 | 1.000 | 0.00000 | +0.4553 | 0.00 |
| 281 | 33720 | -0.0021 | +0.2231 | +0.6276 | 0.024 | 0.00000 | 0.00178 | 1.000 | 0.00000 | +0.7448 | 0.00 |

_Entropy 趋势：+1.6024 → +0.6276（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**1625** / 1753
- **未收敛 slot**：**128** / 1753

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=14 （占比 100.0%）
  - slot[001] → action_index=14 （占比 100.0%）
  - slot[002] → action_index=14 （占比 100.0%）
  - slot[003] → action_index=14 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[007] → action_index=0 （占比 100.0%）

最分散 slot 示例（前 8 个）：
  - slot[640] entropy=1.082 (uniform≈4.159)
  - slot[592] entropy=1.082 (uniform≈4.159)
  - slot[615] entropy=1.082 (uniform≈4.159)
  - slot[623] entropy=1.082 (uniform≈4.159)
  - slot[656] entropy=1.082 (uniform≈4.159)
  - slot[1735] entropy=0.970 (uniform≈4.159)
  - slot[1687] entropy=0.970 (uniform≈4.159)
  - slot[1710] entropy=0.970 (uniform≈4.159)

## 6. 自动诊断（auto-flags）

- ✓ 暂无异常。

## 7. 原始数据文件（machine-readable）

| 文件 | 内容 |
|------|------|
| `episodes.jsonl` | 完整 per-episode 记录（append-only） |
| `ppo_updates.jsonl` | 完整 per-PPO-update 记录（append-only） |
| `top_candidates.jsonl` | Top-20 训练期 best：含每条候选的完整 `slots` 列表（人类可读） |
| `pareto_frontier.jsonl` | 训练期非支配候选（质量 / 稳定性 / cost 多目标） |
| `pareto_frontier.json` | Pareto frontier 元数据 + 完整候选列表 |
| `pareto_frontier.html` | 可直接用浏览器打开的 Pareto frontier 表格 |
| `first_invalid_counts.json` | (L, B) → 首次 invalid 计数 |
| `action_histogram.npz` | (num_slots, max_levels) 频次矩阵 |
| `baseline_action_vec.json` | static_skeletons baseline 的完整 `slots` 视图（参照系） |
| `best_action_vec.json` | **训练期最优**：`slots` 列表（按 SF/K 选）+ `action_vec` 兜底字段。**可直接喂给 `Paean/run_final_eval.sh --action-config`** |

**重跑 final eval 的最简命令**（无需等训练结束）：

```bash
bash Paean/run_final_eval.sh \
    --preset mrpc-final-eval-only \
    --action-config Parting Chapter/persistent/rl/bert-large/rte/s1t0.001_s2t0.001_s2st2.0__bertlarge_rte_stage2_k3_4gpu_a9559610_20260731/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。