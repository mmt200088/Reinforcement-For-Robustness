# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=50000）

_更新时间: 2026-08-03 14:52:39_  ·  累计用时: **12h47m24s**

**Run meta**：
- `profile` = `rte`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692615, 'shared': 675648, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `d09b454c6c11d66408a443f5ed5888b6e7354b1b02fc6dfe78b43df52728a390`
- `run_context_hash` = `3a1a8451d9aa5d7ecb167d830be06e09cdeca5f4fbb50834ad53fc81237113fd`
- `cost_model_revision` = `network_weighted_compute_communication_v3`
- `resource_objective` = `{'compute_axis': 'learnable_block4_fusion_count', 'communication_axis': 'layerwise_precision_preset_utility', 'selection': 'network_weighted_sum_then_balance', 'ppo_surrogate': '(compute+rho*communication)/(1+rho)'}`
- `communication_importance_ratio` = `1.0`
- `network_axis_weights` = `[0.5, 0.5]`
- `compute_axis_denominator` = `12`
- `communication_axis_denominator` = `12`
- `resource_credit_mode` = `separable_weighted_per_slot_v1`
- `strict_resource_order` = `['weighted_score', 'balance_tiebreak']`
- `total_episodes_planned` = `50000`
- `rollout_size` = `120`
- `ppo_lr` = `5e-05`
- `gamma` = `1.0`
- `gae_lambda` = `1.0`
- `entropy_regularization` = `{'kind': 'disabled', 'coefficient': 0.0, 'optimization_role': 'monitor_only'}`
- `termination` = `{'mode': 'convergence_or_max_episodes', 'episode_limit': 50000, 'minimum_episodes': 90000, 'patience_updates': 100, 'requires_robust_feasible_candidate': True, 'frontier_stall_update_windows': 100, 'selected_action_stable_update_windows': 100, 'strict_revalidation_required': True, 'strict_revalidation_trials': 15, 'strict_revalidation_diagnostic_probability': 0.95, 'selection_order': 'feasible,weighted_resource_score,balance_tiebreak,confidence_vector,safety_margin_vector,action_lexicographic', 'entropy_role': 'diagnostic_only', 'validation_banks': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374610485242, 9223369376183642441, 9223369373462094616, 9223369367519304431, 9223369369126278334], 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176, 9223369376183642441, 9223369374471229688, 9223369381487745579, 9223369373462094616, 9223369371816794793, 9223369369093017722, 9223369367519304431, 9223369369160147806, 9223369370810182029, 9223369369126278334, 9223369366472158479, 9223369373481396188], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720442862338, 9223366722049835729, 9223366719328304288, 9223366712311789175, 9223366713885471174], 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432, 9223366722049835729, 9223366720000359264, 9223366726943404467, 9223366719328304288, 9223366717345776913, 9223366714699796418, 9223366712311789175, 9223366714697633734, 9223366717352133909, 9223366713885471174, 9223366712042801271, 9223366717976134308], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066309088426, 9223364067882557049, 9223364065161271752, 9223364058178311071, 9223364059751730542], 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296, 9223364067882557049, 9223364065563944904, 9223364072514465051, 9223364065161271752, 9223364062916452473, 9223364060268243626, 9223364058178311071, 9223364060293474862, 9223364062941552893, 9223364059751730542, 9223364057638909151, 9223364063582862860], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'counts_only_finite_ppo_updates': True}`
- `ppo_mode` = `{'factorized_actor_clip': True, 'behavior_log_prob_source': 'sampling_time_per_slot_v1', 'actor_credit_mode': 'shared_constraint_plus_separable_axis_resource', 'actor_advantage_normalization': 'per_slot_center_shared_scale_v1', 'entropy_average_active_slots': True, 'entropy_normalize_active_slots': True}`
- `stage2_k_trials` = `3`
- `baseline_groups` = `5`
- `baseline_trials_per_group` = `3`
- `constraint_bootstrap_samples` = `4096`
- `constraint_probabilities` = `{'online': 0.5, 'promotion': 0.8, 'final': 0.95}`
- `constraint_limits` = `{'loss': 0.7414938961754242, 'metric1': 0.7320796875, 'metric2': 0.7320796875, 'loss_std': 0.010484647372479504, 'metric1_std': 0.00924387466109315, 'metric2_std': 0.00924387466109315}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.7328125, 'metric2_mean': 0.7328125, 'loss_mean': 0.7407531430323918, 'metric1_std': 0.004621937330546575, 'metric2_std': 0.004621937330546575, 'loss_std': 0.005242323686239752, 'metric1_threshold': 0.7320796875, 'metric2_threshold': 0.7320796875, 'loss_threshold': 0.7414938961754242, 'metric1_std_threshold': 0.00924387466109315, 'metric2_std_threshold': 0.00924387466109315, 'loss_std_threshold': 0.010484647372479504, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [0.7371926009654999, 0.7382281422615051, 0.740156963467598], 'metric1_trials': [0.734375, 0.73828125, 0.734375], 'metric2_trials': [0.734375, 0.73828125, 0.734375]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [0.7336441874504089, 0.7414851784706116, 0.7549940943717957], 'metric1_trials': [0.73828125, 0.734375, 0.72265625], 'metric2_trials': [0.73828125, 0.734375, 0.72265625]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [0.7385600805282593, 0.7385584563016891, 0.7336225509643555], 'metric1_trials': [0.73828125, 0.7265625, 0.73828125], 'metric2_trials': [0.73828125, 0.7265625, 0.73828125]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [0.7395022064447403, 0.7412469834089279, 0.7454909980297089], 'metric1_trials': [0.73046875, 0.734375, 0.73046875], 'metric2_trials': [0.73046875, 0.734375, 0.73046875]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [0.7458194643259048, 0.7417405843734741, 0.7410546541213989], 'metric1_trials': [0.73046875, 0.73046875, 0.73046875], 'metric2_trials': [0.73046875, 0.73046875, 0.73046875]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7407531430323918, 'metric1_mean': 0.7328125, 'metric2_mean': 0.7328125, 'loss_std': 0.005242323686239752, 'metric1_std': 0.004621937330546575, 'metric2_std': 0.004621937330546575, 'limits': {'loss': 0.7414938961754242, 'metric1': 0.7320796875, 'metric2': 0.7320796875, 'loss_std': 0.010484647372479504, 'metric1_std': 0.00924387466109315, 'metric2_std': 0.00924387466109315}}, 'limits': {'loss': 0.7414938961754242, 'metric1': 0.7320796875, 'metric2': 0.7320796875, 'loss_std': 0.010484647372479504, 'metric1_std': 0.00924387466109315, 'metric2_std': 0.00924387466109315}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [0.7371926009654999, 0.7382281422615051, 0.740156963467598], 'metric1_trials': [0.734375, 0.73828125, 0.734375], 'metric2_trials': [0.734375, 0.73828125, 0.734375]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [0.7336441874504089, 0.7414851784706116, 0.7549940943717957], 'metric1_trials': [0.73828125, 0.734375, 0.72265625], 'metric2_trials': [0.73828125, 0.734375, 0.72265625]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [0.7385600805282593, 0.7385584563016891, 0.7336225509643555], 'metric1_trials': [0.73828125, 0.7265625, 0.73828125], 'metric2_trials': [0.73828125, 0.7265625, 0.73828125]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [0.7395022064447403, 0.7412469834089279, 0.7454909980297089], 'metric1_trials': [0.73046875, 0.734375, 0.73046875], 'metric2_trials': [0.73046875, 0.734375, 0.73046875]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [0.7458194643259048, 0.7417405843734741, 0.7410546541213989], 'metric1_trials': [0.73046875, 0.73046875, 0.73046875], 'metric2_trials': [0.73046875, 0.73046875, 0.73046875]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7407531430323918, 'metric1_mean': 0.7328125, 'metric2_mean': 0.7328125, 'loss_std': 0.005242323686239752, 'metric1_std': 0.004621937330546575, 'metric2_std': 0.004621937330546575, 'limits': {'loss': 0.7414938961754242, 'metric1': 0.7320796875, 'metric2': 0.7320796875, 'loss_std': 0.010484647372479504, 'metric1_std': 0.00924387466109315, 'metric2_std': 0.00924387466109315}}, 'limits': {'loss': 0.7414938961754242, 'metric1': 0.7320796875, 'metric2': 0.7320796875, 'loss_std': 0.010484647372479504, 'metric1_std': 0.00924387466109315, 'metric2_std': 0.00924387466109315}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374610485242, 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176], 'loss_trials': [0.7377777665530731, 0.7484287662626604, 0.7520842289666406], 'metric1_trials': [0.7328519862051045, 0.7364620945083535, 0.7292418781170346], 'metric2_trials': [0.7328519862051045, 0.7364620945083535, 0.7292418781170346]}, {'group_index': 1001, 'group_probe_seed': 9223369376183642441, 'trial_seeds': [9223369376183642441, 9223369374471229688, 9223369381487745579], 'loss_trials': [0.7508780062198639, 0.744312898454253, 0.7498056920857206], 'metric1_trials': [0.7328519862051045, 0.7400722028116027, 0.7292418781170346], 'metric2_trials': [0.7328519862051045, 0.7400722028116027, 0.7292418781170346]}, {'group_index': 1002, 'group_probe_seed': 9223369373462094616, 'trial_seeds': [9223369373462094616, 9223369371816794793, 9223369369093017722], 'loss_trials': [0.7403472257435106, 0.7451054589412703, 0.7497665361376876], 'metric1_trials': [0.7364620945083535, 0.7256317698137855, 0.7256317695986063], 'metric2_trials': [0.7364620945083535, 0.7256317698137855, 0.7256317695986063]}, {'group_index': 1003, 'group_probe_seed': 9223369367519304431, 'trial_seeds': [9223369367519304431, 9223369369160147806, 9223369370810182029], 'loss_trials': [0.746262741433154, 0.745708949083886, 0.7538385029734257], 'metric1_trials': [0.7364620945083535, 0.7292418779018554, 0.7184115532072873], 'metric2_trials': [0.7364620945083535, 0.7292418779018554, 0.7184115532072873]}, {'group_index': 1004, 'group_probe_seed': 9223369369126278334, 'trial_seeds': [9223369369126278334, 9223369366472158479, 9223369373481396188], 'loss_trials': [0.7499739708452879, 0.7442759772500407, 0.7408111385920418], 'metric1_trials': [0.7364620945083535, 0.7292418779018554, 0.7328519862051045], 'metric2_trials': [0.7364620945083535, 0.7292418779018554, 0.7328519862051045]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7466251906361677, 'metric1_mean': 0.7314079429411862, 'metric2_mean': 0.7314079429411862, 'loss_std': 0.004629171308808742, 'metric1_std': 0.005592755671099497, 'metric2_std': 0.005592755671099497, 'limits': {'loss': 0.7473718158268038, 'metric1': 0.730676534998245, 'metric2': 0.730676534998245, 'loss_std': 0.009258342617617484, 'metric1_std': 0.011185511342198994, 'metric2_std': 0.011185511342198994}}, 'limits': {'loss': 0.7473718158268038, 'metric1': 0.730676534998245, 'metric2': 0.730676534998245, 'loss_std': 0.009258342617617484, 'metric1_std': 0.011185511342198994, 'metric2_std': 0.011185511342198994}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720442862338, 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432], 'loss_trials': [0.7515303176232624, 0.7494548451599231, 0.7445294101746074], 'metric1_trials': [0.7256317695986063, 0.7328519862051045, 0.7364620945083535], 'metric2_trials': [0.7256317695986063, 0.7328519862051045, 0.7364620945083535]}, {'group_index': 2001, 'group_probe_seed': 9223366722049835729, 'trial_seeds': [9223366722049835729, 9223366720000359264, 9223366726943404467], 'loss_trials': [0.7470425013385524, 0.7351807195572216, 0.7485129889168034], 'metric1_trials': [0.7220216612953572, 0.7328519862051045, 0.7184115532072873], 'metric2_trials': [0.7220216612953572, 0.7328519862051045, 0.7184115532072873]}, {'group_index': 2002, 'group_probe_seed': 9223366719328304288, 'trial_seeds': [9223366719328304288, 9223366717345776913, 9223366714699796418], 'loss_trials': [0.75797227315524, 0.7528502758229252, 0.753890362672427], 'metric1_trials': [0.7184115532072873, 0.7256317700289647, 0.7184115532072873], 'metric2_trials': [0.7184115532072873, 0.7256317700289647, 0.7184115532072873]}, {'group_index': 2003, 'group_probe_seed': 9223366712311789175, 'trial_seeds': [9223366712311789175, 9223366714697633734, 9223366717352133909], 'loss_trials': [0.7423844352525925, 0.751074994729314, 0.7454658383066474], 'metric1_trials': [0.7256317698137855, 0.7364620945083535, 0.7184115532072873], 'metric2_trials': [0.7256317698137855, 0.7364620945083535, 0.7184115532072873]}, {'group_index': 2004, 'group_probe_seed': 9223366713885471174, 'trial_seeds': [9223366713885471174, 9223366712042801271, 9223366717976134308], 'loss_trials': [0.7475516441066342, 0.743602667904933, 0.7564870943231273], 'metric1_trials': [0.7328519862051045, 0.7328519862051045, 0.7292418779018554], 'metric2_trials': [0.7328519862051045, 0.7328519862051045, 0.7292418779018554]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7485020246029473, 'metric1_mean': 0.727075813020323, 'metric2_mean': 0.727075813020323, 'loss_std': 0.005885857326336365, 'metric1_std': 0.006795118670758171, 'metric2_std': 0.006795118670758171, 'limits': {'loss': 0.7492505266275502, 'metric1': 0.7263487372073026, 'metric2': 0.7263487372073026, 'loss_std': 0.01177171465267273, 'metric1_std': 0.013590237341516343, 'metric2_std': 0.013590237341516343}}, 'limits': {'loss': 0.7492505266275502, 'metric1': 0.7263487372073026, 'metric2': 0.7263487372073026, 'loss_std': 0.01177171465267273, 'metric1_std': 0.013590237341516343, 'metric2_std': 0.013590237341516343}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066309088426, 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296], 'loss_trials': [0.7509026146537561, 0.7503560105816122, 0.7429584774730008], 'metric1_trials': [0.7292418781170346, 0.7256317698137855, 0.7364620945083535], 'metric2_trials': [0.7292418781170346, 0.7256317698137855, 0.7364620945083535]}, {'group_index': 3001, 'group_probe_seed': 9223364067882557049, 'trial_seeds': [9223364067882557049, 9223364065563944904, 9223364072514465051], 'loss_trials': [0.7599973915286012, 0.7506387031465661, 0.7451582660743906], 'metric1_trials': [0.7292418781170346, 0.7220216615105364, 0.7256317698137855], 'metric2_trials': [0.7292418781170346, 0.7220216615105364, 0.7256317698137855]}, {'group_index': 3002, 'group_probe_seed': 9223364065161271752, 'trial_seeds': [9223364065161271752, 9223364062916452473, 9223364060268243626], 'loss_trials': [0.7470880786649587, 0.7484602596785618, 0.7500854004376202], 'metric1_trials': [0.7256317695986063, 0.7292418781170346, 0.714801444688859], 'metric2_trials': [0.7256317695986063, 0.7292418781170346, 0.714801444688859]}, {'group_index': 3003, 'group_probe_seed': 9223364058178311071, 'trial_seeds': [9223364058178311071, 9223364060293474862, 9223364062941552893], 'loss_trials': [0.7488654178402484, 0.7511099406958487, 0.756267782152775], 'metric1_trials': [0.7292418781170346, 0.7256317698137855, 0.7220216615105364], 'metric2_trials': [0.7292418781170346, 0.7256317698137855, 0.7220216615105364]}, {'group_index': 3004, 'group_probe_seed': 9223364059751730542, 'trial_seeds': [9223364059751730542, 9223364057638909151, 9223364063582862860], 'loss_trials': [0.7462482256579486, 0.75025090908746, 0.7489238022036501], 'metric1_trials': [0.7256317698137855, 0.7292418779018554, 0.7328519862051045], 'metric2_trials': [0.7256317698137855, 0.7292418779018554, 0.7328519862051045]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7498207519918001, 'metric1_mean': 0.7268351391764756, 'metric2_mean': 0.7268351391764756, 'loss_std': 0.004155885673918359, 'metric1_std': 0.0050443186207541554, 'metric2_std': 0.0050443186207541554, 'limits': {'loss': 0.7505705727437918, 'metric1': 0.7261083040372991, 'metric2': 0.7261083040372991, 'loss_std': 0.008311771347836719, 'metric1_std': 0.010088637241508311, 'metric2_std': 0.010088637241508311}}, 'limits': {'loss': 0.7505705727437918, 'metric1': 0.7261083040372991, 'metric2': 0.7261083040372991, 'loss_std': 0.008311771347836719, 'metric1_std': 0.010088637241508311, 'metric2_std': 0.010088637241508311}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.7475636076195575, 'metric1_mean': 0.7292418779807545, 'metric2_mean': 0.7292418779807545, 'loss_std': 0.005289659403850023, 'metric1_std': 0.006499577903622569, 'metric2_std': 0.006499577903622569, 'limits': {'loss': 0.748311171227177, 'metric1': 0.7285126361027737, 'metric2': 0.7285126361027737, 'loss_std': 0.010579318807700046, 'metric1_std': 0.012999155807245137, 'metric2_std': 0.012999155807245137}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.7483159890769717, 'metric1_mean': 0.7284396317126614, 'metric2_mean': 0.7284396317126614, 'loss_std': 0.00500949282750986, 'metric1_std': 0.00610373748471218, 'metric2_std': 0.00610373748471218, 'limits': {'loss': 0.7490643050660486, 'metric1': 0.7277111920809487, 'metric2': 0.7277111920809487, 'loss_std': 0.01001898565501972, 'metric1_std': 0.01220747496942436, 'metric2_std': 0.01220747496942436}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374610485242, 9223369376183642441, 9223369373462094616, 9223369367519304431, 9223369369126278334], 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176, 9223369376183642441, 9223369374471229688, 9223369381487745579, 9223369373462094616, 9223369371816794793, 9223369369093017722, 9223369367519304431, 9223369369160147806, 9223369370810182029, 9223369369126278334, 9223369366472158479, 9223369373481396188], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720442862338, 9223366722049835729, 9223366719328304288, 9223366712311789175, 9223366713885471174], 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432, 9223366722049835729, 9223366720000359264, 9223366726943404467, 9223366719328304288, 9223366717345776913, 9223366714699796418, 9223366712311789175, 9223366714697633734, 9223366717352133909, 9223366713885471174, 9223366712042801271, 9223366717976134308], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066309088426, 9223364067882557049, 9223364065161271752, 9223364058178311071, 9223364059751730542], 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296, 9223364067882557049, 9223364065563944904, 9223364072514465051, 9223364065161271752, 9223364062916452473, 9223364060268243626, 9223364058178311071, 9223364060293474862, 9223364062941552893, 9223364059751730542, 9223364057638909151, 9223364063582862860], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 277, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **50000**
- 最近 50 回合 mean return: **+0.4716** (min=-3.5000, max=+1.2507)
- 最近 50 回合 mean terminal reward: **+0.4716**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+1.5838**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **417**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 208 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 33 |
| 2 | 366 | 3 | +0.4792 | +1.4795 | +1.4795 | +0.0000 | 12 | 0 | 0 | 29 |
| 3 | 899 | 3 | +0.4375 | +1.4381 | +1.4381 | +0.0000 | 12 | 0 | 0 | 31 |
| 4 | 1417 | 3 | +0.4375 | +1.4381 | +1.4381 | +0.0000 | 12 | 0 | 0 | 30 |
| 5 | 3275 | 3 | +0.4375 | +1.4380 | +1.4380 | +0.0000 | 12 | 0 | 0 | 29 |
| 6 | 723 | 3 | +0.4375 | +1.4379 | +1.4379 | +0.0000 | 12 | 0 | 0 | 29 |
| 7 | 1395 | 3 | +0.4375 | +1.4377 | +1.4377 | +0.0000 | 12 | 0 | 0 | 29 |
| 8 | 13 | 3 | +0.4167 | +1.4172 | +1.4172 | +0.0000 | 12 | 0 | 0 | 31 |
| 9 | 364 | 3 | +0.4167 | +1.4171 | +1.4171 | +0.0000 | 12 | 0 | 0 | 28 |
| 10 | 286 | 3 | +0.4167 | +1.4169 | +1.4169 | +0.0000 | 12 | 0 | 0 | 29 |
| 11 | 601 | 3 | +0.4167 | +1.4169 | +1.4169 | +0.0000 | 12 | 0 | 0 | 30 |
| 12 | 1025 | 3 | +0.3958 | +1.3963 | +1.3963 | +0.0000 | 12 | 0 | 0 | 30 |
| 13 | 5347 | 3 | +0.3958 | +1.3963 | +1.3963 | +0.0000 | 12 | 0 | 0 | 27 |
| 14 | 3548 | 3 | +0.3750 | +1.3757 | +1.3757 | +0.0000 | 12 | 0 | 0 | 29 |
| 15 | 1744 | 3 | +0.3750 | +1.3757 | +1.3757 | +0.0000 | 12 | 0 | 0 | 28 |
| 16 | 132 | 3 | +0.3750 | +1.3756 | +1.3756 | +0.0000 | 12 | 0 | 0 | 30 |
| 17 | 672 | 3 | +0.3750 | +1.3756 | +1.3756 | +0.0000 | 12 | 0 | 0 | 29 |
| 18 | 347 | 3 | +0.3750 | +1.3756 | +1.3756 | +0.0000 | 12 | 0 | 0 | 30 |
| 19 | 1891 | 3 | +0.3750 | +1.3755 | +1.3755 | +0.0000 | 12 | 0 | 0 | 27 |
| 20 | 1043 | 3 | +0.3750 | +1.3755 | +1.3755 | +0.0000 | 12 | 0 | 0 | 30 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 232 个槽与 baseline 不同_（172 SF + 60 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 11 | -2 |
| `L0.B2.K` | 13 | 10 | -3 |
| `L0.B3.K` | 13 | 10 | -3 |
| `L0.B4.K` | 13 | 12 | -1 |
| `L0.B5.K` | 13 | 11 | -2 |
| `L1.B1.K` | 13 | 11 | -2 |
| `L1.B2.K` | 13 | 10 | -3 |
| `L1.B3.K` | 13 | 10 | -3 |
| `L1.B4.K` | 13 | 12 | -1 |
| `L1.B5.K` | 13 | 11 | -2 |
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
| `L2.B1.K` | 13 | 7 | -6 |
| `L2.B2.K` | 13 | 6 | -7 |
| `L2.B3.K` | 13 | 6 | -7 |
| `L2.B4.K` | 13 | 8 | -5 |
| `L2.B5.K` | 13 | 7 | -6 |
| `L3.B1.K` | 13 | 9 | -4 |
| `L3.B2.K` | 13 | 8 | -5 |
| `L3.B3.K` | 13 | 8 | -5 |
| `L3.B4.K` | 13 | 10 | -3 |
| `L3.B5.K` | 13 | 9 | -4 |
| `L4.B1.K` | 13 | 11 | -2 |
| `L4.B2.K` | 13 | 10 | -3 |
| `L4.B3.K` | 13 | 10 | -3 |
| `L4.B4.K` | 13 | 12 | -1 |
| `L4.B5.K` | 13 | 11 | -2 |
| `L5.B1.K` | 13 | 7 | -6 |
| `L5.B2.K` | 13 | 6 | -7 |
| `L5.B3.K` | 13 | 6 | -7 |
| `L5.B4.K` | 13 | 8 | -5 |
| `L5.B5.K` | 13 | 7 | -6 |
| `L6.B1.K` | 13 | 11 | -2 |
| `L6.B2.K` | 13 | 10 | -3 |
| `L6.B3.K` | 13 | 10 | -3 |
| `L6.B4.K` | 13 | 12 | -1 |
| `L6.B5.K` | 13 | 11 | -2 |
| `L7.B1.K` | 13 | 9 | -4 |
| `L7.B2.K` | 13 | 8 | -5 |
| `L7.B3.K` | 13 | 8 | -5 |
| `L7.B4.K` | 13 | 10 | -3 |
| `L7.B5.K` | 13 | 9 | -4 |
| `L8.B1.K` | 13 | 7 | -6 |
| `L8.B2.K` | 13 | 6 | -7 |
| `L8.B3.K` | 13 | 6 | -7 |
| `L8.B4.K` | 13 | 8 | -5 |
| `L8.B5.K` | 13 | 7 | -6 |
| `L9.B1.K` | 13 | 7 | -6 |
| `L9.B2.K` | 13 | 6 | -7 |
| `L9.B3.K` | 13 | 6 | -7 |
| `L9.B4.K` | 13 | 8 | -5 |
| `L9.B5.K` | 13 | 7 | -6 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L1.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L6.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L9.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L11.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L0.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L0.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L0.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L1.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L1.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L1.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L2.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L2.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L2.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L3.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L3.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 408 | 48960 | -0.0007 | +0.3643 | +0.1993 | 0.004 | 0.00000 | 0.00073 | 1.000 | 0.00000 | +0.0474 | 0.00 |
| 409 | 49080 | +0.0013 | +0.3584 | +0.2027 | 0.004 | 0.00000 | 0.00041 | 1.000 | 0.00000 | +0.0953 | 0.00 |
| 410 | 49200 | -0.0010 | +0.3940 | +0.1921 | 0.004 | 0.00000 | -0.00064 | 1.000 | 0.00000 | -0.0487 | 0.00 |
| 411 | 49320 | -0.0020 | +0.2318 | +0.2001 | 0.011 | 0.00000 | -0.00011 | 1.000 | 0.00000 | +0.5360 | 0.00 |
| 412 | 49440 | -0.0008 | +0.3766 | +0.2026 | 0.007 | 0.00000 | 0.00019 | 1.000 | 0.00000 | -0.0068 | 0.00 |
| 413 | 49560 | -0.0009 | +0.2958 | +0.2037 | 0.010 | 0.00000 | 0.00134 | 1.000 | 0.00000 | +0.3331 | 0.00 |
| 414 | 49680 | +0.0013 | +0.2785 | +0.1999 | 0.007 | 0.00000 | 0.00063 | 1.000 | 0.00000 | +0.4129 | 0.00 |
| 415 | 49800 | +0.0043 | +0.2597 | +0.2050 | 0.007 | 0.00000 | 0.00017 | 1.000 | 0.00000 | +0.4258 | 0.00 |
| 416 | 49920 | +0.0027 | +0.3248 | +0.2109 | 0.004 | 0.00000 | 0.00071 | 1.000 | 0.00000 | +0.2505 | 0.00 |
| 417 | 50000 | -0.0010 | +0.2443 | +0.2014 | 0.006 | 0.00000 | 0.00056 | 1.000 | 0.00000 | +0.3176 | 0.00 |

_Entropy 趋势：+1.5943 → +0.2014（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**867** / 877
- **未收敛 slot**：**10** / 877

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
  - slot[665] entropy=0.855 (uniform≈4.159)
  - slot[688] entropy=0.855 (uniform≈4.159)
  - slot[696] entropy=0.855 (uniform≈4.159)
  - slot[713] entropy=0.855 (uniform≈4.159)
  - slot[729] entropy=0.855 (uniform≈4.159)
  - slot[446] entropy=0.719 (uniform≈4.159)
  - slot[469] entropy=0.719 (uniform≈4.159)
  - slot[477] entropy=0.719 (uniform≈4.159)

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
    --action-config Parting Chapter/persistent/rl/bert-base/rte/s1t0.001_s2t0.001_s2st2.0__bertbase_rte_stage2_k3_1gpu_1b34e949_20260731/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。