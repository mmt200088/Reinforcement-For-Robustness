# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=50000）

_更新时间: 2026-08-07 02:05:21_  ·  累计用时: **8h05m15s**

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
- `algorithm_contract_hash` = `18cab7242a9cf204f1d6dafa95af5c12c1af01dc17c4d3c21457ede9c5db1535`
- `run_context_hash` = `8b7c2960b32f6686738d1f4f765fda8b350bdd679113b1c04bc5fd59a0187eec`
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
- `constraint_limits` = `{'loss': 0.7649918528348207, 'metric1': 0.7169700520833333, 'metric2': 0.7169700520833333, 'loss_std': 0.014054367524510166, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.010591903986513653}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.7205729166666667, 'metric2_mean': 0.7205729166666667, 'loss_mean': 0.7611859232187271, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005295951993256827, 'loss_std': 0.007027183762255083, 'metric1_threshold': 0.7169700520833333, 'metric2_threshold': 0.7169700520833333, 'loss_threshold': 0.7649918528348207, 'metric1_std_threshold': 0.010591903986513653, 'metric2_std_threshold': 0.010591903986513653, 'loss_std_threshold': 0.014054367524510166, 'limit_tolerance': 0.005, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [0.7621274292469025, 0.769024521112442, 0.7741009816527367], 'metric1_trials': [0.72265625, 0.7109375, 0.71875], 'metric2_trials': [0.72265625, 0.7109375, 0.71875]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [0.7558750733733177, 0.7589182704687119, 0.7563389092683792], 'metric1_trials': [0.7265625, 0.72265625, 0.72265625], 'metric2_trials': [0.7265625, 0.72265625, 0.72265625]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [0.7622007727622986, 0.748275525867939, 0.7566607668995857], 'metric1_trials': [0.71875, 0.7265625, 0.71484375], 'metric2_trials': [0.71875, 0.7265625, 0.71484375]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [0.773246094584465, 0.7661299258470535, 0.7601815462112427], 'metric1_trials': [0.7109375, 0.71875, 0.7265625], 'metric2_trials': [0.7109375, 0.71875, 0.7265625]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [0.761761873960495, 0.7573509514331818, 0.7555962055921555], 'metric1_trials': [0.72265625, 0.7265625, 0.71875], 'metric2_trials': [0.72265625, 0.7265625, 0.71875]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7611859232187271, 'metric1_mean': 0.7205729166666667, 'metric2_mean': 0.7205729166666667, 'loss_std': 0.007027183762255083, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005295951993256827, 'limits': {'loss': 0.7649918528348207, 'metric1': 0.7169700520833333, 'metric2': 0.7169700520833333, 'loss_std': 0.014054367524510166, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.010591903986513653}}, 'limits': {'loss': 0.7649918528348207, 'metric1': 0.7169700520833333, 'metric2': 0.7169700520833333, 'loss_std': 0.014054367524510166, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.010591903986513653}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.005, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [0.7621274292469025, 0.769024521112442, 0.7741009816527367], 'metric1_trials': [0.72265625, 0.7109375, 0.71875], 'metric2_trials': [0.72265625, 0.7109375, 0.71875]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [0.7558750733733177, 0.7589182704687119, 0.7563389092683792], 'metric1_trials': [0.7265625, 0.72265625, 0.72265625], 'metric2_trials': [0.7265625, 0.72265625, 0.72265625]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [0.7622007727622986, 0.748275525867939, 0.7566607668995857], 'metric1_trials': [0.71875, 0.7265625, 0.71484375], 'metric2_trials': [0.71875, 0.7265625, 0.71484375]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [0.773246094584465, 0.7661299258470535, 0.7601815462112427], 'metric1_trials': [0.7109375, 0.71875, 0.7265625], 'metric2_trials': [0.7109375, 0.71875, 0.7265625]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [0.761761873960495, 0.7573509514331818, 0.7555962055921555], 'metric1_trials': [0.72265625, 0.7265625, 0.71875], 'metric2_trials': [0.72265625, 0.7265625, 0.71875]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7611859232187271, 'metric1_mean': 0.7205729166666667, 'metric2_mean': 0.7205729166666667, 'loss_std': 0.007027183762255083, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005295951993256827, 'limits': {'loss': 0.7649918528348207, 'metric1': 0.7169700520833333, 'metric2': 0.7169700520833333, 'loss_std': 0.014054367524510166, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.010591903986513653}}, 'limits': {'loss': 0.7649918528348207, 'metric1': 0.7169700520833333, 'metric2': 0.7169700520833333, 'loss_std': 0.014054367524510166, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.010591903986513653}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.005, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374610485242, 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176], 'loss_trials': [0.7574706112004359, 0.7527782191868724, 0.7460028847632425], 'metric1_trials': [0.7184115534224665, 0.7220216617257156, 0.736462094938712], 'metric2_trials': [0.7184115534224665, 0.7220216617257156, 0.736462094938712]}, {'group_index': 1001, 'group_probe_seed': 9223369376183642441, 'trial_seeds': [9223369376183642441, 9223369374471229688, 9223369381487745579], 'loss_trials': [0.7497955446639216, 0.7514948022925036, 0.7434239256252881], 'metric1_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629], 'metric2_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629]}, {'group_index': 1002, 'group_probe_seed': 9223369373462094616, 'trial_seeds': [9223369373462094616, 9223369371816794793, 9223369369093017722], 'loss_trials': [0.7480020152963025, 0.7464951334877565, 0.7465039263563465], 'metric1_trials': [0.7328519866354629, 0.7292418783322138, 0.7220216619408948], 'metric2_trials': [0.7328519866354629, 0.7292418783322138, 0.7220216619408948]}, {'group_index': 1003, 'group_probe_seed': 9223369367519304431, 'trial_seeds': [9223369367519304431, 9223369369160147806, 9223369370810182029], 'loss_trials': [0.7549279463420275, 0.7404530226969116, 0.7467087771918369], 'metric1_trials': [0.7220216617257156, 0.7328519866354629, 0.7328519866354629], 'metric2_trials': [0.7220216617257156, 0.7328519866354629, 0.7328519866354629]}, {'group_index': 1004, 'group_probe_seed': 9223369369126278334, 'trial_seeds': [9223369369126278334, 9223369366472158479, 9223369373481396188], 'loss_trials': [0.750729343951394, 0.7478134836530858, 0.7401806126863087], 'metric1_trials': [0.7220216617257156, 0.7220216617257156, 0.736462094938712], 'metric2_trials': [0.7220216617257156, 0.7220216617257156, 0.736462094938712]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7481853499596157, 'metric1_mean': 0.7277978350252596, 'metric2_mean': 0.7277978350252596, 'loss_std': 0.004870651447074666, 'metric1_std': 0.0060716091252485645, 'metric2_std': 0.0060716091252485645, 'limits': {'loss': 0.7519262767094137, 'metric1': 0.7241588458501333, 'metric2': 0.7241588458501333, 'loss_std': 0.009741302894149331, 'metric1_std': 0.012143218250497129, 'metric2_std': 0.012143218250497129}}, 'limits': {'loss': 0.7519262767094137, 'metric1': 0.7241588458501333, 'metric2': 0.7241588458501333, 'loss_std': 0.009741302894149331, 'metric1_std': 0.012143218250497129, 'metric2_std': 0.012143218250497129}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.005, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720442862338, 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432], 'loss_trials': [0.7487246669586816, 0.7544800964073154, 0.749240790678706], 'metric1_trials': [0.7328519866354629, 0.7256317700289647, 0.7256317700289647], 'metric2_trials': [0.7328519866354629, 0.7256317700289647, 0.7256317700289647]}, {'group_index': 2001, 'group_probe_seed': 9223366722049835729, 'trial_seeds': [9223366722049835729, 9223366720000359264, 9223366726943404467], 'loss_trials': [0.7451088232684222, 0.7521736703624794, 0.7485704189603509], 'metric1_trials': [0.7292418783322138, 0.7220216617257156, 0.7328519866354629], 'metric2_trials': [0.7292418783322138, 0.7220216617257156, 0.7328519866354629]}, {'group_index': 2002, 'group_probe_seed': 9223366719328304288, 'trial_seeds': [9223366719328304288, 9223366717345776913, 9223366714699796418], 'loss_trials': [0.7439117879213409, 0.7505279213512848, 0.7549202450776359], 'metric1_trials': [0.7328519866354629, 0.7292418783322138, 0.7184115534224665], 'metric2_trials': [0.7328519866354629, 0.7292418783322138, 0.7184115534224665]}, {'group_index': 2003, 'group_probe_seed': 9223366712311789175, 'trial_seeds': [9223366712311789175, 9223366714697633734, 9223366717352133909], 'loss_trials': [0.744356605120084, 0.7470374453799389, 0.741348792929942], 'metric1_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629], 'metric2_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629]}, {'group_index': 2004, 'group_probe_seed': 9223366713885471174, 'trial_seeds': [9223366713885471174, 9223366712042801271, 9223366717976134308], 'loss_trials': [0.7406907139702394, 0.747737707644163, 0.7469494041553043], 'metric1_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629], 'metric2_trials': [0.7292418783322138, 0.7256317700289647, 0.7328519866354629]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7477186060123926, 'metric1_mean': 0.7282791827846808, 'metric2_mean': 0.7282791827846808, 'loss_std': 0.004251081207803092, 'metric1_std': 0.004414437857656371, 'metric2_std': 0.004414437857656371, 'limits': {'loss': 0.7514571990424544, 'metric1': 0.7246377868707574, 'metric2': 0.7246377868707574, 'loss_std': 0.008502162415606184, 'metric1_std': 0.008828875715312741, 'metric2_std': 0.008828875715312741}}, 'limits': {'loss': 0.7514571990424544, 'metric1': 0.7246377868707574, 'metric2': 0.7246377868707574, 'loss_std': 0.008502162415606184, 'metric1_std': 0.008828875715312741, 'metric2_std': 0.008828875715312741}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.005, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066309088426, 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296], 'loss_trials': [0.7467093431131935, 0.7438621503781756, 0.7460311756667678], 'metric1_trials': [0.7256317700289647, 0.7292418783322138, 0.7256317700289647], 'metric2_trials': [0.7256317700289647, 0.7292418783322138, 0.7256317700289647]}, {'group_index': 3001, 'group_probe_seed': 9223364067882557049, 'trial_seeds': [9223364067882557049, 9223364065563944904, 9223364072514465051], 'loss_trials': [0.7531577994246775, 0.7486975210238019, 0.7475596967586972], 'metric1_trials': [0.7256317700289647, 0.7184115534224665, 0.7220216617257156], 'metric2_trials': [0.7256317700289647, 0.7184115534224665, 0.7220216617257156]}, {'group_index': 3002, 'group_probe_seed': 9223364065161271752, 'trial_seeds': [9223364065161271752, 9223364062916452473, 9223364060268243626], 'loss_trials': [0.7457765934699709, 0.7426663335910343, 0.7463100946336877], 'metric1_trials': [0.7256317700289647, 0.7292418783322138, 0.7256317700289647], 'metric2_trials': [0.7256317700289647, 0.7292418783322138, 0.7256317700289647]}, {'group_index': 3003, 'group_probe_seed': 9223364058178311071, 'trial_seeds': [9223364058178311071, 9223364060293474862, 9223364062941552893], 'loss_trials': [0.7486163846852547, 0.7470722174816613, 0.7407148508388643], 'metric1_trials': [0.7292418783322138, 0.736462094938712, 0.7256317700289647], 'metric2_trials': [0.7292418783322138, 0.736462094938712, 0.7256317700289647]}, {'group_index': 3004, 'group_probe_seed': 9223364059751730542, 'trial_seeds': [9223364059751730542, 9223364057638909151, 9223364063582862860], 'loss_trials': [0.7405007995853355, 0.742225334102066, 0.7432831779714095], 'metric1_trials': [0.7292418783322138, 0.7328519866354629, 0.736462094938712], 'metric2_trials': [0.7292418783322138, 0.7328519866354629, 0.736462094938712]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.7455455648483067, 'metric1_mean': 0.7277978350109143, 'metric2_mean': 0.7277978350109143, 'loss_std': 0.0034014491522789438, 'metric1_std': 0.004881757428034733, 'metric2_std': 0.004881757428034733, 'limits': {'loss': 0.7492732926725482, 'metric1': 0.7241588458358598, 'metric2': 0.7241588458358598, 'loss_std': 0.0068028983045578875, 'metric1_std': 0.009763514856069466, 'metric2_std': 0.009763514856069466}}, 'limits': {'loss': 0.7492732926725482, 'metric1': 0.7241588458358598, 'metric2': 0.7241588458358598, 'loss_std': 0.0068028983045578875, 'metric1_std': 0.009763514856069466, 'metric2_std': 0.009763514856069466}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.005, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.7479519779860041, 'metric1_mean': 0.7280385089049701, 'metric2_mean': 0.7280385089049701, 'loss_std': 0.004498133680804319, 'metric1_std': 0.00522151221013147, 'metric2_std': 0.00522151221013147, 'limits': {'loss': 0.751691737875934, 'metric1': 0.7243983163604453, 'metric2': 0.7243983163604453, 'loss_std': 0.008996267361608638, 'metric1_std': 0.01044302442026294, 'metric2_std': 0.01044302442026294}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.7471498402734381, 'metric1_mean': 0.727958284273618, 'metric2_mean': 0.727958284273618, 'loss_std': 0.004281697895725859, 'metric1_std': 0.005056234946028204, 'metric2_std': 0.005056234946028204, 'limits': {'loss': 0.7508855894748052, 'metric1': 0.7243184928522499, 'metric2': 0.7243184928522499, 'loss_std': 0.008563395791451718, 'metric1_std': 0.010112469892056409, 'metric2_std': 0.010112469892056409}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374610485242, 9223369376183642441, 9223369373462094616, 9223369367519304431, 9223369369126278334], 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176, 9223369376183642441, 9223369374471229688, 9223369381487745579, 9223369373462094616, 9223369371816794793, 9223369369093017722, 9223369367519304431, 9223369369160147806, 9223369370810182029, 9223369369126278334, 9223369366472158479, 9223369373481396188], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720442862338, 9223366722049835729, 9223366719328304288, 9223366712311789175, 9223366713885471174], 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432, 9223366722049835729, 9223366720000359264, 9223366726943404467, 9223366719328304288, 9223366717345776913, 9223366714699796418, 9223366712311789175, 9223366714697633734, 9223366717352133909, 9223366713885471174, 9223366712042801271, 9223366717976134308], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066309088426, 9223364067882557049, 9223364065161271752, 9223364058178311071, 9223364059751730542], 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296, 9223364067882557049, 9223364065563944904, 9223364072514465051, 9223364065161271752, 9223364062916452473, 9223364060268243626, 9223364058178311071, 9223364060293474862, 9223364062941552893, 9223364059751730542, 9223364057638909151, 9223364063582862860], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 277, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **50000**
- 最近 50 回合 mean return: **+1.1663** (min=-3.2761, max=+1.5214)
- 最近 50 回合 mean terminal reward: **+1.1663**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+1.6048**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **417**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 1785 | 3 | +0.6042 | +1.6048 | +1.6048 | +0.0000 | 12 | 0 | 0 | 33 |
| 2 | 8781 | 3 | +0.6042 | +1.6047 | +1.6047 | +0.0000 | 12 | 0 | 0 | 28 |
| 3 | 25621 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 28 |
| 4 | 37012 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 28 |
| 5 | 39358 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 28 |
| 6 | 39191 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 27 |
| 7 | 40134 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 28 |
| 8 | 26412 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 29 |
| 9 | 36989 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 27 |
| 10 | 24287 | 3 | +0.5625 | +1.5632 | +1.5632 | +0.0000 | 12 | 0 | 0 | 27 |
| 11 | 22015 | 3 | +0.5625 | +1.5630 | +1.5630 | +0.0000 | 12 | 0 | 0 | 28 |
| 12 | 24271 | 3 | +0.5625 | +1.5630 | +1.5630 | +0.0000 | 12 | 0 | 0 | 28 |
| 13 | 16991 | 3 | +0.5625 | +1.5629 | +1.5629 | +0.0000 | 12 | 0 | 0 | 28 |
| 14 | 25587 | 3 | +0.5625 | +1.5629 | +1.5629 | +0.0000 | 12 | 0 | 0 | 28 |
| 15 | 17621 | 3 | +0.5625 | +1.5629 | +1.5629 | +0.0000 | 12 | 0 | 0 | 29 |
| 16 | 38160 | 3 | +0.5625 | +1.5628 | +1.5628 | +0.0000 | 12 | 0 | 0 | 28 |
| 17 | 45139 | 3 | +0.5625 | +1.5627 | +1.5627 | +0.0000 | 12 | 0 | 0 | 29 |
| 18 | 27693 | 3 | +0.5625 | +1.5627 | +1.5627 | +0.0000 | 12 | 0 | 0 | 28 |
| 19 | 24837 | 3 | +0.5417 | +1.5424 | +1.5424 | +0.0000 | 12 | 0 | 0 | 27 |
| 20 | 40411 | 3 | +0.5417 | +1.5424 | +1.5424 | +0.0000 | 12 | 0 | 0 | 27 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 232 个槽与 baseline 不同_（172 SF + 60 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 9 | -4 |
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 8 | -5 |
| `L0.B4.K` | 13 | 10 | -3 |
| `L0.B5.K` | 13 | 9 | -4 |
| `L1.B1.K` | 13 | 7 | -6 |
| `L1.B2.K` | 13 | 6 | -7 |
| `L1.B3.K` | 13 | 6 | -7 |
| `L1.B4.K` | 13 | 8 | -5 |
| `L1.B5.K` | 13 | 7 | -6 |
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
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 8 | -5 |
| `L4.B3.K` | 13 | 8 | -5 |
| `L4.B4.K` | 13 | 10 | -3 |
| `L4.B5.K` | 13 | 9 | -4 |
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
| `L8.B1.K` | 13 | 11 | -2 |
| `L8.B2.K` | 13 | 10 | -3 |
| `L8.B3.K` | 13 | 10 | -3 |
| `L8.B4.K` | 13 | 12 | -1 |
| `L8.B5.K` | 13 | 11 | -2 |
| `L9.B1.K` | 13 | 9 | -4 |
| `L9.B2.K` | 13 | 8 | -5 |
| `L9.B3.K` | 13 | 8 | -5 |
| `L9.B4.K` | 13 | 10 | -3 |
| `L9.B5.K` | 13 | 9 | -4 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L3.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L7.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
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
| 408 | 48960 | -0.0025 | +0.1413 | +0.7696 | 0.063 | 0.00000 | 0.00373 | 1.000 | 0.00000 | +1.1892 | 0.00 |
| 409 | 49080 | -0.0038 | +0.1705 | +0.7704 | 0.031 | 0.00000 | 0.00275 | 1.000 | 0.00000 | +1.1526 | 0.00 |
| 410 | 49200 | -0.0022 | +0.1826 | +0.7554 | 0.023 | 0.00000 | 0.00218 | 1.000 | 0.00000 | +1.1510 | 0.00 |
| 411 | 49320 | -0.0078 | +0.2047 | +0.7156 | 0.034 | 0.00000 | 0.00250 | 1.000 | 0.00000 | +1.1108 | 0.00 |
| 412 | 49440 | -0.0009 | +0.1801 | +0.6975 | 0.016 | 0.00000 | 0.00227 | 1.000 | 0.00000 | +1.1466 | 0.00 |
| 413 | 49560 | -0.0020 | +0.0941 | +0.7176 | 0.022 | 0.00000 | 0.00172 | 1.000 | 0.00000 | +1.2595 | 0.00 |
| 414 | 49680 | -0.0027 | +0.1503 | +0.7553 | 0.039 | 0.00000 | 0.00339 | 1.000 | 0.00000 | +1.1885 | 0.00 |
| 415 | 49800 | -0.0017 | +0.2323 | +0.7599 | 0.016 | 0.00000 | 0.00140 | 1.000 | 0.00000 | +1.0631 | 0.00 |
| 416 | 49920 | -0.0026 | +0.1438 | +0.7616 | 0.021 | 0.00000 | 0.00115 | 1.000 | 0.00000 | +1.1858 | 0.00 |
| 417 | 50000 | -0.0036 | +0.1374 | +0.7663 | 0.056 | 0.00000 | 0.00172 | 1.000 | 0.00000 | +1.1434 | 0.00 |

_Entropy 趋势：+1.5827 → +0.7663（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**830** / 877
- **未收敛 slot**：**47** / 877

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
  - slot[446] entropy=1.094 (uniform≈4.159)
  - slot[469] entropy=1.094 (uniform≈4.159)
  - slot[477] entropy=1.094 (uniform≈4.159)
  - slot[494] entropy=1.094 (uniform≈4.159)
  - slot[510] entropy=1.094 (uniform≈4.159)
  - slot[008] entropy=1.087 (uniform≈4.159)
  - slot[031] entropy=1.087 (uniform≈4.159)
  - slot[039] entropy=1.087 (uniform≈4.159)

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
    --action-config Parting Chapter/persistent/rl/bert-base/rte/s1t0.001_s2t0.005_s2st2.0/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。