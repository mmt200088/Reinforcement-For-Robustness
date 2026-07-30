# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=24600）

_更新时间: 2026-07-31 00:45:00_  ·  累计用时: **20h05m33s**

**Run meta**：
- `profile` = `mrpc_large`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692999, 'shared': 676032, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `b160c617eb11bc86bd9d615fff62f225c3c7cce1130e25d3d90492f5dfad46c5`
- `run_context_hash` = `65eca6f3b3362a2c41254b9aa6bddec7b40a71422469e0b7028bbab7c45b1ae2`
- `cost_model_revision` = `network_weighted_compute_communication_v3`
- `resource_objective` = `{'compute_axis': 'learnable_block4_fusion_count', 'communication_axis': 'layerwise_precision_preset_utility', 'selection': 'network_weighted_sum_then_balance', 'ppo_surrogate': '(compute+rho*communication)/(1+rho)'}`
- `communication_importance_ratio` = `1.0`
- `network_axis_weights` = `[0.5, 0.5]`
- `compute_axis_denominator` = `24`
- `communication_axis_denominator` = `24`
- `resource_credit_mode` = `separable_weighted_per_slot_v1`
- `strict_resource_order` = `['weighted_score', 'balance_tiebreak']`
- `total_episodes_planned` = `150000`
- `rollout_size` = `120`
- `ppo_lr` = `5e-05`
- `gamma` = `1.0`
- `gae_lambda` = `1.0`
- `entropy_regularization` = `{'kind': 'disabled', 'coefficient': 0.0, 'optimization_role': 'monitor_only'}`
- `termination` = `{'mode': 'convergence_or_max_episodes', 'episode_limit': 150000, 'minimum_episodes': 90000, 'patience_updates': 100, 'requires_robust_feasible_candidate': True, 'frontier_stall_update_windows': 100, 'selected_action_stable_update_windows': 100, 'strict_revalidation_required': True, 'strict_revalidation_trials': 15, 'strict_revalidation_diagnostic_probability': 0.95, 'selection_order': 'feasible,weighted_resource_score,balance_tiebreak,confidence_vector,safety_margin_vector,action_lexicographic', 'entropy_role': 'diagnostic_only', 'validation_banks': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374610485242, 9223369376183642441, 9223369373462094616, 9223369367519304431, 9223369369126278334], 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176, 9223369376183642441, 9223369374471229688, 9223369381487745579, 9223369373462094616, 9223369371816794793, 9223369369093017722, 9223369367519304431, 9223369369160147806, 9223369370810182029, 9223369369126278334, 9223369366472158479, 9223369373481396188], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720442862338, 9223366722049835729, 9223366719328304288, 9223366712311789175, 9223366713885471174], 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432, 9223366722049835729, 9223366720000359264, 9223366726943404467, 9223366719328304288, 9223366717345776913, 9223366714699796418, 9223366712311789175, 9223366714697633734, 9223366717352133909, 9223366713885471174, 9223366712042801271, 9223366717976134308], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066309088426, 9223364067882557049, 9223364065161271752, 9223364058178311071, 9223364059751730542], 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296, 9223364067882557049, 9223364065563944904, 9223364072514465051, 9223364065161271752, 9223364062916452473, 9223364060268243626, 9223364058178311071, 9223364060293474862, 9223364062941552893, 9223364059751730542, 9223364057638909151, 9223364063582862860], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'counts_only_finite_ppo_updates': True}`
- `ppo_mode` = `{'factorized_actor_clip': True, 'behavior_log_prob_source': 'sampling_time_per_slot_v1', 'actor_credit_mode': 'shared_constraint_plus_separable_axis_resource', 'actor_advantage_normalization': 'per_slot_center_shared_scale_v1', 'entropy_average_active_slots': True, 'entropy_normalize_active_slots': True}`
- `stage2_k_trials` = `3`
- `baseline_groups` = `5`
- `baseline_trials_per_group` = `3`
- `constraint_bootstrap_samples` = `4096`
- `constraint_probabilities` = `{'online': 0.5, 'promotion': 0.8, 'final': 0.95}`
- `constraint_limits` = `{'loss': 1.4659895252545672, 'metric1': 0.8798484375, 'metric2': 0.8754977236268819, 'loss_std': 0.06357000968748781, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.011686122223259689}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.8807291666666667, 'metric2_mean': 0.8763740977246065, 'loss_mean': 1.464525000254313, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005843061111629844, 'loss_std': 0.03178500484374391, 'metric1_threshold': 0.8798484375, 'metric2_threshold': 0.8754977236268819, 'loss_threshold': 1.4659895252545672, 'metric1_std_threshold': 0.010591903986513653, 'metric2_std_threshold': 0.011686122223259689, 'loss_std_threshold': 0.06357000968748781, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [1.4654162526130676, 1.4529426991939545, 1.4721398949623108], 'metric1_trials': [0.8828125, 0.88671875, 0.87890625], 'metric2_trials': [0.8783542798913043, 0.8832667913055633, 0.8746122744996712]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [1.4199133515357971, 1.5049909055233002, 1.49202099442482], 'metric1_trials': [0.88671875, 0.875, 0.87890625], 'metric2_trials': [0.8832667913055633, 0.8695755614769699, 0.873979150084333]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [1.4714169800281525, 1.4965392053127289, 1.4732028245925903], 'metric1_trials': [0.8828125, 0.875, 0.875], 'metric2_trials': [0.8789527425331237, 0.8695755614769699, 0.8702445652173912]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [1.439372718334198, 1.4705437123775482, 1.5244487822055817], 'metric1_trials': [0.87890625, 0.875, 0.875], 'metric2_trials': [0.873979150084333, 0.8708829253686653, 0.8702445652173912]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [1.416411578655243, 1.4367862939834595, 1.4317288100719452], 'metric1_trials': [0.890625, 0.8828125, 0.88671875], 'metric2_trials': [0.8870225596975823, 0.8789527425331237, 0.8827018051771118]}], 'pooled': {'trial_count': 15, 'loss_mean': 1.464525000254313, 'metric1_mean': 0.8807291666666667, 'metric2_mean': 0.8763740977246065, 'loss_std': 0.03178500484374391, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005843061111629844, 'limits': {'loss': 1.4659895252545672, 'metric1': 0.8798484375, 'metric2': 0.8754977236268819, 'loss_std': 0.06357000968748781, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.011686122223259689}}, 'limits': {'loss': 1.4659895252545672, 'metric1': 0.8798484375, 'metric2': 0.8754977236268819, 'loss_std': 0.06357000968748781, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.011686122223259689}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029817999954, 'trial_seeds': [9223372029817999954, 9223372031530380259, 9223372034187043120], 'loss_trials': [1.4654162526130676, 1.4529426991939545, 1.4721398949623108], 'metric1_trials': [0.8828125, 0.88671875, 0.87890625], 'metric2_trials': [0.8783542798913043, 0.8832667913055633, 0.8746122744996712]}, {'group_index': 1, 'group_probe_seed': 9223372031391419425, 'trial_seeds': [9223372031391419425, 9223372028875945360, 9223372034811445059], 'loss_trials': [1.4199133515357971, 1.5049909055233002, 1.49202099442482], 'metric1_trials': [0.88671875, 0.875, 0.87890625], 'metric2_trials': [0.8832667913055633, 0.8695755614769699, 0.873979150084333]}, {'group_index': 2, 'group_probe_seed': 9223372024374642672, 'trial_seeds': [9223372024374642672, 9223372026219544129, 9223372020284044434], 'loss_trials': [1.4714169800281525, 1.4965392053127289, 1.4732028245925903], 'metric1_trials': [0.8828125, 0.875, 0.875], 'metric2_trials': [0.8789527425331237, 0.8695755614769699, 0.8702445652173912]}, {'group_index': 3, 'group_probe_seed': 9223372021686649159, 'trial_seeds': [9223372021686649159, 9223372023598659830, 9223372025236959781], 'loss_trials': [1.439372718334198, 1.4705437123775482, 1.5244487822055817], 'metric1_trials': [0.87890625, 0.875, 0.875], 'metric2_trials': [0.873979150084333, 0.8708829253686653, 0.8702445652173912]}, {'group_index': 4, 'group_probe_seed': 9223372023260085014, 'trial_seeds': [9223372023260085014, 9223372020941980327, 9223372027893614708], 'loss_trials': [1.416411578655243, 1.4367862939834595, 1.4317288100719452], 'metric1_trials': [0.890625, 0.8828125, 0.88671875], 'metric2_trials': [0.8870225596975823, 0.8789527425331237, 0.8827018051771118]}], 'pooled': {'trial_count': 15, 'loss_mean': 1.464525000254313, 'metric1_mean': 0.8807291666666667, 'metric2_mean': 0.8763740977246065, 'loss_std': 0.03178500484374391, 'metric1_std': 0.005295951993256827, 'metric2_std': 0.005843061111629844, 'limits': {'loss': 1.4659895252545672, 'metric1': 0.8798484375, 'metric2': 0.8754977236268819, 'loss_std': 0.06357000968748781, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.011686122223259689}}, 'limits': {'loss': 1.4659895252545672, 'metric1': 0.8798484375, 'metric2': 0.8754977236268819, 'loss_std': 0.06357000968748781, 'metric1_std': 0.010591903986513653, 'metric2_std': 0.011686122223259689}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374610485242, 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176], 'loss_trials': [1.43987382627001, 1.4428303054734772, 1.4642451323714911], 'metric1_trials': [0.8823529435139076, 0.8774509827295939, 0.8823529435139076], 'metric2_trials': [0.8780024964430051, 0.8729192671281303, 0.8780024964430051]}, {'group_index': 1001, 'group_probe_seed': 9223369376183642441, 'trial_seeds': [9223369376183642441, 9223369374471229688, 9223369381487745579], 'loss_trials': [1.3981007173949598, 1.4515278105642282, 1.4477227575638716], 'metric1_trials': [0.8848039239060645, 0.8823529435139076, 0.8799019631217507], 'metric2_trials': [0.8803554057407017, 0.8776142035761674, 0.8756547374194432]}, {'group_index': 1002, 'group_probe_seed': 9223369373462094616, 'trial_seeds': [9223369373462094616, 9223369371816794793, 9223369369093017722], 'loss_trials': [1.4711793590994442, 1.470947031881295, 1.4321411871442609], 'metric1_trials': [0.8774509827295939, 0.8823529435139076, 0.8799019631217507], 'metric2_trials': [0.872514795391841, 0.8772143745753889, 0.8752641464105189]}, {'group_index': 1003, 'group_probe_seed': 9223369367519304431, 'trial_seeds': [9223369367519304431, 9223369369160147806, 9223369370810182029], 'loss_trials': [1.4902806796279608, 1.4750620103349872, 1.4653292543747847], 'metric1_trials': [0.8823529435139076, 0.8799019631217507, 0.8848039239060645], 'metric2_trials': [0.8776142035761674, 0.8744477621844, 0.8795723433197307]}, {'group_index': 1004, 'group_probe_seed': 9223369369126278334, 'trial_seeds': [9223369369126278334, 9223369366472158479, 9223369373481396188], 'loss_trials': [1.4263720559138877, 1.4277325330996047, 1.4639130526897954], 'metric1_trials': [0.8823529411764706, 0.8848039239060645, 0.8799019631217507], 'metric2_trials': [0.8772143745753889, 0.8803554057407017, 0.8752641464105189]}], 'pooled': {'trial_count': 15, 'loss_mean': 1.4511505142536039, 'metric1_mean': 0.8815359498940263, 'metric2_mean': 0.8768006772623406, 'loss_std': 0.023871179638843446, 'metric1_std': 0.0023919118864737824, 'metric2_std': 0.0024277322050519413, 'limits': {'loss': 1.4526016647678572, 'metric1': 0.8806544139441322, 'metric2': 0.8759238765850782, 'loss_std': 0.04774235927768689, 'metric1_std': 0.004783823772947565, 'metric2_std': 0.004855464410103883}}, 'limits': {'loss': 1.4526016647678572, 'metric1': 0.8806544139441322, 'metric2': 0.8759238765850782, 'loss_std': 0.04774235927768689, 'metric1_std': 0.004783823772947565, 'metric2_std': 0.004855464410103883}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720442862338, 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432], 'loss_trials': [1.4255255764605952, 1.4087055290446562, 1.4745950184616388], 'metric1_trials': [0.8872549042982214, 0.8848039239060645, 0.8799019607843137], 'metric2_trials': [0.8823304423014144, 0.8799695857939744, 0.874861908593718]}, {'group_index': 2001, 'group_probe_seed': 9223366722049835729, 'trial_seeds': [9223366722049835729, 9223366720000359264, 9223366726943404467], 'loss_trials': [1.4726967951830696, 1.4508690319809259, 1.383217082304113], 'metric1_trials': [0.8774509827295939, 0.8872549042982214, 0.8946078431372549], 'metric2_trials': [0.872514795391841, 0.8823304423014144, 0.8905379244010676]}, {'group_index': 2002, 'group_probe_seed': 9223366719328304288, 'trial_seeds': [9223366719328304288, 9223366717345776913, 9223366714699796418], 'loss_trials': [1.4402457031549192, 1.4161935133092545, 1.4477879580329447], 'metric1_trials': [0.8774509827295939, 0.8872549019607843, 0.8848039239060645], 'metric2_trials': [0.8720983068493634, 0.8827136117604937, 0.8795723433197307]}, {'group_index': 2003, 'group_probe_seed': 9223366712311789175, 'trial_seeds': [9223366712311789175, 9223366714697633734, 9223366717352133909], 'loss_trials': [1.453993886124854, 1.4019295608296114, 1.4869157800487443], 'metric1_trials': [0.8848039239060645, 0.8872549042982214, 0.8725490219452802], 'metric2_trials': [0.8799695857939744, 0.88308572575788, 0.8674153872075145]}, {'group_index': 2004, 'group_probe_seed': 9223366713885471174, 'trial_seeds': [9223366713885471174, 9223366712042801271, 9223366717976134308], 'loss_trials': [1.4188810657052433, 1.4269084042193843, 1.439177629994411], 'metric1_trials': [0.8872549042982214, 0.8848039239060645, 0.8848039239060645], 'metric2_trials': [0.8827136117604937, 0.8795723433197307, 0.8803554057407017]}], 'pooled': {'trial_count': 15, 'loss_mean': 1.4365095023236245, 'metric1_mean': 0.8841503286673351, 'metric2_mean': 0.8793360946862209, 'loss_std': 0.028916437721309547, 'metric1_std': 0.0053591636914910685, 'metric2_std': 0.005626967904869104, 'limits': {'loss': 1.437946011825948, 'metric1': 0.8832661783386678, 'metric2': 0.8784567585915346, 'loss_std': 0.05783287544261909, 'metric1_std': 0.010718327382982137, 'metric2_std': 0.011253935809738208}}, 'limits': {'loss': 1.437946011825948, 'metric1': 0.8832661783386678, 'metric2': 0.8784567585915346, 'loss_std': 0.05783287544261909, 'metric1_std': 0.010718327382982137, 'metric2_std': 0.011253935809738208}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066309088426, 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296], 'loss_trials': [1.3838641316282982, 1.4621210659251493, 1.450803939033957], 'metric1_trials': [0.8897058846903783, 0.8823529435139076, 0.8872549042982214], 'metric2_trials': [0.8854466650708848, 0.8776142035761674, 0.8823304423014144]}, {'group_index': 3001, 'group_probe_seed': 9223364067882557049, 'trial_seeds': [9223364067882557049, 9223364065563944904, 9223364072514465051], 'loss_trials': [1.4549369625016755, 1.4171423257565965, 1.4369392722260719], 'metric1_trials': [0.8848039239060645, 0.8848039215686274, 0.8799019631217507], 'metric2_trials': [0.8799695857939744, 0.8803554057407017, 0.874861908593718]}, {'group_index': 3002, 'group_probe_seed': 9223364065161271752, 'trial_seeds': [9223364065161271752, 9223364062916452473, 9223364060268243626], 'loss_trials': [1.458494032130522, 1.476066364961512, 1.4358822738423067], 'metric1_trials': [0.8823529435139076, 0.8799019631217507, 0.8848039239060645], 'metric2_trials': [0.8768027481262776, 0.8752641464105189, 0.8795723433197307]}, {'group_index': 3003, 'group_probe_seed': 9223364058178311071, 'trial_seeds': [9223364058178311071, 9223364060293474862, 9223364062941552893], 'loss_trials': [1.4992342696470373, 1.3867665225384282, 1.480980391595878], 'metric1_trials': [0.8725490219452802, 0.8872549019607843, 0.8799019607843137], 'metric2_trials': [0.8669822391233379, 0.8823304423014144, 0.874861908593718]}, {'group_index': 3004, 'group_probe_seed': 9223364059751730542, 'trial_seeds': [9223364059751730542, 9223364057638909151, 9223364063582862860], 'loss_trials': [1.4443284249773212, 1.4240769077749813, 1.4314446309033562], 'metric1_trials': [0.8872549042982214, 0.8823529435139076, 0.8848039239060645], 'metric2_trials': [0.8823304423014144, 0.8780024964430051, 0.8795723433197307]}], 'pooled': {'trial_count': 15, 'loss_mean': 1.4428721010295393, 'metric1_mean': 0.8833333352032829, 'metric2_mean': 0.8784198214010672, 'loss_std': 0.032088781358337176, 'metric1_std': 0.004224958782178252, 'metric2_std': 0.004420669274359838, 'limits': {'loss': 1.4443149731305687, 'metric1': 0.8824500018680796, 'metric2': 0.8775414015796662, 'loss_std': 0.06417756271667435, 'metric1_std': 0.008449917564356504, 'metric2_std': 0.008841338548719676}}, 'limits': {'loss': 1.4443149731305687, 'metric1': 0.8824500018680796, 'metric2': 0.8775414015796662, 'loss_std': 0.06417756271667435, 'metric1_std': 0.008449917564356504, 'metric2_std': 0.008841338548719676}, 'bootstrap': {'samples': 4096, 'seed': 20260725}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 1.4438300082886144, 'metric1_mean': 0.8828431392806808, 'metric2_mean': 0.8780683859742807, 'loss_std': 0.02709600075928488, 'metric1_std': 0.004288913591259322, 'metric2_std': 0.0044489661015755955, 'limits': {'loss': 1.4452738382969028, 'metric1': 0.8819602961414001, 'metric2': 0.8771903175883065, 'loss_std': 0.05419200151856976, 'metric1_std': 0.008577827182518644, 'metric2_std': 0.008897932203151191}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 1.4435107058689227, 'metric1_mean': 0.8830065379215484, 'metric2_mean': 0.8781855311165429, 'loss_std': 0.02849100287304511, 'metric1_std': 0.004225881856290228, 'metric2_std': 0.004392227603376292, 'limits': {'loss': 1.4449542165747915, 'metric1': 0.8821235313836269, 'metric2': 0.8773073455854263, 'loss_std': 0.05698200574609022, 'metric1_std': 0.008451763712580456, 'metric2_std': 0.008784455206752584}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374610485242, 9223369376183642441, 9223369373462094616, 9223369367519304431, 9223369369126278334], 'trial_seeds': [9223369374610485242, 9223369377127634507, 9223369378701057176, 9223369376183642441, 9223369374471229688, 9223369381487745579, 9223369373462094616, 9223369371816794793, 9223369369093017722, 9223369367519304431, 9223369369160147806, 9223369370810182029, 9223369369126278334, 9223369366472158479, 9223369373481396188], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720442862338, 9223366722049835729, 9223366719328304288, 9223366712311789175, 9223366713885471174], 'trial_seeds': [9223366720442862338, 9223366722690446003, 9223366724270094432, 9223366722049835729, 9223366720000359264, 9223366726943404467, 9223366719328304288, 9223366717345776913, 9223366714699796418, 9223366712311789175, 9223366714697633734, 9223366717352133909, 9223366713885471174, 9223366712042801271, 9223366717976134308], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066309088426, 9223364067882557049, 9223364065161271752, 9223364058178311071, 9223364059751730542], 'trial_seeds': [9223364066309088426, 9223364068218494235, 9223364069859940296, 9223364067882557049, 9223364065563944904, 9223364072514465051, 9223364065161271752, 9223364062916452473, 9223364060268243626, 9223364058178311071, 9223364060293474862, 9223364062941552893, 9223364059751730542, 9223364057638909151, 9223364063582862860], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 408, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **24600**
- 最近 50 回合 mean return: **+0.9076** (min=-3.5000, max=+1.4590)
- 最近 50 回合 mean terminal reward: **+0.9076**
- 最近 50 回合 mean invalid 子步数: **0.00** / 95
- 训练期 best reward: **+1.5524**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **205**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 515 | 3 | +0.5521 | +1.5524 | +1.5524 | +0.0000 | 24 | 0 | 0 | 61 |
| 2 | 1749 | 3 | +0.5417 | +1.5424 | +1.5424 | +0.0000 | 24 | 0 | 0 | 60 |
| 3 | 21212 | 3 | +0.5312 | +1.5319 | +1.5319 | +0.0000 | 24 | 0 | 0 | 60 |
| 4 | 2466 | 3 | +0.5312 | +1.5319 | +1.5319 | +0.0000 | 24 | 0 | 0 | 62 |
| 5 | 23523 | 3 | +0.5208 | +1.5215 | +1.5215 | +0.0000 | 24 | 0 | 0 | 59 |
| 6 | 19394 | 3 | +0.5208 | +1.5213 | +1.5213 | +0.0000 | 24 | 0 | 0 | 59 |
| 7 | 1922 | 3 | +0.5208 | +1.5213 | +1.5213 | +0.0000 | 24 | 0 | 0 | 62 |
| 8 | 821 | 3 | +0.5208 | +1.5212 | +1.5212 | +0.0000 | 24 | 0 | 0 | 61 |
| 9 | 1219 | 3 | +0.5208 | +1.5210 | +1.5210 | +0.0000 | 24 | 0 | 0 | 63 |
| 10 | 18560 | 3 | +0.5104 | +1.5111 | +1.5111 | +0.0000 | 24 | 0 | 0 | 60 |
| 11 | 1012 | 3 | +0.5104 | +1.5111 | +1.5111 | +0.0000 | 24 | 0 | 0 | 63 |
| 12 | 21201 | 3 | +0.5104 | +1.5110 | +1.5110 | +0.0000 | 24 | 0 | 0 | 59 |
| 13 | 5261 | 3 | +0.5104 | +1.5110 | +1.5110 | +0.0000 | 24 | 0 | 0 | 60 |
| 14 | 6432 | 3 | +0.5104 | +1.5110 | +1.5110 | +0.0000 | 24 | 0 | 0 | 60 |
| 15 | 22372 | 3 | +0.5104 | +1.5110 | +1.5110 | +0.0000 | 24 | 0 | 0 | 59 |
| 16 | 1264 | 3 | +0.5104 | +1.5108 | +1.5108 | +0.0000 | 24 | 0 | 0 | 62 |
| 17 | 723 | 3 | +0.5104 | +1.5106 | +1.5106 | +0.0000 | 24 | 0 | 0 | 60 |
| 18 | 14450 | 3 | +0.5000 | +1.5007 | +1.5007 | +0.0000 | 24 | 0 | 0 | 61 |
| 19 | 21989 | 3 | +0.5000 | +1.5007 | +1.5007 | +0.0000 | 24 | 0 | 0 | 59 |
| 20 | 20959 | 3 | +0.5000 | +1.5007 | +1.5007 | +0.0000 | 24 | 0 | 0 | 59 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 436 个槽与 baseline 不同_（316 SF + 120 K）

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
| `L11.B1.K` | 13 | 9 | -4 |
| `L11.B2.K` | 13 | 8 | -5 |
| `L11.B3.K` | 13 | 8 | -5 |
| `L11.B4.K` | 13 | 10 | -3 |
| `L11.B5.K` | 13 | 9 | -4 |
| `L12.B1.K` | 13 | 11 | -2 |
| `L12.B2.K` | 13 | 10 | -3 |
| `L12.B3.K` | 13 | 10 | -3 |
| `L12.B4.K` | 13 | 12 | -1 |
| `L12.B5.K` | 13 | 11 | -2 |
| `L13.B1.K` | 13 | 9 | -4 |
| `L13.B2.K` | 13 | 8 | -5 |
| `L13.B3.K` | 13 | 8 | -5 |
| `L13.B4.K` | 13 | 10 | -3 |
| `L13.B5.K` | 13 | 9 | -4 |
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
| `L17.B1.K` | 13 | 7 | -6 |
| `L17.B2.K` | 13 | 6 | -7 |
| `L17.B3.K` | 13 | 6 | -7 |
| `L17.B4.K` | 13 | 8 | -5 |
| `L17.B5.K` | 13 | 7 | -6 |
| `L18.B1.K` | 13 | 7 | -6 |
| `L18.B2.K` | 13 | 6 | -7 |
| `L18.B3.K` | 13 | 6 | -7 |
| `L18.B4.K` | 13 | 8 | -5 |
| `L18.B5.K` | 13 | 7 | -6 |
| `L19.B1.K` | 13 | 9 | -4 |
| `L19.B2.K` | 13 | 8 | -5 |
| `L19.B3.K` | 13 | 8 | -5 |
| `L19.B4.K` | 13 | 10 | -3 |
| `L19.B5.K` | 13 | 9 | -4 |
| `L2.B1.K` | 13 | 11 | -2 |
| `L2.B2.K` | 13 | 10 | -3 |
| `L2.B3.K` | 13 | 10 | -3 |
| `L2.B4.K` | 13 | 12 | -1 |
| `L2.B5.K` | 13 | 11 | -2 |
| `L20.B1.K` | 13 | 9 | -4 |
| `L20.B2.K` | 13 | 8 | -5 |
| `L20.B3.K` | 13 | 8 | -5 |
| `L20.B4.K` | 13 | 10 | -3 |
| `L20.B5.K` | 13 | 9 | -4 |
| `L21.B1.K` | 13 | 9 | -4 |
| `L21.B2.K` | 13 | 8 | -5 |
| `L21.B3.K` | 13 | 8 | -5 |
| `L21.B4.K` | 13 | 10 | -3 |
| `L21.B5.K` | 13 | 9 | -4 |
| `L22.B1.K` | 13 | 7 | -6 |
| `L22.B2.K` | 13 | 6 | -7 |
| `L22.B3.K` | 13 | 6 | -7 |
| `L22.B4.K` | 13 | 8 | -5 |
| `L22.B5.K` | 13 | 7 | -6 |
| `L23.B1.K` | 13 | 9 | -4 |
| `L23.B2.K` | 13 | 8 | -5 |
| `L23.B3.K` | 13 | 8 | -5 |
| `L23.B4.K` | 13 | 10 | -3 |
| `L23.B5.K` | 13 | 9 | -4 |
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
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B2.K` | 13 | 8 | -5 |
| `L5.B3.K` | 13 | 8 | -5 |
| `L5.B4.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 9 | -4 |
| `L6.B1.K` | 13 | 7 | -6 |
| `L6.B2.K` | 13 | 6 | -7 |
| `L6.B3.K` | 13 | 6 | -7 |
| `L6.B4.K` | 13 | 8 | -5 |
| `L6.B5.K` | 13 | 7 | -6 |
| `L7.B1.K` | 13 | 7 | -6 |
| `L7.B2.K` | 13 | 6 | -7 |
| `L7.B3.K` | 13 | 6 | -7 |
| `L7.B4.K` | 13 | 8 | -5 |
| `L7.B5.K` | 13 | 7 | -6 |
| `L8.B1.K` | 13 | 11 | -2 |
| `L8.B2.K` | 13 | 10 | -3 |
| `L8.B3.K` | 13 | 10 | -3 |
| `L8.B4.K` | 13 | 12 | -1 |
| `L8.B5.K` | 13 | 11 | -2 |
| `L9.B1.K` | 13 | 7 | -6 |
| `L9.B2.K` | 13 | 6 | -7 |
| `L9.B3.K` | 13 | 6 | -7 |
| `L9.B4.K` | 13 | 8 | -5 |
| `L9.B5.K` | 13 | 7 | -6 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L1.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L6.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L7.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L8.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L15.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L17.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L18.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L19.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L20.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L21.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L0.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L0.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L0.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L1.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L1.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L1.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L2.B2.R.gamma_r` | R | 28 | 15 | -13 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 196 | 23520 | -0.0027 | +0.1774 | +0.8173 | 0.033 | 0.00000 | 0.00409 | 1.000 | 0.00000 | +0.9099 | 0.00 |
| 197 | 23640 | -0.0023 | +0.3180 | +0.8184 | 0.025 | 0.00000 | 0.00102 | 1.000 | 0.00000 | +0.5193 | 0.00 |
| 198 | 23760 | -0.0029 | +0.1819 | +0.8157 | 0.040 | 0.00000 | 0.00236 | 1.000 | 0.00000 | +0.9100 | 0.00 |
| 199 | 23880 | -0.0036 | +0.2552 | +0.8127 | 0.033 | 0.00000 | 0.00464 | 1.000 | 0.00000 | +0.6908 | 0.00 |
| 200 | 24000 | -0.0033 | +0.2160 | +0.7932 | 0.036 | 0.00000 | 0.00165 | 1.000 | 0.00000 | +0.8163 | 0.00 |
| 201 | 24120 | -0.0027 | +0.2398 | +0.7638 | 0.027 | 0.00000 | 0.00096 | 1.000 | 0.00000 | +0.7856 | 0.00 |
| 202 | 24240 | -0.0020 | +0.2011 | +0.7441 | 0.024 | 0.00000 | 0.00135 | 1.000 | 0.00000 | +0.8642 | 0.00 |
| 203 | 24360 | -0.0022 | +0.1137 | +0.7564 | 0.029 | 0.00000 | 0.00298 | 1.000 | 0.00000 | +1.1011 | 0.00 |
| 204 | 24480 | -0.0015 | +0.1958 | +0.7698 | 0.020 | 0.00000 | 0.00071 | 1.000 | 0.00000 | +0.8773 | 0.00 |
| 205 | 24600 | -0.0023 | +0.1578 | +0.7779 | 0.029 | 0.00000 | 0.00230 | 1.000 | 0.00000 | +0.9544 | 0.00 |

_Entropy 趋势：+1.5916 → +0.7779（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**1575** / 1753
- **未收敛 slot**：**178** / 1753

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
  - slot[1614] entropy=1.061 (uniform≈4.159)
  - slot[1637] entropy=1.061 (uniform≈4.159)
  - slot[1645] entropy=1.061 (uniform≈4.159)
  - slot[1678] entropy=1.061 (uniform≈4.159)
  - slot[1662] entropy=1.061 (uniform≈4.159)
  - slot[446] entropy=1.055 (uniform≈4.159)
  - slot[469] entropy=1.055 (uniform≈4.159)
  - slot[477] entropy=1.055 (uniform≈4.159)

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
    --action-config /hy-tmp/Reinforcement-For-Robustness/Parting Chapter/persistent/rl/bert-large/mrpc/s1t0.001_s2t0.001_s2st2.0__bertlarge_mrpc_stage2_k3_4gpu_0764e710_20260730/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。