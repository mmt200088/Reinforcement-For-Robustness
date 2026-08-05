# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=60000）

_更新时间: 2026-08-05 01:02:32_  ·  累计用时: **2h00m27s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692615, 'shared': 675648, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `81f8ec5981cac6eaa26116aa09049d979e38159431ff4b7142403ba34cb947a2`
- `run_context_hash` = `db8b489cc0235eef16eeb38cfceeb7ec49cdc7463bc574441020ed433c6b5f9e`
- `cost_model_revision` = `network_weighted_compute_communication_v3`
- `resource_objective` = `{'compute_axis': 'learnable_block4_fusion_count', 'communication_axis': 'layerwise_precision_preset_utility', 'selection': 'network_weighted_sum_then_balance', 'ppo_surrogate': '(compute+rho*communication)/(1+rho)'}`
- `communication_importance_ratio` = `1.0`
- `network_axis_weights` = `[0.5, 0.5]`
- `compute_axis_denominator` = `12`
- `communication_axis_denominator` = `12`
- `resource_credit_mode` = `separable_weighted_per_slot_v1`
- `strict_resource_order` = `['weighted_score', 'balance_tiebreak']`
- `total_episodes_planned` = `60000`
- `rollout_size` = `120`
- `ppo_lr` = `5e-05`
- `gamma` = `1.0`
- `gae_lambda` = `1.0`
- `entropy_regularization` = `{'kind': 'disabled', 'coefficient': 0.0, 'optimization_role': 'monitor_only'}`
- `termination` = `{'mode': 'convergence_or_max_episodes', 'episode_limit': 60000, 'minimum_episodes': 90000, 'patience_updates': 100, 'requires_robust_feasible_candidate': True, 'frontier_stall_update_windows': 100, 'selected_action_stable_update_windows': 100, 'strict_revalidation_required': True, 'strict_revalidation_trials': 15, 'strict_revalidation_diagnostic_probability': 0.95, 'selection_order': 'feasible,weighted_resource_score,balance_tiebreak,confidence_vector,safety_margin_vector,action_lexicographic', 'entropy_role': 'diagnostic_only', 'validation_banks': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'counts_only_finite_ppo_updates': True}`
- `ppo_mode` = `{'factorized_actor_clip': True, 'behavior_log_prob_source': 'sampling_time_per_slot_v1', 'actor_credit_mode': 'shared_constraint_plus_separable_axis_resource', 'actor_advantage_normalization': 'per_slot_center_shared_scale_v1', 'entropy_average_active_slots': True, 'entropy_normalize_active_slots': True}`
- `stage2_k_trials` = `3`
- `baseline_groups` = `5`
- `baseline_trials_per_group` = `3`
- `constraint_bootstrap_samples` = `4096`
- `constraint_probabilities` = `{'online': 0.5, 'promotion': 0.8, 'final': 0.95}`
- `constraint_limits` = `{'loss': 0.36673159655655424, 'metric1': 0.8691820312499999, 'metric2': 0.8672057724962936, 'loss_std': 0.006548915639219971, 'metric1_std': 0.00750898519315043, 'metric2_std': 0.007604260174128674}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.8700520833333333, 'metric2_mean': 0.8680738463426363, 'loss_mean': 0.366365231325229, 'metric1_std': 0.003754492596575215, 'metric2_std': 0.003802130087064337, 'loss_std': 0.0032744578196099855, 'metric1_threshold': 0.8691820312499999, 'metric2_threshold': 0.8672057724962936, 'loss_threshold': 0.36673159655655424, 'metric1_std_threshold': 0.00750898519315043, 'metric2_std_threshold': 0.007604260174128674, 'loss_std_threshold': 0.006548915639219971, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.3643719367682934, 0.36491648107767105, 0.36902937293052673], 'metric1_trials': [0.8671875, 0.875, 0.875], 'metric2_trials': [0.8657526199222126, 0.8731498947979395, 0.8731498947979395]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.37323790043592453, 0.36398253589868546, 0.3657107576727867], 'metric1_trials': [0.87109375, 0.8671875, 0.8671875], 'metric2_trials': [0.8689182062100606, 0.8652217632228107, 0.8652217632228107]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.3672001361846924, 0.3638284169137478, 0.36494939029216766], 'metric1_trials': [0.875, 0.875, 0.8671875], 'metric2_trials': [0.8731498947979395, 0.8731498947979395, 0.8652217632228107]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.36913568526506424, 0.3716467469930649, 0.36631297692656517], 'metric1_trials': [0.86328125, 0.87109375, 0.87109375], 'metric2_trials': [0.8609738550712764, 0.8689182062100606, 0.8689182062100606]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.3603984862565994, 0.3661981001496315, 0.3645595461130142], 'metric1_trials': [0.8671875, 0.87109375, 0.8671875], 'metric2_trials': [0.8652217632228107, 0.8689182062100606, 0.8652217632228107]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.366365231325229, 'metric1_mean': 0.8700520833333333, 'metric2_mean': 0.8680738463426363, 'loss_std': 0.0032744578196099855, 'metric1_std': 0.003754492596575215, 'metric2_std': 0.003802130087064337, 'limits': {'loss': 0.36673159655655424, 'metric1': 0.8691820312499999, 'metric2': 0.8672057724962936, 'loss_std': 0.006548915639219971, 'metric1_std': 0.00750898519315043, 'metric2_std': 0.007604260174128674}}, 'limits': {'loss': 0.36673159655655424, 'metric1': 0.8691820312499999, 'metric2': 0.8672057724962936, 'loss_std': 0.006548915639219971, 'metric1_std': 0.00750898519315043, 'metric2_std': 0.007604260174128674}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.3643719367682934, 0.36491648107767105, 0.36902937293052673], 'metric1_trials': [0.8671875, 0.875, 0.875], 'metric2_trials': [0.8657526199222126, 0.8731498947979395, 0.8731498947979395]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.37323790043592453, 0.36398253589868546, 0.3657107576727867], 'metric1_trials': [0.87109375, 0.8671875, 0.8671875], 'metric2_trials': [0.8689182062100606, 0.8652217632228107, 0.8652217632228107]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.3672001361846924, 0.3638284169137478, 0.36494939029216766], 'metric1_trials': [0.875, 0.875, 0.8671875], 'metric2_trials': [0.8731498947979395, 0.8731498947979395, 0.8652217632228107]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.36913568526506424, 0.3716467469930649, 0.36631297692656517], 'metric1_trials': [0.86328125, 0.87109375, 0.87109375], 'metric2_trials': [0.8609738550712764, 0.8689182062100606, 0.8689182062100606]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.3603984862565994, 0.3661981001496315, 0.3645595461130142], 'metric1_trials': [0.8671875, 0.87109375, 0.8671875], 'metric2_trials': [0.8652217632228107, 0.8689182062100606, 0.8652217632228107]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.366365231325229, 'metric1_mean': 0.8700520833333333, 'metric2_mean': 0.8680738463426363, 'loss_std': 0.0032744578196099855, 'metric1_std': 0.003754492596575215, 'metric2_std': 0.003802130087064337, 'limits': {'loss': 0.36673159655655424, 'metric1': 0.8691820312499999, 'metric2': 0.8672057724962936, 'loss_std': 0.006548915639219971, 'metric1_std': 0.00750898519315043, 'metric2_std': 0.007604260174128674}}, 'limits': {'loss': 0.36673159655655424, 'metric1': 0.8691820312499999, 'metric2': 0.8672057724962936, 'loss_std': 0.006548915639219971, 'metric1_std': 0.00750898519315043, 'metric2_std': 0.007604260174128674}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374594418853, 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823], 'loss_trials': [0.3467133687991722, 0.34456994369918226, 0.34663345065771367], 'metric1_trials': [0.8823529411764706, 0.8774509803921569, 0.8676470588235294], 'metric2_trials': [0.8804175573703057, 0.8757519695939993, 0.8651163952781273]}, {'group_index': 1001, 'group_probe_seed': 9223369376165613078, 'trial_seeds': [9223369376165613078, 9223369374455707559, 9223369381474450804], 'loss_trials': [0.33863064471413107, 0.3434281279059017, 0.348548338693731], 'metric1_trials': [0.8774509803921569, 0.8774509803921569, 0.875], 'metric2_trials': [0.8760590106101505, 0.8760590106101505, 0.8731066156760428]}, {'group_index': 1002, 'group_probe_seed': 9223369373444064327, 'trial_seeds': [9223369373444064327, 9223369371801271798, 9223369369079723813], 'loss_trials': [0.3461841613638635, 0.33982300290874407, 0.34462553028966864], 'metric1_trials': [0.8725490196078431, 0.8774509803921569, 0.875], 'metric2_trials': [0.8704523538178311, 0.8754349555940685, 0.8734248592957317]}, {'group_index': 1003, 'group_probe_seed': 9223369367499062704, 'trial_seeds': [9223369367499062704, 9223369369146835969, 9223369370794643154], 'loss_trials': [0.34663857198229026, 0.3467714470975539, 0.3405250766698052], 'metric1_trials': [0.8799019607843137, 0.8774509803921569, 0.8799019607843137], 'metric2_trials': [0.8783885903037421, 0.8760590106101505, 0.8780828268260019]}, {'group_index': 1004, 'group_probe_seed': 9223369369139590113, 'trial_seeds': [9223369369139590113, 9223369366492400208, 9223369373499408515], 'loss_trials': [0.34576646837533687, 0.34367938368928197, 0.34467193659614115], 'metric1_trials': [0.8799019607843137, 0.8774509803921569, 0.875], 'metric2_trials': [0.8780828268260019, 0.8754349555940685, 0.8734248592957317]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.3444806302295012, 'metric1_mean': 0.8767973856209149, 'metric2_mean': 0.8750197198201403, 'loss_std': 0.0028536132111306066, 'metric1_std': 0.0035235063158247956, 'metric2_std': 0.003694169070615343, 'limits': {'loss': 0.34482511085973067, 'metric1': 0.875920588235294, 'metric2': 0.8741447001003201, 'loss_std': 0.005707226422261213, 'metric1_std': 0.007047012631649591, 'metric2_std': 0.007388338141230686}}, 'limits': {'loss': 0.34482511085973067, 'metric1': 0.875920588235294, 'metric2': 0.8741447001003201, 'loss_std': 0.005707226422261213, 'metric1_std': 0.007047012631649591, 'metric2_std': 0.007388338141230686}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720425372765, 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903], 'loss_trials': [0.3431911585377712, 0.337412813130547, 0.34209097249835146], 'metric1_trials': [0.8799019607843137, 0.8799019607843137, 0.8823529411764706], 'metric2_trials': [0.8783885903037421, 0.8786846733338582, 0.8810166501857444]}, {'group_index': 2001, 'group_probe_seed': 9223366722063803790, 'trial_seeds': [9223366722063803790, 9223366720019946559, 9223366726961026796], 'loss_trials': [0.34087198505214616, 0.34296826170940026, 0.34335697398466225], 'metric1_trials': [0.8700980392156863, 0.875, 0.8823529411764706], 'metric2_trials': [0.8681304045260836, 0.8731066156760428, 0.8804175573703057]}, {'group_index': 2002, 'group_probe_seed': 9223366719342124031, 'trial_seeds': [9223366719342124031, 9223366717365510734, 9223366714717300893], 'loss_trials': [0.34461691566542085, 0.33892078025668276, 0.34492587341981773], 'metric1_trials': [0.8774509803921569, 0.8823529411764706, 0.875], 'metric2_trials': [0.8757519695939993, 0.8807218908102394, 0.8731066156760428]}, {'group_index': 2003, 'group_probe_seed': 9223366712331899176, 'trial_seeds': [9223366712331899176, 9223366714711074969, 9223366717367542346], 'loss_trials': [0.3461957527141945, 0.34940166099398745, 0.3454670356769188], 'metric1_trials': [0.8774509803921569, 0.8700980392156863, 0.8799019607843137], 'metric2_trials': [0.8760590106101505, 0.868461128287721, 0.8783885903037421]}, {'group_index': 2004, 'group_probe_seed': 9223366713905190553, 'trial_seeds': [9223366713905190553, 9223366712056639272, 9223366717992199675], 'loss_trials': [0.34333768428540695, 0.3404042510425343, 0.34553347148147284], 'metric1_trials': [0.8823529411764706, 0.8799019607843137, 0.8774509803921569], 'metric2_trials': [0.8807218908102394, 0.8780828268260019, 0.8760590106101505]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.343246372696621, 'metric1_mean': 0.8781045751633986, 'metric2_mean': 0.8764731616616043, 'loss_std': 0.0030395035530748084, 'metric1_std': 0.004087300920748966, 'metric2_std': 0.004178594971950085, 'limits': {'loss': 0.34358961906931756, 'metric1': 0.8772264705882352, 'metric2': 0.8755966884999427, 'loss_std': 0.006079007106149617, 'metric1_std': 0.008174601841497932, 'metric2_std': 0.00835718994390017}}, 'limits': {'loss': 0.34358961906931756, 'metric1': 0.8772264705882352, 'metric2': 0.8755966884999427, 'loss_std': 0.006079007106149617, 'metric1_std': 0.008174601841497932, 'metric2_std': 0.00835718994390017}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066322514933, 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119], 'loss_trials': [0.34511508193670537, 0.3432379435090458, 0.347538805475422], 'metric1_trials': [0.8700980392156863, 0.875, 0.8725490196078431], 'metric2_trials': [0.8681304045260836, 0.8737330273474853, 0.8707820483777593]}, {'group_index': 3001, 'group_probe_seed': 9223364067895867686, 'trial_seeds': [9223364067895867686, 9223364065584185495, 9223364072532478532], 'loss_trials': [0.34348442975212545, 0.3419162163547441, 0.3489556674863778], 'metric1_trials': [0.8774509803921569, 0.8799019607843137, 0.8725490196078431], 'metric2_trials': [0.8757519695939993, 0.8783885903037421, 0.8707820483777593]}, {'group_index': 3002, 'group_probe_seed': 9223364065181531799, 'trial_seeds': [9223364065181531799, 9223364062929749798, 9223364060283768309], 'loss_trials': [0.3403668894487269, 0.34595319687151443, 0.34181450979382383], 'metric1_trials': [0.8774509803921569, 0.8799019607843137, 0.8725490196078431], 'metric2_trials': [0.8763562668883809, 0.8783885903037421, 0.8707820483777593]}, {'group_index': 3003, 'group_probe_seed': 9223364058162918592, 'trial_seeds': [9223364058162918592, 9223364060275314033, 9223364062921426850], 'loss_trials': [0.3438196719861498, 0.34222857274261176, 0.3459544485690547], 'metric1_trials': [0.875, 0.875, 0.8725490196078431], 'metric2_trials': [0.8737330273474853, 0.8734248592957317, 0.8707820483777593]}, {'group_index': 3004, 'group_probe_seed': 9223364059736205873, 'trial_seeds': [9223364059736205873, 9223364057620878208, 9223364063562602835], 'loss_trials': [0.34542227609484805, 0.34355547030766803, 0.3443184424849117], 'metric1_trials': [0.875, 0.8774509803921569, 0.8725490196078431], 'metric2_trials': [0.8734248592957317, 0.8757519695939993, 0.8711013710345565]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.34424544152091535, 'metric1_mean': 0.875, 'metric2_mean': 0.873420875269465, 'loss_std': 0.0022946105696462844, 'metric1_std': 0.0029294818856235036, 'metric2_std': 0.0030502766900744626, 'limits': {'loss': 0.34458968696243625, 'metric1': 0.874125, 'metric2': 0.8725474543941956, 'loss_std': 0.004589221139292569, 'metric1_std': 0.005858963771247007, 'metric2_std': 0.006100553380148925}}, 'limits': {'loss': 0.34458968696243625, 'metric1': 0.874125, 'metric2': 0.8725474543941956, 'loss_std': 0.004589221139292569, 'metric1_std': 0.005858963771247007, 'metric2_std': 0.006100553380148925}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.34386350146306105, 'metric1_mean': 0.8774509803921569, 'metric2_mean': 0.8757464407408722, 'loss_std': 0.0029639739187597076, 'metric1_std': 0.0038079379060408283, 'metric2_std': 0.00394509417642693, 'limits': {'loss': 0.3442073649645241, 'metric1': 0.8765735294117647, 'metric2': 0.8748706943001313, 'loss_std': 0.005927947837519415, 'metric1_std': 0.0076158758120816565, 'metric2_std': 0.00789018835285386}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.3439908148156791, 'metric1_mean': 0.8766339869281047, 'metric2_mean': 0.8749712522504033, 'loss_std': 0.002738368831606181, 'metric1_std': 0.003694991967864737, 'metric2_std': 0.0038009880646880136, 'limits': {'loss': 0.34433480563049473, 'metric1': 0.8757573529411766, 'metric2': 0.8740962809981528, 'loss_std': 0.005476737663212362, 'metric1_std': 0.007389983935729474, 'metric2_std': 0.007601976129376027}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 408, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **60000**
- 最近 50 回合 mean return: **+0.9152** (min=-3.2140, max=+1.4379)
- 最近 50 回合 mean terminal reward: **+0.9152**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+1.6465**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **501**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 3159 | 3 | +0.6458 | +1.6465 | +1.6465 | +0.0000 | 12 | 0 | 0 | 30 |
| 2 | 2558 | 3 | +0.6458 | +1.6462 | +1.6462 | +0.0000 | 12 | 0 | 0 | 31 |
| 3 | 2687 | 3 | +0.6250 | +1.6257 | +1.6257 | +0.0000 | 12 | 0 | 0 | 30 |
| 4 | 1722 | 3 | +0.6250 | +1.6253 | +1.6253 | +0.0000 | 12 | 0 | 0 | 29 |
| 5 | 2518 | 3 | +0.6250 | +1.6252 | +1.6252 | +0.0000 | 12 | 0 | 0 | 31 |
| 6 | 1306 | 3 | +0.6250 | +1.6252 | +1.6252 | +0.0000 | 12 | 0 | 0 | 32 |
| 7 | 2170 | 3 | +0.6042 | +1.6049 | +1.6049 | +0.0000 | 12 | 0 | 0 | 29 |
| 8 | 2855 | 3 | +0.6042 | +1.6048 | +1.6048 | +0.0000 | 12 | 0 | 0 | 30 |
| 9 | 1189 | 3 | +0.6042 | +1.6048 | +1.6048 | +0.0000 | 12 | 0 | 0 | 32 |
| 10 | 2469 | 3 | +0.5833 | +1.5840 | +1.5840 | +0.0000 | 12 | 0 | 0 | 29 |
| 11 | 2321 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 28 |
| 12 | 3041 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 29 |
| 13 | 3077 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 30 |
| 14 | 2436 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 30 |
| 15 | 2067 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 29 |
| 16 | 134 | 3 | +0.5833 | +1.5838 | +1.5838 | +0.0000 | 12 | 0 | 0 | 32 |
| 17 | 3679 | 3 | +0.5833 | +1.5837 | +1.5837 | +0.0000 | 12 | 0 | 0 | 28 |
| 18 | 2536 | 3 | +0.5833 | +1.5837 | +1.5837 | +0.0000 | 12 | 0 | 0 | 29 |
| 19 | 3319 | 3 | +0.5833 | +1.5836 | +1.5836 | +0.0000 | 12 | 0 | 0 | 29 |
| 20 | 1923 | 3 | +0.5833 | +1.5836 | +1.5836 | +0.0000 | 12 | 0 | 0 | 29 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 178 个槽与 baseline 不同_（118 SF + 60 K）

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
| `L10.B1.K` | 13 | 7 | -6 |
| `L10.B2.K` | 13 | 6 | -7 |
| `L10.B3.K` | 13 | 6 | -7 |
| `L10.B4.K` | 13 | 8 | -5 |
| `L10.B5.K` | 13 | 7 | -6 |
| `L11.B1.K` | 13 | 7 | -6 |
| `L11.B2.K` | 13 | 6 | -7 |
| `L11.B3.K` | 13 | 6 | -7 |
| `L11.B4.K` | 13 | 8 | -5 |
| `L11.B5.K` | 13 | 7 | -6 |
| `L2.B1.K` | 13 | 7 | -6 |
| `L2.B2.K` | 13 | 6 | -7 |
| `L2.B3.K` | 13 | 6 | -7 |
| `L2.B4.K` | 13 | 8 | -5 |
| `L2.B5.K` | 13 | 7 | -6 |
| `L3.B1.K` | 13 | 7 | -6 |
| `L3.B2.K` | 13 | 6 | -7 |
| `L3.B3.K` | 13 | 6 | -7 |
| `L3.B4.K` | 13 | 8 | -5 |
| `L3.B5.K` | 13 | 7 | -6 |
| `L4.B1.K` | 13 | 7 | -6 |
| `L4.B2.K` | 13 | 6 | -7 |
| `L4.B3.K` | 13 | 6 | -7 |
| `L4.B4.K` | 13 | 8 | -5 |
| `L4.B5.K` | 13 | 7 | -6 |
| `L5.B1.K` | 13 | 7 | -6 |
| `L5.B2.K` | 13 | 6 | -7 |
| `L5.B3.K` | 13 | 6 | -7 |
| `L5.B4.K` | 13 | 8 | -5 |
| `L5.B5.K` | 13 | 7 | -6 |
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
| `L9.B1.K` | 13 | 11 | -2 |
| `L9.B2.K` | 13 | 10 | -3 |
| `L9.B3.K` | 13 | 10 | -3 |
| `L9.B4.K` | 13 | 12 | -1 |
| `L9.B5.K` | 13 | 11 | -2 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L8.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L9.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L11.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.W.wo` | W | 22 | 11 | -11 |
| `L5.B4.W.wo` | W | 22 | 11 | -11 |
| `L8.B4.W.wo` | W | 22 | 11 | -11 |
| `L9.B4.W.wo` | W | 22 | 11 | -11 |
| `L10.B4.W.wo` | W | 22 | 11 | -11 |
| `L11.B4.W.wo` | W | 22 | 11 | -11 |
| `L0.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |
| `L2.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |
| `L3.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |
| `L4.B4.S.ln_mean_inv_d` | S | 20 | 11 | -9 |
| `L4.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |
| `L5.B4.S.ln_mean_inv_d` | S | 20 | 11 | -9 |
| `L5.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |
| `L6.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 492 | 59000 | +0.0001 | +0.2448 | +0.4277 | 0.013 | 0.00000 | 0.00249 | 1.000 | 0.00000 | +0.9233 | 0.00 |
| 493 | 59120 | -0.0038 | +0.2522 | +0.4361 | 0.015 | 0.00000 | 0.00006 | 1.000 | 0.00000 | +0.8923 | 0.00 |
| 494 | 59240 | -0.0026 | +0.2213 | +0.4331 | 0.009 | 0.00000 | 0.00060 | 1.000 | 0.00000 | +0.9562 | 0.00 |
| 495 | 59360 | +0.0001 | +0.2377 | +0.4306 | 0.026 | 0.00000 | 0.00079 | 1.000 | 0.00000 | +0.9244 | 0.00 |
| 496 | 59480 | -0.0041 | +0.1211 | +0.4404 | 0.018 | 0.00000 | -0.00026 | 1.000 | 0.00000 | +1.1428 | 0.00 |
| 497 | 59600 | -0.0022 | +0.1818 | +0.4508 | 0.012 | 0.00000 | 0.00097 | 1.000 | 0.00000 | +1.0254 | 0.00 |
| 498 | 59720 | -0.0019 | +0.2786 | +0.4505 | 0.010 | 0.00000 | 0.00026 | 1.000 | 0.00000 | +0.8512 | 0.00 |
| 499 | 59840 | -0.0005 | +0.2438 | +0.4413 | 0.013 | 0.00000 | 0.00072 | 1.000 | 0.00000 | +0.9250 | 0.00 |
| 500 | 59960 | -0.0020 | +0.2830 | +0.4210 | 0.016 | 0.00000 | 0.00142 | 1.000 | 0.00000 | +0.8489 | 0.00 |
| 501 | 60000 | -0.0029 | +0.2006 | +0.4284 | 0.002 | 0.00000 | 0.00001 | 1.000 | 0.00000 | +0.9766 | 0.00 |

_Entropy 趋势：+1.6030 → +0.4284（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**851** / 877
- **未收敛 slot**：**26** / 877

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
  - slot[081] entropy=1.049 (uniform≈4.159)
  - slot[104] entropy=1.049 (uniform≈4.159)
  - slot[112] entropy=1.049 (uniform≈4.159)
  - slot[129] entropy=1.049 (uniform≈4.159)
  - slot[145] entropy=1.049 (uniform≈4.159)
  - slot[592] entropy=0.725 (uniform≈4.159)
  - slot[615] entropy=0.725 (uniform≈4.159)
  - slot[623] entropy=0.725 (uniform≈4.159)

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
    --action-config /hy-tmp/rfr_stage2_runs_5c222da6/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.001_s2st2.0__bertbase_mrpc_stage2_k3_4gpu_stage1best20260624_5c222da6_20260804/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。