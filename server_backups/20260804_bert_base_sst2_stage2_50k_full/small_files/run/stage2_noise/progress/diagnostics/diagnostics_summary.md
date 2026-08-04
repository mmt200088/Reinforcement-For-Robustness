# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=50000）

_更新时间: 2026-08-04 01:50:54_  ·  累计用时: **11h37m55s**

**Run meta**：
- `profile` = `sst2`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692615, 'shared': 675648, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `3e5a6b1b85b1fea7e6061667fb93f6b9e5277f520208b7db65f82fcfca03fc8a`
- `run_context_hash` = `ad31df9bfb3f45131a538f4b7ae02c2dcd21c34621762103dde6a5203c118fdb`
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
- `termination` = `{'mode': 'convergence_or_max_episodes', 'episode_limit': 50000, 'minimum_episodes': 90000, 'patience_updates': 100, 'requires_robust_feasible_candidate': True, 'frontier_stall_update_windows': 100, 'selected_action_stable_update_windows': 100, 'strict_revalidation_required': True, 'strict_revalidation_trials': 15, 'strict_revalidation_diagnostic_probability': 0.95, 'selection_order': 'feasible,weighted_resource_score,balance_tiebreak,confidence_vector,safety_margin_vector,action_lexicographic', 'entropy_role': 'diagnostic_only', 'validation_banks': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'counts_only_finite_ppo_updates': True}`
- `ppo_mode` = `{'factorized_actor_clip': True, 'behavior_log_prob_source': 'sampling_time_per_slot_v1', 'actor_credit_mode': 'shared_constraint_plus_separable_axis_resource', 'actor_advantage_normalization': 'per_slot_center_shared_scale_v1', 'entropy_average_active_slots': True, 'entropy_normalize_active_slots': True}`
- `stage2_k_trials` = `3`
- `baseline_groups` = `5`
- `baseline_trials_per_group` = `3`
- `constraint_bootstrap_samples` = `4096`
- `constraint_probabilities` = `{'online': 0.5, 'promotion': 0.8, 'final': 0.95}`
- `constraint_limits` = `{'loss': 0.3652359535373747, 'metric1': 0.8907750000000001, 'metric2': 0.8907750000000001, 'loss_std': 0.008336110293608338, 'metric1_std': 0.0046376315601926845, 'metric2_std': 0.0046376315601926845}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.8916666666666667, 'metric2_mean': 0.8916666666666667, 'loss_mean': 0.36487108245491984, 'metric1_std': 0.0023188157800963422, 'metric2_std': 0.0023188157800963422, 'loss_std': 0.004168055146804169, 'metric1_threshold': 0.8907750000000001, 'metric2_threshold': 0.8907750000000001, 'loss_threshold': 0.3652359535373747, 'metric1_std_threshold': 0.0046376315601926845, 'metric2_std_threshold': 0.0046376315601926845, 'loss_std_threshold': 0.008336110293608338, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.3620949387550354, 0.3675203360617161, 0.37134650349617004], 'metric1_trials': [0.890625, 0.890625, 0.890625], 'metric2_trials': [0.890625, 0.890625, 0.890625]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.3594938889145851, 0.3642980419099331, 0.3690343126654625], 'metric1_trials': [0.89453125, 0.890625, 0.890625], 'metric2_trials': [0.89453125, 0.890625, 0.890625]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.35955433920025826, 0.3729441463947296, 0.36488429829478264], 'metric1_trials': [0.89453125, 0.89453125, 0.890625], 'metric2_trials': [0.89453125, 0.89453125, 0.890625]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.36326731741428375, 0.36366962268948555, 0.36704543232917786], 'metric1_trials': [0.890625, 0.89453125, 0.88671875], 'metric2_trials': [0.890625, 0.89453125, 0.88671875]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.365468867123127, 0.358896940946579, 0.3635472506284714], 'metric1_trials': [0.890625, 0.89453125, 0.890625], 'metric2_trials': [0.890625, 0.89453125, 0.890625]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.36487108245491984, 'metric1_mean': 0.8916666666666667, 'metric2_mean': 0.8916666666666667, 'loss_std': 0.004168055146804169, 'metric1_std': 0.0023188157800963422, 'metric2_std': 0.0023188157800963422, 'limits': {'loss': 0.3652359535373747, 'metric1': 0.8907750000000001, 'metric2': 0.8907750000000001, 'loss_std': 0.008336110293608338, 'metric1_std': 0.0046376315601926845, 'metric2_std': 0.0046376315601926845}}, 'limits': {'loss': 0.3652359535373747, 'metric1': 0.8907750000000001, 'metric2': 0.8907750000000001, 'loss_std': 0.008336110293608338, 'metric1_std': 0.0046376315601926845, 'metric2_std': 0.0046376315601926845}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.3620949387550354, 0.3675203360617161, 0.37134650349617004], 'metric1_trials': [0.890625, 0.890625, 0.890625], 'metric2_trials': [0.890625, 0.890625, 0.890625]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.3594938889145851, 0.3642980419099331, 0.3690343126654625], 'metric1_trials': [0.89453125, 0.890625, 0.890625], 'metric2_trials': [0.89453125, 0.890625, 0.890625]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.35955433920025826, 0.3729441463947296, 0.36488429829478264], 'metric1_trials': [0.89453125, 0.89453125, 0.890625], 'metric2_trials': [0.89453125, 0.89453125, 0.890625]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.36326731741428375, 0.36366962268948555, 0.36704543232917786], 'metric1_trials': [0.890625, 0.89453125, 0.88671875], 'metric2_trials': [0.890625, 0.89453125, 0.88671875]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.365468867123127, 0.358896940946579, 0.3635472506284714], 'metric1_trials': [0.890625, 0.89453125, 0.890625], 'metric2_trials': [0.890625, 0.89453125, 0.890625]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.36487108245491984, 'metric1_mean': 0.8916666666666667, 'metric2_mean': 0.8916666666666667, 'loss_std': 0.004168055146804169, 'metric1_std': 0.0023188157800963422, 'metric2_std': 0.0023188157800963422, 'limits': {'loss': 0.3652359535373747, 'metric1': 0.8907750000000001, 'metric2': 0.8907750000000001, 'loss_std': 0.008336110293608338, 'metric1_std': 0.0046376315601926845, 'metric2_std': 0.0046376315601926845}}, 'limits': {'loss': 0.3652359535373747, 'metric1': 0.8907750000000001, 'metric2': 0.8907750000000001, 'loss_std': 0.008336110293608338, 'metric1_std': 0.0046376315601926845, 'metric2_std': 0.0046376315601926845}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374594418853, 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823], 'loss_trials': [0.27910497483857183, 0.28008785686635096, 0.27976851165294647], 'metric1_trials': [0.9220183497175164, 0.9231651387083422, 0.9220183497175164], 'metric2_trials': [0.9220183497175164, 0.9231651387083422, 0.9220183497175164]}, {'group_index': 1001, 'group_probe_seed': 9223369376165613078, 'trial_seeds': [9223369376165613078, 9223369374455707559, 9223369381474450804], 'loss_trials': [0.27737544873438846, 0.28066762331702294, 0.27886779872922723], 'metric1_trials': [0.9231651387083422, 0.9208715607266907, 0.9220183497175164], 'metric2_trials': [0.9231651387083422, 0.9208715607266907, 0.9220183497175164]}, {'group_index': 1002, 'group_probe_seed': 9223369373444064327, 'trial_seeds': [9223369373444064327, 9223369371801271798, 9223369369079723813], 'loss_trials': [0.27777130389158877, 0.27431155891593445, 0.27724376792481187], 'metric1_trials': [0.9243119276991678, 0.9243119276991678, 0.9243119276991678], 'metric2_trials': [0.9243119276991678, 0.9243119276991678, 0.9243119276991678]}, {'group_index': 1003, 'group_probe_seed': 9223369367499062704, 'trial_seeds': [9223369367499062704, 9223369369146835969, 9223369370794643154], 'loss_trials': [0.27691450756077374, 0.2783252401362865, 0.2771784350139285], 'metric1_trials': [0.9243119276991678, 0.9231651387083422, 0.9254587166899935], 'metric2_trials': [0.9243119276991678, 0.9231651387083422, 0.9254587166899935]}, {'group_index': 1004, 'group_probe_seed': 9223369369139590113, 'trial_seeds': [9223369369139590113, 9223369366492400208, 9223369373499408515], 'loss_trials': [0.2806311330390633, 0.28111104626174366, 0.2785301578974505], 'metric1_trials': [0.9266055056808191, 0.9243119276991678, 0.9220183497175164], 'metric2_trials': [0.9266055056808191, 0.9243119276991678, 0.9220183497175164]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.278525957652006, 'metric1_mean': 0.9234709491058957, 'metric2_mean': 0.9234709491058957, 'loss_std': 0.0018104458461781826, 'metric1_std': 0.0015304166038247861, 'metric2_std': 0.0015304166038247861, 'limits': {'loss': 0.278804483609658, 'metric1': 0.9225474781567897, 'metric2': 0.9225474781567897, 'loss_std': 0.003620891692356365, 'metric1_std': 0.0030608332076495722, 'metric2_std': 0.0030608332076495722}}, 'limits': {'loss': 0.278804483609658, 'metric1': 0.9225474781567897, 'metric2': 0.9225474781567897, 'loss_std': 0.003620891692356365, 'metric1_std': 0.0030608332076495722, 'metric2_std': 0.0030608332076495722}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720425372765, 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903], 'loss_trials': [0.2815989446612673, 0.27639198070819226, 0.27415892310919016], 'metric1_trials': [0.9231651387083422, 0.9231651387083422, 0.9243119276991678], 'metric2_trials': [0.9231651387083422, 0.9231651387083422, 0.9243119276991678]}, {'group_index': 2001, 'group_probe_seed': 9223366722063803790, 'trial_seeds': [9223366722063803790, 9223366720019946559, 9223366726961026796], 'loss_trials': [0.27729049851314735, 0.2797071719798473, 0.27544426747144907], 'metric1_trials': [0.9254587166899935, 0.9243119276991678, 0.9266055056808191], 'metric2_trials': [0.9254587166899935, 0.9243119276991678, 0.9266055056808191]}, {'group_index': 2002, 'group_probe_seed': 9223366719342124031, 'trial_seeds': [9223366719342124031, 9223366717365510734, 9223366714717300893], 'loss_trials': [0.27794860433274454, 0.2792020851592405, 0.2779196286146794], 'metric1_trials': [0.9231651387083422, 0.9243119276991678, 0.9208715607266907], 'metric2_trials': [0.9231651387083422, 0.9243119276991678, 0.9208715607266907]}, {'group_index': 2003, 'group_probe_seed': 9223366712331899176, 'trial_seeds': [9223366712331899176, 9223366714711074969, 9223366717367542346], 'loss_trials': [0.2814661248698147, 0.2811240275655318, 0.2787590268828453], 'metric1_trials': [0.9243119276991678, 0.9208715607266907, 0.9254587166899935], 'metric2_trials': [0.9243119276991678, 0.9208715607266907, 0.9254587166899935]}, {'group_index': 2004, 'group_probe_seed': 9223366713905190553, 'trial_seeds': [9223366713905190553, 9223366712056639272, 9223366717992199675], 'loss_trials': [0.2818331492876788, 0.28047609465931533, 0.27761558873937764], 'metric1_trials': [0.9208715607266907, 0.9220183497175164, 0.9254587166899935], 'metric2_trials': [0.9208715607266907, 0.9220183497175164, 0.9254587166899935]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.27872907443695477, 'metric1_mean': 0.9236238543046723, 'metric2_mean': 0.9236238543046723, 'loss_std': 0.002345877940481228, 'metric1_std': 0.0018287085533913146, 'metric2_std': 0.0018287085533913146, 'limits': {'loss': 0.2790078035113917, 'metric1': 0.9227002304503676, 'metric2': 0.9227002304503676, 'loss_std': 0.004691755880962456, 'metric1_std': 0.0036574171067826292, 'metric2_std': 0.0036574171067826292}}, 'limits': {'loss': 0.2790078035113917, 'metric1': 0.9227002304503676, 'metric2': 0.9227002304503676, 'loss_std': 0.004691755880962456, 'metric1_std': 0.0036574171067826292, 'metric2_std': 0.0036574171067826292}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066322514933, 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119], 'loss_trials': [0.27521545945777803, 0.28079099221787324, 0.2773881013923829], 'metric1_trials': [0.9243119276991678, 0.9254587166899935, 0.9197247717358651], 'metric2_trials': [0.9243119276991678, 0.9254587166899935, 0.9197247717358651]}, {'group_index': 3001, 'group_probe_seed': 9223364067895867686, 'trial_seeds': [9223364067895867686, 9223364065584185495, 9223364072532478532], 'loss_trials': [0.27590092428780477, 0.2799894925924616, 0.276039891901913], 'metric1_trials': [0.9231651387083422, 0.9231651387083422, 0.9254587166899935], 'metric2_trials': [0.9231651387083422, 0.9231651387083422, 0.9254587166899935]}, {'group_index': 3002, 'group_probe_seed': 9223364065181531799, 'trial_seeds': [9223364065181531799, 9223364062929749798, 9223364060283768309], 'loss_trials': [0.27619840252563493, 0.2808644362135765, 0.278936781057524], 'metric1_trials': [0.9208715607266907, 0.9243119276991678, 0.9231651387083422], 'metric2_trials': [0.9208715607266907, 0.9243119276991678, 0.9231651387083422]}, {'group_index': 3003, 'group_probe_seed': 9223364058162918592, 'trial_seeds': [9223364058162918592, 9223364060275314033, 9223364062921426850], 'loss_trials': [0.2803892860992239, 0.2789066718258989, 0.28070453438190146], 'metric1_trials': [0.9231651387083422, 0.9231651387083422, 0.9243119276991678], 'metric2_trials': [0.9231651387083422, 0.9231651387083422, 0.9243119276991678]}, {'group_index': 3004, 'group_probe_seed': 9223364059736205873, 'trial_seeds': [9223364059736205873, 9223364057620878208, 9223364063562602835], 'loss_trials': [0.2792704203123346, 0.27587895696863124, 0.2776028476997253], 'metric1_trials': [0.9208715607266907, 0.9266055056808191, 0.9243119276991678], 'metric2_trials': [0.9208715607266907, 0.9266055056808191, 0.9243119276991678]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.27827181326231093, 'metric1_mean': 0.9234709491058956, 'metric2_mean': 0.9234709491058956, 'loss_std': 0.002060856037625062, 'metric1_std': 0.0018626392490883822, 'metric2_std': 0.0018626392490883822, 'limits': {'loss': 0.2785500850755732, 'metric1': 0.9225474781567896, 'metric2': 0.9225474781567896, 'loss_std': 0.004121712075250124, 'metric1_std': 0.0037252784981767644, 'metric2_std': 0.0037252784981767644}}, 'limits': {'loss': 0.2785500850755732, 'metric1': 0.9225474781567896, 'metric2': 0.9225474781567896, 'loss_std': 0.004121712075250124, 'metric1_std': 0.0037252784981767644, 'metric2_std': 0.0037252784981767644}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.2786275160444804, 'metric1_mean': 0.9235474017052842, 'metric2_mean': 0.9235474017052842, 'loss_std': 0.002061481120060271, 'metric1_std': 0.0016586684456133133, 'metric2_std': 0.0016586684456133133, 'limits': {'loss': 0.27890614356052484, 'metric1': 0.9226238543035788, 'metric2': 0.9226238543035788, 'loss_std': 0.004122962240120542, 'metric1_std': 0.0033173368912266265, 'metric2_std': 0.0033173368912266265}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.2785089484504239, 'metric1_mean': 0.9235219175054876, 'metric2_mean': 0.9235219175054876, 'loss_std': 0.0020447630394188677, 'metric1_std': 0.0017083661051949949, 'metric2_std': 0.0017083661051949949, 'limits': {'loss': 0.2787874573988743, 'metric1': 0.9225983955879822, 'metric2': 0.9225983955879822, 'loss_std': 0.004089526078837735, 'metric1_std': 0.0034167322103899897, 'metric2_std': 0.0034167322103899897}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 872, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **50000**
- 最近 50 回合 mean return: **-0.2091** (min=-3.5000, max=+1.3756)
- 最近 50 回合 mean terminal reward: **-0.2091**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+1.5627**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **417**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 1652 | 3 | +0.5625 | +1.5627 | +1.5627 | +0.0000 | 12 | 0 | 0 | 33 |
| 2 | 1847 | 3 | +0.5625 | +1.5626 | +1.5626 | +0.0000 | 12 | 0 | 0 | 31 |
| 3 | 307 | 3 | +0.5417 | +1.5423 | +1.5423 | +0.0000 | 12 | 0 | 0 | 33 |
| 4 | 2665 | 3 | +0.5208 | +1.5215 | +1.5215 | +0.0000 | 12 | 0 | 0 | 31 |
| 5 | 173 | 3 | +0.5208 | +1.5215 | +1.5215 | +0.0000 | 12 | 0 | 0 | 31 |
| 6 | 139 | 3 | +0.5208 | +1.5214 | +1.5214 | +0.0000 | 12 | 0 | 0 | 34 |
| 7 | 113 | 3 | +0.5208 | +1.5213 | +1.5213 | +0.0000 | 12 | 0 | 0 | 31 |
| 8 | 98 | 3 | +0.5208 | +1.5211 | +1.5211 | +0.0000 | 12 | 0 | 0 | 32 |
| 9 | 299 | 3 | +0.5000 | +1.5006 | +1.5006 | +0.0000 | 12 | 0 | 0 | 31 |
| 10 | 4767 | 3 | +0.5000 | +1.5005 | +1.5005 | +0.0000 | 12 | 0 | 0 | 30 |
| 11 | 2931 | 3 | +0.5000 | +1.5005 | +1.5005 | +0.0000 | 12 | 0 | 0 | 31 |
| 12 | 1177 | 3 | +0.5000 | +1.5004 | +1.5004 | +0.0000 | 12 | 0 | 0 | 33 |
| 13 | 2670 | 3 | +0.5000 | +1.5004 | +1.5004 | +0.0000 | 12 | 0 | 0 | 31 |
| 14 | 2104 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 12 | 0 | 0 | 30 |
| 15 | 1923 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 12 | 0 | 0 | 31 |
| 16 | 1271 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 12 | 0 | 0 | 29 |
| 17 | 2900 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 12 | 0 | 0 | 29 |
| 18 | 1686 | 3 | +0.5000 | +1.5003 | +1.5003 | +0.0000 | 12 | 0 | 0 | 32 |
| 19 | 4100 | 3 | +0.5000 | +1.5001 | +1.5001 | +0.0000 | 12 | 0 | 0 | 30 |
| 20 | 1574 | 3 | +0.4792 | +1.4799 | +1.4799 | +0.0000 | 12 | 0 | 0 | 30 |

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
| `L1.B1.K` | 13 | 9 | -4 |
| `L1.B2.K` | 13 | 8 | -5 |
| `L1.B3.K` | 13 | 8 | -5 |
| `L1.B4.K` | 13 | 10 | -3 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B1.K` | 13 | 9 | -4 |
| `L10.B2.K` | 13 | 8 | -5 |
| `L10.B3.K` | 13 | 8 | -5 |
| `L10.B4.K` | 13 | 10 | -3 |
| `L10.B5.K` | 13 | 9 | -4 |
| `L11.B1.K` | 13 | 11 | -2 |
| `L11.B2.K` | 13 | 10 | -3 |
| `L11.B3.K` | 13 | 10 | -3 |
| `L11.B4.K` | 13 | 12 | -1 |
| `L11.B5.K` | 13 | 11 | -2 |
| `L2.B1.K` | 13 | 9 | -4 |
| `L2.B2.K` | 13 | 8 | -5 |
| `L2.B3.K` | 13 | 8 | -5 |
| `L2.B4.K` | 13 | 10 | -3 |
| `L2.B5.K` | 13 | 9 | -4 |
| `L3.B1.K` | 13 | 11 | -2 |
| `L3.B2.K` | 13 | 10 | -3 |
| `L3.B3.K` | 13 | 10 | -3 |
| `L3.B4.K` | 13 | 12 | -1 |
| `L3.B5.K` | 13 | 11 | -2 |
| `L4.B1.K` | 13 | 7 | -6 |
| `L4.B2.K` | 13 | 6 | -7 |
| `L4.B3.K` | 13 | 6 | -7 |
| `L4.B4.K` | 13 | 8 | -5 |
| `L4.B5.K` | 13 | 7 | -6 |
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B2.K` | 13 | 8 | -5 |
| `L5.B3.K` | 13 | 8 | -5 |
| `L5.B4.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 9 | -4 |
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
| `L8.B1.K` | 13 | 9 | -4 |
| `L8.B2.K` | 13 | 8 | -5 |
| `L8.B3.K` | 13 | 8 | -5 |
| `L8.B4.K` | 13 | 10 | -3 |
| `L8.B5.K` | 13 | 9 | -4 |
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
| `L6.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L7.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L8.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
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
| 408 | 48960 | -0.0013 | +0.3702 | +0.3409 | 0.008 | 0.00000 | 0.00176 | 1.000 | 0.00000 | +0.1037 | 0.00 |
| 409 | 49080 | -0.0024 | +0.3288 | +0.3481 | 0.023 | 0.00000 | 0.00328 | 1.000 | 0.00000 | +0.3431 | 0.00 |
| 410 | 49200 | -0.0039 | +0.3230 | +0.3374 | 0.014 | 0.00000 | 0.00119 | 1.000 | 0.00000 | +0.3425 | 0.00 |
| 411 | 49320 | +0.0028 | +0.3780 | +0.3286 | 0.009 | 0.00000 | 0.00165 | 1.000 | 0.00000 | +0.1568 | 0.00 |
| 412 | 49440 | -0.0008 | +0.3897 | +0.3202 | 0.007 | 0.00000 | 0.00151 | 1.000 | 0.00000 | +0.0598 | 0.00 |
| 413 | 49560 | -0.0038 | +0.3669 | +0.3159 | 0.012 | 0.00000 | -0.00009 | 1.000 | 0.00000 | +0.1693 | 0.00 |
| 414 | 49680 | -0.0001 | +0.3334 | +0.3138 | 0.008 | 0.00000 | 0.00003 | 1.000 | 0.00000 | +0.2879 | 0.00 |
| 415 | 49800 | +0.0006 | +0.3862 | +0.2975 | 0.012 | 0.00000 | 0.00049 | 1.000 | 0.00000 | +0.1169 | 0.00 |
| 416 | 49920 | -0.0046 | +0.3026 | +0.2815 | 0.015 | 0.00000 | 0.00156 | 1.000 | 0.00000 | +0.3468 | 0.00 |
| 417 | 50000 | -0.0018 | +0.4361 | +0.2694 | 0.009 | 0.00000 | 0.00177 | 1.000 | 0.00000 | +0.0840 | 0.00 |

_Entropy 趋势：+1.5980 → +0.2694（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**835** / 877
- **未收敛 slot**：**42** / 877

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
  - slot[227] entropy=0.803 (uniform≈4.159)
  - slot[250] entropy=0.803 (uniform≈4.159)
  - slot[258] entropy=0.803 (uniform≈4.159)
  - slot[275] entropy=0.803 (uniform≈4.159)
  - slot[291] entropy=0.803 (uniform≈4.159)
  - slot[154] entropy=0.793 (uniform≈4.159)
  - slot[177] entropy=0.793 (uniform≈4.159)
  - slot[185] entropy=0.793 (uniform≈4.159)

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
    --action-config Parting Chapter/persistent/rl/bert-base/sst2/s1t0.001_s2t0.001_s2st2.0__bertbase_sst2_stage2_k3_4gpu_stage1best20260625_1b34e949_20260803/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。