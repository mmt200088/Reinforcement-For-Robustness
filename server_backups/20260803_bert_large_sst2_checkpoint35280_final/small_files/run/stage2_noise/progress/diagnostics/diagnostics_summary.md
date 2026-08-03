# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=35280）

_更新时间: 2026-08-02 23:44:59_  ·  累计用时: **26h31m43s**

**Run meta**：
- `profile` = `sst2_large`
- `fixed_label` = `Stage-1 config (manual; softmax fixed deg6)`
- `fixed_source` = `manual`
- `rl_variant` = `blb_v3_layerwise_robust_shared_gtrxl_small_v1`
- `policy_network_variant` = `shared_gtrxl_small_v1`
- `policy_network` = `{'variant': 'shared_gtrxl_small_v1', 'critic_kind': 'shared_gtrxl', 'shares_actor_trunk': True, 'total': 692999, 'shared': 676032, 'actor_only': 8646, 'critic_only': 8321}`
- `decision_granularity` = `layer`
- `reward_design` = `robust_constrained`
- `algorithm_revision` = `network_weighted_hml_three_bank_convergence_v12`
- `algorithm_contract_hash` = `3f314495ed3e108af0c5a5a0b432cc95f996a3a8870402a77c8093ad68fedf72`
- `run_context_hash` = `877be4a23a8029bdd8d73f1a9deadff9dbeedd61776eded3e96d8edf1f150234`
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
- `constraint_limits` = `{'loss': 0.28338395645444586, 'metric1': 0.91132734375, 'metric2': 0.91132734375, 'loss_std': 0.01022568900004183, 'metric1_std': 0.008282022088895464, 'metric2_std': 0.008282022088895464}`
- `baseline_preflight_metrics` = `{'ok': True, 'trial_count': 15, 'metric1_mean': 0.9122395833333333, 'metric2_mean': 0.9122395833333333, 'loss_mean': 0.28310085559884707, 'metric1_std': 0.004141011044447732, 'metric2_std': 0.004141011044447732, 'loss_std': 0.005112844500020915, 'metric1_threshold': 0.91132734375, 'metric2_threshold': 0.91132734375, 'loss_threshold': 0.28338395645444586, 'metric1_std_threshold': 0.008282022088895464, 'metric2_std_threshold': 0.008282022088895464, 'loss_std_threshold': 0.01022568900004183, 'limit_tolerance': 0.001, 'stability_tolerance': 2.0, 'stability_floor': 0.0, 'threshold_source': 'robust_all_max_blb_baseline', 'robust_reference': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.28724250569939613, 0.2844076007604599, 0.28059323877096176], 'metric1_trials': [0.91015625, 0.91015625, 0.91796875], 'metric2_trials': [0.91015625, 0.91015625, 0.91796875]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.2914871387183666, 0.2913746237754822, 0.2893592268228531], 'metric1_trials': [0.91015625, 0.91015625, 0.921875], 'metric2_trials': [0.91015625, 0.91015625, 0.921875]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.27522461116313934, 0.28238487988710403, 0.2757323496043682], 'metric1_trials': [0.9140625, 0.91015625, 0.9140625], 'metric2_trials': [0.9140625, 0.91015625, 0.9140625]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.2836213782429695, 0.28524618595838547, 0.2799094319343567], 'metric1_trials': [0.9140625, 0.91015625, 0.9140625], 'metric2_trials': [0.9140625, 0.91015625, 0.9140625]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.28075500950217247, 0.28021232783794403, 0.27896232530474663], 'metric1_trials': [0.90625, 0.90625, 0.9140625], 'metric2_trials': [0.90625, 0.90625, 0.9140625]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.28310085559884707, 'metric1_mean': 0.9122395833333333, 'metric2_mean': 0.9122395833333333, 'loss_std': 0.005112844500020915, 'metric1_std': 0.004141011044447732, 'metric2_std': 0.004141011044447732, 'limits': {'loss': 0.28338395645444586, 'metric1': 0.91132734375, 'metric2': 0.91132734375, 'loss_std': 0.01022568900004183, 'metric1_std': 0.008282022088895464, 'metric2_std': 0.008282022088895464}}, 'limits': {'loss': 0.28338395645444586, 'metric1': 0.91132734375, 'metric2': 0.91132734375, 'loss_std': 0.01022568900004183, 'metric1_std': 0.008282022088895464, 'metric2_std': 0.008282022088895464}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'group_count': 5, 'groups': [{'group_index': 0, 'group_probe_seed': 9223372029836031245, 'trial_seeds': [9223372029836031245, 9223372031545904316, 9223372034200340079], 'loss_trials': [0.28724250569939613, 0.2844076007604599, 0.28059323877096176], 'metric1_trials': [0.91015625, 0.91015625, 0.91796875], 'metric2_trials': [0.91015625, 0.91015625, 0.91796875]}, {'group_index': 1, 'group_probe_seed': 9223372031409449854, 'trial_seeds': [9223372031409449854, 9223372028891468495, 9223372034824738844], 'loss_trials': [0.2914871387183666, 0.2913746237754822, 0.2893592268228531], 'metric1_trials': [0.91015625, 0.91015625, 0.921875], 'metric2_trials': [0.91015625, 0.91015625, 0.921875]}, {'group_index': 2, 'group_probe_seed': 9223372024390705327, 'trial_seeds': [9223372024390705327, 9223372026237032734, 9223372020303762381], 'loss_trials': [0.27522461116313934, 0.28238487988710403, 0.2757323496043682], 'metric1_trials': [0.9140625, 0.91015625, 0.9140625], 'metric2_trials': [0.9140625, 0.91015625, 0.9140625]}, {'group_index': 3, 'group_probe_seed': 9223372021669160472, 'trial_seeds': [9223372021669160472, 9223372023582597033, 9223372025223124346], 'loss_trials': [0.2836213782429695, 0.28524618595838547, 0.2799094319343567], 'metric1_trials': [0.9140625, 0.91015625, 0.9140625], 'metric2_trials': [0.9140625, 0.91015625, 0.9140625]}, {'group_index': 4, 'group_probe_seed': 9223372023240350793, 'trial_seeds': [9223372023240350793, 9223372020928161272, 9223372027877568299], 'loss_trials': [0.28075500950217247, 0.28021232783794403, 0.27896232530474663], 'metric1_trials': [0.90625, 0.90625, 0.9140625], 'metric2_trials': [0.90625, 0.90625, 0.9140625]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.28310085559884707, 'metric1_mean': 0.9122395833333333, 'metric2_mean': 0.9122395833333333, 'loss_std': 0.005112844500020915, 'metric1_std': 0.004141011044447732, 'metric2_std': 0.004141011044447732, 'limits': {'loss': 0.28338395645444586, 'metric1': 0.91132734375, 'metric2': 0.91132734375, 'loss_std': 0.01022568900004183, 'metric1_std': 0.008282022088895464, 'metric2_std': 0.008282022088895464}}, 'limits': {'loss': 0.28338395645444586, 'metric1': 0.91132734375, 'metric2': 0.91132734375, 'loss_std': 0.01022568900004183, 'metric1_std': 0.008282022088895464, 'metric2_std': 0.008282022088895464}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0, 'authoritative_validation_full': {'ok': True, 'schema_version': 'stage2_validation_banks_v1', 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only', 'banks': {'A': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 1000, 'group_probe_seed': 9223369374594418853, 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823], 'loss_trials': [0.22700296984900029, 0.23004410436394018, 0.2292845182189154], 'metric1_trials': [0.9346330269761042, 0.9380733939485812, 0.9346330269761042], 'metric2_trials': [0.9346330269761042, 0.9380733939485812, 0.9346330269761042]}, {'group_index': 1001, 'group_probe_seed': 9223369376165613078, 'trial_seeds': [9223369376165613078, 9223369374455707559, 9223369381474450804], 'loss_trials': [0.22841152494106817, 0.23256351942316109, 0.22944692406085654], 'metric1_trials': [0.9380733939485812, 0.9357798159669298, 0.9369266049577556], 'metric2_trials': [0.9380733939485812, 0.9357798159669298, 0.9369266049577556]}, {'group_index': 1002, 'group_probe_seed': 9223369373444064327, 'trial_seeds': [9223369373444064327, 9223369371801271798, 9223369369079723813], 'loss_trials': [0.22773832342493425, 0.2300930288406687, 0.23136635999613947], 'metric1_trials': [0.9357798159669298, 0.9311926600036271, 0.9357798159669298], 'metric2_trials': [0.9357798159669298, 0.9311926600036271, 0.9357798159669298]}, {'group_index': 1003, 'group_probe_seed': 9223369367499062704, 'trial_seeds': [9223369367499062704, 9223369369146835969, 9223369370794643154], 'loss_trials': [0.2263338578130127, 0.23170167686195547, 0.23022320921267939], 'metric1_trials': [0.9357798159669298, 0.9346330269761042, 0.9369266049577556], 'metric2_trials': [0.9357798159669298, 0.9346330269761042, 0.9369266049577556]}, {'group_index': 1004, 'group_probe_seed': 9223369369139590113, 'trial_seeds': [9223369369139590113, 9223369366492400208, 9223369373499408515], 'loss_trials': [0.230688736936368, 0.22768690676317302, 0.23096695979800794], 'metric1_trials': [0.9323394489944528, 0.9369266049577556, 0.9346330269761042], 'metric2_trials': [0.9323394489944528, 0.9369266049577556, 0.9346330269761042]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.22957017470025867, 'metric1_mean': 0.9354740055693765, 'metric2_mean': 0.9354740055693765, 'loss_std': 0.0018178136440177937, 'metric1_std': 0.0019124068528275171, 'metric2_std': 0.0019124068528275171, 'limits': {'loss': 0.2297997448749589, 'metric1': 0.9345385315638072, 'metric2': 0.9345385315638072, 'loss_std': 0.0036356272880355874, 'metric1_std': 0.0038248137056550342, 'metric2_std': 0.0038248137056550342}}, 'limits': {'loss': 0.2297997448749589, 'metric1': 0.9345385315638072, 'metric2': 0.9345385315638072, 'loss_std': 0.0036356272880355874, 'metric1_std': 0.0038248137056550342, 'metric2_std': 0.0038248137056550342}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'B': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 2000, 'group_probe_seed': 9223366720425372765, 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903], 'loss_trials': [0.22785029826907938, 0.22692619742603476, 0.22856403463477387], 'metric1_trials': [0.9357798159669298, 0.9334862379852785, 0.9346330269761042], 'metric2_trials': [0.9357798159669298, 0.9334862379852785, 0.9346330269761042]}, {'group_index': 2001, 'group_probe_seed': 9223366722063803790, 'trial_seeds': [9223366722063803790, 9223366720019946559, 9223366726961026796], 'loss_trials': [0.22848142751859962, 0.23140044619730854, 0.2285524161036955], 'metric1_trials': [0.9346330269761042, 0.9369266049577556, 0.9357798159669298], 'metric2_trials': [0.9346330269761042, 0.9369266049577556, 0.9357798159669298]}, {'group_index': 2002, 'group_probe_seed': 9223366719342124031, 'trial_seeds': [9223366719342124031, 9223366717365510734, 9223366714717300893], 'loss_trials': [0.2313183351941065, 0.22629305556279802, 0.2309905780017923], 'metric1_trials': [0.9323394489944528, 0.9392201829394069, 0.9403669719302327], 'metric2_trials': [0.9323394489944528, 0.9392201829394069, 0.9403669719302327]}, {'group_index': 2003, 'group_probe_seed': 9223366712331899176, 'trial_seeds': [9223366712331899176, 9223366714711074969, 9223366717367542346], 'loss_trials': [0.22742090955239916, 0.22654667168582251, 0.22945415126074345], 'metric1_trials': [0.9357798159669298, 0.9357798159669298, 0.9334862379852785], 'metric2_trials': [0.9357798159669298, 0.9357798159669298, 0.9334862379852785]}, {'group_index': 2004, 'group_probe_seed': 9223366713905190553, 'trial_seeds': [9223366713905190553, 9223366712056639272, 9223366717992199675], 'loss_trials': [0.22834765610344912, 0.226713913022925, 0.23134385350100492], 'metric1_trials': [0.9357798159669298, 0.9403669719302327, 0.9392201829394069], 'metric2_trials': [0.9357798159669298, 0.9403669719302327, 0.9392201829394069]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.22868026293563554, 'metric1_mean': 0.9362385315632601, 'metric2_mean': 0.9362385315632601, 'loss_std': 0.0018330122196968976, 'metric1_std': 0.0025199553466676043, 'metric2_std': 0.0025199553466676043, 'limits': {'loss': 0.22890894319857114, 'metric1': 0.9353022930316969, 'metric2': 0.9353022930316969, 'loss_std': 0.003666024439393795, 'metric1_std': 0.005039910693335209, 'metric2_std': 0.005039910693335209}}, 'limits': {'loss': 0.22890894319857114, 'metric1': 0.9353022930316969, 'metric2': 0.9353022930316969, 'loss_std': 0.003666024439393795, 'metric1_std': 0.005039910693335209, 'metric2_std': 0.005039910693335209}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}, 'C': {'ok': True, 'threshold_source': 'robust_all_max_blb_baseline', 'trial_count': 15, 'group_count': 5, 'groups': [{'group_index': 3000, 'group_probe_seed': 9223364066322514933, 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119], 'loss_trials': [0.23136619991118754, 0.230497899678869, 0.22770537176263442], 'metric1_trials': [0.9357798159669298, 0.9346330269761042, 0.9357798159669298], 'metric2_trials': [0.9357798159669298, 0.9346330269761042, 0.9357798159669298]}, {'group_index': 3001, 'group_probe_seed': 9223364067895867686, 'trial_seeds': [9223364067895867686, 9223364065584185495, 9223364072532478532], 'loss_trials': [0.22721700260945416, 0.22784654180937952, 0.23156198230358438], 'metric1_trials': [0.9346330269761042, 0.9357798159669298, 0.9380733939485812], 'metric2_trials': [0.9346330269761042, 0.9357798159669298, 0.9380733939485812]}, {'group_index': 3002, 'group_probe_seed': 9223364065181531799, 'trial_seeds': [9223364065181531799, 9223364062929749798, 9223364060283768309], 'loss_trials': [0.22886348796000175, 0.22561202041052897, 0.22934866853810232], 'metric1_trials': [0.9357798159669298, 0.9369266049577556, 0.9392201829394069], 'metric2_trials': [0.9357798159669298, 0.9369266049577556, 0.9392201829394069]}, {'group_index': 3003, 'group_probe_seed': 9223364058162918592, 'trial_seeds': [9223364058162918592, 9223364060275314033, 9223364062921426850], 'loss_trials': [0.22600078828837894, 0.2307887245482261, 0.22842852279133752], 'metric1_trials': [0.9369266049577556, 0.9369266049577556, 0.9357798159669298], 'metric2_trials': [0.9369266049577556, 0.9369266049577556, 0.9357798159669298]}, {'group_index': 3004, 'group_probe_seed': 9223364059736205873, 'trial_seeds': [9223364059736205873, 9223364057620878208, 9223364063562602835], 'loss_trials': [0.22891025149494137, 0.22610324826262412, 0.22805578200095292], 'metric1_trials': [0.9380733939485812, 0.9380733939485812, 0.9357798159669298], 'metric2_trials': [0.9380733939485812, 0.9380733939485812, 0.9357798159669298]}], 'pooled': {'trial_count': 15, 'loss_mean': 0.22855376615801354, 'metric1_mean': 0.9365443419608136, 'metric2_mean': 0.9365443419608136, 'loss_std': 0.0019092981801168024, 'metric1_std': 0.0013476368151215756, 'metric2_std': 0.0013476368151215756, 'limits': {'loss': 0.22878231992417153, 'metric1': 0.9356077976188528, 'metric2': 0.9356077976188528, 'loss_std': 0.0038185963602336047, 'metric1_std': 0.0026952736302431513, 'metric2_std': 0.0026952736302431513}}, 'limits': {'loss': 0.22878231992417153, 'metric1': 0.9356077976188528, 'metric2': 0.9356077976188528, 'loss_std': 0.0038185963602336047, 'metric1_std': 0.0026952736302431513, 'metric2_std': 0.0026952736302431513}, 'bootstrap': {'samples': 4096, 'seed': 42}, 'precision_tolerance': 0.001, 'stability_multiplier': 2.0}}, 'promotion_reference_ab': {'trial_count': 30, 'loss_mean': 0.2291252188179471, 'metric1_mean': 0.9358562685663184, 'metric2_mean': 0.9358562685663184, 'loss_std': 0.0018498918468219027, 'metric1_std': 0.002232119750557938, 'metric2_std': 0.002232119750557938, 'limits': {'loss': 0.22935434403676502, 'metric1': 0.9349204122977521, 'metric2': 0.9349204122977521, 'loss_std': 0.0036997836936438055, 'metric1_std': 0.004464239501115876, 'metric2_std': 0.004464239501115876}}, 'final_reference_abc': {'trial_count': 45, 'loss_mean': 0.22893473459796926, 'metric1_mean': 0.9360856263644834, 'metric2_mean': 0.9360856263644834, 'loss_std': 0.001868046807478947, 'metric1_std': 0.0019923067865873743, 'metric2_std': 0.0019923067865873743, 'limits': {'loss': 0.2291636693325672, 'metric1': 0.935149540738119, 'metric2': 0.935149540738119, 'loss_std': 0.003736093614957894, 'metric1_std': 0.003984613573174749, 'metric2_std': 0.003984613573174749}}, 'contract': {'schema_version': 'layerwise_validation_banks_v1', 'banks': {'A': {'probe_seeds': [9223369374594418853, 9223369376165613078, 9223369373444064327, 9223369367499062704, 9223369369139590113], 'trial_seeds': [9223369374594418853, 9223369377110143252, 9223369378681338823, 9223369376165613078, 9223369374455707559, 9223369381474450804, 9223369373444064327, 9223369371801271798, 9223369369079723813, 9223369367499062704, 9223369369146835969, 9223369370794643154, 9223369369139590113, 9223369366492400208, 9223369373499408515], 'trials_per_probe': 3, 'trial_count': 15}, 'B': {'probe_seeds': [9223366720425372765, 9223366722063803790, 9223366719342124031, 9223366712331899176, 9223366713905190553], 'trial_seeds': [9223366720425372765, 9223366722674382316, 9223366724256259903, 9223366722063803790, 9223366720019946559, 9223366726961026796, 9223366719342124031, 9223366717365510734, 9223366714717300893, 9223366712331899176, 9223366714711074969, 9223366717367542346, 9223366713905190553, 9223366712056639272, 9223366717992199675], 'trials_per_probe': 3, 'trial_count': 15}, 'C': {'probe_seeds': [9223364066322514933, 9223364067895867686, 9223364065181531799, 9223364058162918592, 9223364059736205873], 'trial_seeds': [9223364066322514933, 9223364068238621252, 9223364069878100119, 9223364067895867686, 9223364065584185495, 9223364072532478532, 9223364065181531799, 9223364062929749798, 9223364060283768309, 9223364058162918592, 9223364060275314033, 9223364062921426850, 9223364059736205873, 9223364057620878208, 9223364063562602835], 'trials_per_probe': 3, 'trial_count': 15}}, 'promotion_trial_count': 30, 'final_trial_count': 45, 'hard_gate': 'joint_six_point_plus_compute_and_communication_counterfactual_six_point_v1', 'bootstrap_probability_role': 'diagnostic_tiebreak_only'}, 'split': 'validation_full', 'example_count': 872, 'fidelity': 'F4'}}`
- `borderline_retest_enabled` = `False`
- `borderline_retest_trials_multiplier` = `1`

## 1. 训练进度（training progress）

- 已完成回合数: **35280**
- 最近 50 回合 mean return: **+0.9229** (min=-3.4700, max=+1.6463)
- 最近 50 回合 mean terminal reward: **+0.9229**
- 最近 50 回合 mean invalid 子步数: **0.00** / 95
- 训练期 best reward: **+1.6983**
- 训练期 worst reward: **-3.5000**
- PPO 更新次数: **294**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 18970 | 3 | +0.6979 | +1.6983 | +1.6983 | +0.0000 | 24 | 0 | 0 | 64 |
| 2 | 13131 | 3 | +0.6771 | +1.6778 | +1.6778 | +0.0000 | 24 | 0 | 0 | 62 |
| 3 | 35206 | 3 | +0.6667 | +1.6674 | +1.6674 | +0.0000 | 24 | 0 | 0 | 63 |
| 4 | 33732 | 3 | +0.6667 | +1.6674 | +1.6674 | +0.0000 | 24 | 0 | 0 | 64 |
| 5 | 20272 | 3 | +0.6562 | +1.6569 | +1.6569 | +0.0000 | 24 | 0 | 0 | 62 |
| 6 | 34135 | 3 | +0.6458 | +1.6465 | +1.6465 | +0.0000 | 24 | 0 | 0 | 63 |
| 7 | 34049 | 3 | +0.6458 | +1.6464 | +1.6464 | +0.0000 | 24 | 0 | 0 | 60 |
| 8 | 22833 | 3 | +0.6458 | +1.6464 | +1.6464 | +0.0000 | 24 | 0 | 0 | 61 |
| 9 | 35262 | 3 | +0.6458 | +1.6463 | +1.6463 | +0.0000 | 24 | 0 | 0 | 62 |
| 10 | 19717 | 3 | +0.6354 | +1.6361 | +1.6361 | +0.0000 | 24 | 0 | 0 | 61 |
| 11 | 23693 | 3 | +0.6354 | +1.6361 | +1.6361 | +0.0000 | 24 | 0 | 0 | 62 |
| 12 | 12226 | 3 | +0.6354 | +1.6360 | +1.6360 | +0.0000 | 24 | 0 | 0 | 62 |
| 13 | 17978 | 3 | +0.6354 | +1.6360 | +1.6360 | +0.0000 | 24 | 0 | 0 | 60 |
| 14 | 18010 | 3 | +0.6354 | +1.6360 | +1.6360 | +0.0000 | 24 | 0 | 0 | 60 |
| 15 | 11585 | 3 | +0.6354 | +1.6360 | +1.6360 | +0.0000 | 24 | 0 | 0 | 63 |
| 16 | 29832 | 3 | +0.6354 | +1.6360 | +1.6360 | +0.0000 | 24 | 0 | 0 | 61 |
| 17 | 34678 | 3 | +0.6354 | +1.6359 | +1.6359 | +0.0000 | 24 | 0 | 0 | 61 |
| 18 | 18895 | 3 | +0.6354 | +1.6358 | +1.6358 | +0.0000 | 24 | 0 | 0 | 61 |
| 19 | 24615 | 3 | +0.6250 | +1.6257 | +1.6257 | +0.0000 | 24 | 0 | 0 | 62 |
| 20 | 34126 | 3 | +0.6250 | +1.6257 | +1.6257 | +0.0000 | 24 | 0 | 0 | 61 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 453 个槽与 baseline 不同_（333 SF + 120 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 9 | -4 |
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 8 | -5 |
| `L0.B4.K` | 13 | 10 | -3 |
| `L0.B5.K` | 13 | 9 | -4 |
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
| `L12.B1.K` | 13 | 11 | -2 |
| `L12.B2.K` | 13 | 10 | -3 |
| `L12.B3.K` | 13 | 10 | -3 |
| `L12.B4.K` | 13 | 12 | -1 |
| `L12.B5.K` | 13 | 11 | -2 |
| `L13.B1.K` | 13 | 7 | -6 |
| `L13.B2.K` | 13 | 6 | -7 |
| `L13.B3.K` | 13 | 6 | -7 |
| `L13.B4.K` | 13 | 8 | -5 |
| `L13.B5.K` | 13 | 7 | -6 |
| `L14.B1.K` | 13 | 7 | -6 |
| `L14.B2.K` | 13 | 6 | -7 |
| `L14.B3.K` | 13 | 6 | -7 |
| `L14.B4.K` | 13 | 8 | -5 |
| `L14.B5.K` | 13 | 7 | -6 |
| `L15.B1.K` | 13 | 11 | -2 |
| `L15.B2.K` | 13 | 10 | -3 |
| `L15.B3.K` | 13 | 10 | -3 |
| `L15.B4.K` | 13 | 12 | -1 |
| `L15.B5.K` | 13 | 11 | -2 |
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
| `L19.B1.K` | 13 | 7 | -6 |
| `L19.B2.K` | 13 | 6 | -7 |
| `L19.B3.K` | 13 | 6 | -7 |
| `L19.B4.K` | 13 | 8 | -5 |
| `L19.B5.K` | 13 | 7 | -6 |
| `L2.B1.K` | 13 | 7 | -6 |
| `L2.B2.K` | 13 | 6 | -7 |
| `L2.B3.K` | 13 | 6 | -7 |
| `L2.B4.K` | 13 | 8 | -5 |
| `L2.B5.K` | 13 | 7 | -6 |
| `L20.B1.K` | 13 | 7 | -6 |
| `L20.B2.K` | 13 | 6 | -7 |
| `L20.B3.K` | 13 | 6 | -7 |
| `L20.B4.K` | 13 | 8 | -5 |
| `L20.B5.K` | 13 | 7 | -6 |
| `L21.B1.K` | 13 | 7 | -6 |
| `L21.B2.K` | 13 | 6 | -7 |
| `L21.B3.K` | 13 | 6 | -7 |
| `L21.B4.K` | 13 | 8 | -5 |
| `L21.B5.K` | 13 | 7 | -6 |
| `L22.B1.K` | 13 | 9 | -4 |
| `L22.B2.K` | 13 | 8 | -5 |
| `L22.B3.K` | 13 | 8 | -5 |
| `L22.B4.K` | 13 | 10 | -3 |
| `L22.B5.K` | 13 | 9 | -4 |
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
| `L6.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L7.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L8.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L12.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L15.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L16.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L17.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L18.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L19.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L20.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L21.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L22.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L23.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L0.B2.R.gamma_r` | R | 28 | 15 | -13 |
| `L0.B2.R.kt_mask1_r` | R | 28 | 15 | -13 |
| `L0.B2.R.qkt_matmul_r` | R | 28 | 15 | -13 |
| `L1.B2.R.gamma_r` | R | 28 | 15 | -13 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 285 | 34200 | -0.0044 | +0.3269 | +1.0279 | 0.048 | 0.00000 | 0.00380 | 1.000 | 0.00000 | +0.9092 | 0.00 |
| 286 | 34320 | -0.0047 | +0.1934 | +0.9942 | 0.054 | 0.00000 | 0.00395 | 1.000 | 0.00000 | +1.1447 | 0.00 |
| 287 | 34440 | -0.0037 | +0.2375 | +0.9890 | 0.043 | 0.00000 | 0.00409 | 1.000 | 0.00000 | +1.0499 | 0.00 |
| 288 | 34560 | -0.0037 | +0.3195 | +0.9603 | 0.039 | 0.00000 | 0.00421 | 1.000 | 0.00000 | +0.9086 | 0.00 |
| 289 | 34680 | -0.0033 | +0.3181 | +0.9640 | 0.033 | 0.00000 | 0.00161 | 1.000 | 0.00000 | +0.9142 | 0.00 |
| 290 | 34800 | -0.0041 | +0.1809 | +0.9626 | 0.040 | 0.00000 | 0.00548 | 1.000 | 0.00000 | +1.1590 | 0.00 |
| 291 | 34920 | -0.0037 | +0.2786 | +0.9678 | 0.042 | 0.00000 | 0.00261 | 1.000 | 0.00000 | +0.9589 | 0.00 |
| 292 | 35040 | -0.0030 | +0.2358 | +0.9570 | 0.036 | 0.00000 | 0.00183 | 1.000 | 0.00000 | +1.0509 | 0.00 |
| 293 | 35160 | -0.0025 | +0.0890 | +0.9846 | 0.042 | 0.00000 | 0.00265 | 1.000 | 0.00000 | +1.3263 | 0.00 |
| 294 | 35280 | -0.0039 | +0.2919 | +1.0066 | 0.051 | 0.00000 | 0.00247 | 1.000 | 0.00000 | +0.9604 | 0.00 |

_Entropy 趋势：+1.5966 → +1.0066（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**1522** / 1753
- **未收敛 slot**：**231** / 1753

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
  - slot[1468] entropy=1.091 (uniform≈4.159)
  - slot[1491] entropy=1.091 (uniform≈4.159)
  - slot[1499] entropy=1.091 (uniform≈4.159)
  - slot[1516] entropy=1.091 (uniform≈4.159)
  - slot[1532] entropy=1.091 (uniform≈4.159)
  - slot[008] entropy=1.059 (uniform≈4.159)
  - slot[031] entropy=1.059 (uniform≈4.159)
  - slot[039] entropy=1.059 (uniform≈4.159)

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
    --action-config Parting Chapter/persistent/rl/bert-large/sst2/s1t0.001_s2t0.001_s2st2.0__bertlarge_sst2_stage2_k3_4gpu_1b34e949_20260801/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。