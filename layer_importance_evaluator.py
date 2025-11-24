# src/train/rl_layer.py
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from transformers import BertForSequenceClassification, BertConfig
from function_handler import ReversibleLayerHandler
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from typing import Optional, Iterable, Union
import datasets
import pandas as pd

# mount -o remount,size=64G /dev/shm


# from transformers import PeftModel
# Customized Callback function
class LayerImportanceEvaluator(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=100, degree=2, device='gpu'):
        self.batch_size = 16
        self.train_dataset = train_data
        self.data_collator = data_collator
        # Get base model
        # self.model = model.get_base_model() 
        self.model = model
        self.dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.dataloader_test = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        self.dataloader = iter(self.dataloader)
        self.dataloader_test = iter(self.dataloader_test)
        self.reversible_layer_handler = ReversibleLayerHandler(self.model)
        
        self.device = device
        self.degree = degree
        # Determine the way to access layers based on the model type
        # Todo: Add more model types as needed (MOE models)
        # Base layer: Transformer layer 
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'Qwen2ForSequenceClassification': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'BertForSequenceClassification': 'model.bert.encoder.layer'
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            print(model_class_name)
            raise NotImplementedError

        self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers
        bert = getattr(self.model, "bert", self.model)
        self.num_heads = bert.config.num_attention_heads

        # Initialize importance scores for each layer(all zeros)
        self.importance_score_activation = torch.zeros(self.total_layers)
        self.importance_score_softmax = torch.zeros(self.total_layers)

        ### hyper parameters reinforecement learning step/rate
        # number of candidate networks(RL joint action) in each RL step
        self.rl_action = 24
        # updating rate of importance score
        self.rl_lr = rl_lr
        self.update_importance_steps = 20
        self.log_path = f"./importance_scores_lr{self.rl_lr}_steps{self.update_importance_steps}_degree{self.degree}_actions{self.rl_action}_model{model_class_name}.txt" # Path to save importance scores

        self.train_batch_number = 200 # Number of training batches to evaluate importance scores
        self.update_importance_interval_steps = 1 # Set to 0 for continuous evaluation

        # hyper parameters for layer importance evaluation
        # take the saved time and the declined accuracy into account 
        self.time_rate = 0.25 # benefit for reward 
        self.accurate_rate = 0.25 # harmful to reward
        
        # number of layers to be updated in each RL step
        # number of layers to be updated with degree 0,1,2,3,4 polynomial approximation
        self.gelu_approximation_layers = [0,4,4,4,0]
        self.softmax_approximation_layers = [0,0,4,4,4,0,0]
        
        self.group_number = 4 # number of groups to sample from
        self.number_of_layers_per_group = 3 # number of layers of each group


        self.non_linear_approximation_softmax = []
        self.non_linear_approximation_root_reciprocal = []
        
        # mean error in ULP:high->low
        self.non_linear_approximation_gelu_error = []
        self.non_linear_approximation_softmax_error = []
        self.non_linear_approximation_root_reciprocal_error = []

        ###
        self.active_layers_indices = []
        self.trainable_module_name = []
        # self.raw_scaling = None

    # sample layers index based on importance score
    # group_num: number of groups to sample from
    # num: number of layers of each group
    def sampling_less_important_selection(self, importance_score, num, group_num):
        if num == 0:
            return torch.tensor([], dtype=torch.long)
        prob = torch.zeros(group_num, num)
        selects = torch.tensor([], dtype=torch.long)
        for i in range(group_num):
            prob[i] = (-importance_score).sigmoid()[i*num:(i+1)*num]
            select = torch.sort(torch.multinomial(prob[i], 1))[0]
            selects = torch.cat((selects, select + i * num), dim=0)
        return selects

    # def sampling_more_important_selection(self, importance_score, num):
    #     if num == 0:
    #         return torch.tensor([], dtype=torch.long)
    #     prob = (importance_score).sigmoid()
    #     select = torch.sort(torch.multinomial(prob, num))[0]
    #     return select

    def tensor_in_list(self, tensor_list, new_tensor):
        if len(new_tensor) == 0:
            return False
        for tensor in tensor_list:
            if torch.equal(tensor, new_tensor):
                return True
        return False
    

    # RL training step
    def on_evaluate(self, args, state, control, **kwargs):
        # def on_step_begin(self, args, state, control, device, **kwargs):
        # Check if it's time to switch active layers, including at step 0
        
        val_batches = list(self.dataloader)
        val_batches_test = list(self.dataloader_test)

        # compute norm upper bound - input after embedding layer, or compute from norm parameters galma
        # norm_bound = self.compute_euclidean_norm_bound(data_or_tensor = self.train_dataset)
        
        # self.logs(f"Norm Bound: {norm_bound}")
        
        self.logs(f"Model Structure: {self.model}")
        weights = self.get_weights() 
        # self.logs(f"Weights: {weights}")
        per_layer_qkv_spectral_norm = []
        for i in range(self.total_layers):
            self.logs(f"print bias: weights: {weights[0]['query']['b'].shape}")

            Wqk_spectral_norm = []
            # Wk_spectral_norm = []
            Wv_spectral_norm = []
            for head in range(self.num_heads):
                Wq_h = weights[i]["query"]["W_heads"][head]
                Wk_h = weights[i]["key"]["W_heads"][head]
                Wv_h = weights[i]["value"]["W_heads"][head]
                Wqk_h = Wq_h @ Wk_h.T  # [Dh, Dh]
                Wqk_h_spectral_norm = self.compute_spectral_norm(Wqk_h, n_iters=10, eps=1e-12, track_gradients=False)
                # Wk_h_spectral_norm = self.compute_spectral_norm(Wk_h, n_iters=10, eps=1e-12, track_gradients=False)
                Wv_h_spectral_norm = self.compute_spectral_norm(Wv_h, n_iters=10, eps=1e-12, track_gradients=False)
                Wqk_spectral_norm.append(Wqk_h_spectral_norm)
                # Wk_spectral_norm.append(Wk_h_spectral_norm)
                Wv_spectral_norm.append(Wv_h_spectral_norm)
            
            per_layer_qkv_spectral_norm.append({
                "Wqk_spectral_norm": Wqk_spectral_norm,
                "Wv_spectral_norm": Wv_spectral_norm
            })
        
        self.logs(f"Per Layer QKV Spectral Norm: {per_layer_qkv_spectral_norm}")

        # 获取层数 & 头数
        num_layers = len(per_layer_qkv_spectral_norm)
        num_heads = len(per_layer_qkv_spectral_norm[0]["Wqk_spectral_norm"])

        # 构造三个表
        Wqk_table = []
        Wv_table = []

        for layer_idx, layer_data in enumerate(per_layer_qkv_spectral_norm):
            row_qk = [layer_idx] + layer_data["Wqk_spectral_norm"]
            row_v = [layer_idx] + layer_data["Wv_spectral_norm"]
            Wqk_table.append(row_qk)
            Wv_table.append(row_v)

        # 列名：layer + head0, head1, ...
        columns = ["layer"] + [f"head{h}" for h in range(num_heads)]

        df_Wqk = pd.DataFrame(Wqk_table, columns=columns)
        df_Wv = pd.DataFrame(Wv_table, columns=columns)

        # 保存为 Excel，三个表在同一个文件的不同 sheet
        with pd.ExcelWriter("qk_v_spectral_norm_tables.xlsx") as writer:
            df_Wqk.to_excel(writer, sheet_name="Wqk_spectral_norm", index=False)
            df_Wv.to_excel(writer, sheet_name="Wv_spectral_norm", index=False)

        print("三个表已保存到 qk-v_spectral_norm_tables.xlsx")


        # self.logs("Evaluating random combination 4-4-4:")
        # self.get_final_metrics(val_batches_test,[0,3,6,9],[1,4,7,10],[2,5,8,11],[2,5,8,11],[0,3,6,9],[1,4,7,10])
        
        # # accuracy test
        # # self.logs("Evaluating origin:")
        

        
        self.logs("Evaluating origin:")
        self.get_final_metrics(val_batches_test,[],[],[],[],[],[])
        
        self.logs("Evaluating best combination 4-4-4-lr15-group:")
        self.get_final_metrics(val_batches_test,[2,3,6,9],[0,5,7,10],[1,4,8,11],[0,5,6,9],[1,4,7,10],[2,3,8,11])
        
        self.logs("Evaluating worst combination 4-4-4-lr15-group:")
        self.get_final_metrics(val_batches_test,[1,4,8,11],[0,5,7,10],[2,3,6,9],[2,3,8,11],[1,4,7,10],[0,5,6,9])
        
        self.logs("Evaluating all 1:")
        self.get_final_metrics(val_batches_test,[0,1,2,3,4,5,6,7,8,9,10,11],[],[],[0,1,2,3,4,5,6,7,8,9,10,11],[],[])
        
        self.logs("Evaluating all 3:")
        self.get_final_metrics(val_batches_test,[],[],[0,1,2,3,4,5,6,7,8,9,10,11],[],[],[0,1,2,3,4,5,6,7,8,9,10,11])
        
        
        self.logs("Evaluating random1 combination 4-4-4-lr15-group:")
        self.get_final_metrics(val_batches_test,[1,4,6,10],[0,5,7,9],[2,3,8,11],[0,5,8,9],[2,4,7,10],[1,3,6,11])
        
        self.logs("Evaluating random2 combination 4-4-4-lr15-group:")
        self.get_final_metrics(val_batches_test,[2,3,8,11],[0,5,7,9],[1,4,6,10],[1,3,6,11],[2,4,7,10],[0,5,8,9])
        
        self.logs("Evaluating origin:")
        self.get_final_metrics(val_batches_test,[1,2,4,6],[7,3,11,10],[5,0,8,9],[4,10,11,7],[1,2,5,8],[3,6,9,0])
        

        # self.logs("Evaluating middle:")
        # self.get_final_metrics(val_batches_test, [1,0,8],[10,2,7,9,3,4],[1,0,8],[10,2,7,9,3,4])

        # self.logs("Evaluating worst:")
        # self.get_final_metrics(val_batches_test, [3,4,9],[10,2,8,0,1,7],[3,4,9],[10,2,8,0,1,7])
        # self.logs("random:")
        # self.get_final_metrics(val_batches_test, [7,1,10],[5,9,2,8,0,11],[7,1,10],[5,9,2,8,0,11])
        
        # self.logs("Evaluating worst 2:")
        # self.get_final_loss(val_batches_test, [], [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        # self.get_final_metrics(val_batches_test, [], [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        # self.logs("Evaluating middle:")
        # self.get_final_loss(val_batches_test, [9,6,8], [7,2,4])
        # self.get_final_metrics(val_batches_test, [9,6,8], [7,2,4])

        # self.logs("Evaluating best:")
        # self.get_final_loss(val_batches_test, [0,11,10], [9,6,8])
        # self.get_final_metrics(val_batches_test, [0,11,10], [9,6,8])



        for _ in range(self.update_importance_steps):
            self.logs(f"In the {_}'th rl step, updating importance scores...")
            self.logs(f"Current Activation Importance Scores: {self.importance_score_activation.tolist()}")
            self.logs(f"Current Softmax Importance Scores: {self.importance_score_softmax.tolist()}")

            selects_activation = {
                "selects1": [],
                "selects2": [],
                "selects3": [],
                "selects4": []
            }
            selects_softmax = {
                "selects1": [],
                "selects2": [],
                "selects3": [],
                "selects4": [],
                "selects5": [],
                "selects6": []
            }
            # Record the losses for each action
            rets = []
            # Calculate rewards based on the losses
            rewards = []

            
            # RL action: select layers for polynomial approximation
            for k in range(self.rl_action):
                
                total_layers_index = torch.arange(self.total_layers, dtype=torch.long)
                '''activation selection'''
                # todo: classify the approximation degree
                # now: degree 1 polynomial approximation
                select1_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.number_of_layers_per_group, group_num=self.group_number)
                select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation,  num=self.number_of_layers_per_group, group_num=self.group_number)

                while self.tensor_in_list(selects_activation["selects1"], select1_activation):
                    select1_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.number_of_layers_per_group, group_num=self.group_number)
                
                while len(set(select1_activation.tolist()) & set(select2_activation.tolist())) > 0:
                    select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.number_of_layers_per_group, group_num=self.group_number)
                    while self.tensor_in_list(selects_activation["selects2"], select2_activation):
                        select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.number_of_layers_per_group, group_num=self.group_number)
                
                exclude_activation = torch.cat((select1_activation, select2_activation), dim=0)  
                mask_activation = torch.isin(total_layers_index, exclude_activation, invert=True)
                select3_activation = total_layers_index[mask_activation]

                selects_activation["selects1"].append(select1_activation)
                selects_activation["selects2"].append(select2_activation)
                selects_activation["selects3"].append(select3_activation)
                
                ''' softmax selection '''
                # todo: classify the approximation degree
                select1_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.number_of_layers_per_group, group_num=self.group_number)
                select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.number_of_layers_per_group, group_num=self.group_number)
                
                while self.tensor_in_list(selects_softmax["selects2"], select1_softmax):
                    select1_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.number_of_layers_per_group, group_num=self.group_number)
                
                while len(set(select1_softmax.tolist()) & set(select2_softmax.tolist())) > 0:
                    select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.number_of_layers_per_group, group_num=self.group_number)
                    while self.tensor_in_list(selects_softmax["selects3"], select2_softmax):
                        select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.number_of_layers_per_group, group_num=self.group_number)
                
                
                exclude_softmax = torch.cat((select1_softmax, select2_softmax), dim=0)  
                mask_softmax = torch.isin(total_layers_index, exclude_softmax, invert=True)
                select3_softmax = total_layers_index[mask_softmax]

                selects_softmax["selects2"].append(select1_softmax)
                selects_softmax["selects3"].append(select2_softmax)
                selects_softmax["selects4"].append(select3_softmax)

                # self.switch_active_adapter(select)

                # Evaluate the model with the selected layers
                # Replace the GELU function in the selected layers with degree 1 polynomial approximation
                # to do: divide into 4 groups according to importance

                self.reversible_layer_handler.replace_layer_gelu(layer_indices=select1_activation, layer_name=self.layers_attribute, degree=1)
                self.reversible_layer_handler.replace_layer_gelu(layer_indices=select2_activation, layer_name=self.layers_attribute, degree=2)
                self.reversible_layer_handler.replace_layer_gelu(layer_indices=select3_activation, layer_name=self.layers_attribute, degree=3)

                # same replace layer, need better policy
                self.reversible_layer_handler.replace_layer_softmax(layer_indices=select1_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=2, lower_bound=-4)
                self.reversible_layer_handler.replace_layer_softmax(layer_indices=select2_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=3, lower_bound=-10)
                self.reversible_layer_handler.replace_layer_softmax(layer_indices=select3_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=4, lower_bound=-13)

                self.reversible_layer_handler.model.eval()
                
                # do actior to get rewards
                # inference mode is used to avoid gradient calculation
                total_loss = 0
                batch_indices = np.random.choice(len(val_batches), size=self.train_batch_number, replace=True)
                random_batches = [val_batches[i] for i in batch_indices]

                batch_number = 0
                for val_batch in random_batches:
                    batch_number += 1
                    print(f"Evaluating with batch: {batch_number}")
                    with torch.inference_mode():
                        self.reversible_layer_handler.model.to("cuda")
                        device = next(self.reversible_layer_handler.model.parameters()).device
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        outputs = self.reversible_layer_handler.model(**val_batch)
                        total_loss += outputs.loss.item()
                    
                
                total_loss = total_loss / self.train_batch_number  # Average loss over the dataset
                # self.model.train()
                self.logs(f"RL Action {k}, Selected1 Activation Layers: {select1_activation}, Selected2 Activation Layers: {select2_activation}, Selected3 Activation Layers: {select3_activation},"
                          f"Selected1 Softmax Layers: {select1_softmax}, Selected2 Softmax Layers: {select2_softmax}, Selected3 Softmax Layers: {select3_softmax}, Loss: {total_loss}")
                rets.append(total_loss)

                # todo recover the original gelu function
                self.reversible_layer_handler.restore_all() 

            self.logs(f"All action Losses: {rets}")

            for i in range(self.rl_action):
                rewards.append(math.exp(-rets[i]))

            _mean = np.mean(rewards)

            # rewards = np.array([(r - _mean) for r in rewards]).tolist() + \
            #     self.get_time_reward(rewards)


            # without time reward term
            rewards = np.array([(r - _mean) for r in rewards]).tolist()
            self.logs(f"Rewards: {rewards}")
            prob_activation = self.importance_score_activation.sigmoid()
            prob_softmax = self.importance_score_softmax.sigmoid()
            self.logs(f"Probs_Activation:{prob_activation}, Probs_Softmax:{prob_softmax}")
            for k in range(self.rl_action):
                for i in range(self.total_layers):
                    # activation importance score update
                    if i not in selects_activation["selects1"][k]:
                        self.importance_score_activation[i] += rewards[k] * prob_activation[i] * (1 - prob_activation[i]) * self.rl_lr * 0.5
                        if i not in selects_activation["selects2"][k]:
                            self.importance_score_activation[i] += rewards[k] * prob_activation[i] * (1 - prob_activation[i]) * self.rl_lr * 0.5
                    # softmax_importance score update
                    if i not in selects_softmax["selects2"][k]:
                        self.importance_score_softmax[i] += rewards[k] * prob_softmax[i] * (1 - prob_softmax[i]) * self.rl_lr * 0.5
                        if i not in selects_softmax["selects3"][k]:
                            self.importance_score_softmax[i] += rewards[k] * prob_softmax[i] * (1 - prob_softmax[i]) * self.rl_lr * 0.5
                    # else:
                    #     self.importance_score[i] -= rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr

        self.logs(f"Final Activation Importance Scores: {self.importance_score_activation.tolist()}")
        self.logs(f"Final Softmax Importance Scores: {self.importance_score_softmax.tolist()}")
        # less_importance_layers_1_activation = np.argsort(np.array(self.importance_score_activation))[:self.gelu_approximation_layers[1]]
        # less_importance_layers_2_activation = np.argsort(np.array(self.importance_score_activation))[self.gelu_approximation_layers[1]:self.gelu_approximation_layers[2]+self.gelu_approximation_layers[1]]
        # less_importance_layers_3_activation = np.argsort(np.array(self.importance_score_activation))[self.gelu_approximation_layers[2]+self.gelu_approximation_layers[1]:]

        # less_importance_layers_1_softmax = np.argsort(np.array(self.importance_score_softmax))[:self.softmax_approximation_layers[2]]
        # less_importance_layers_2_softmax = np.argsort(np.array(self.importance_score_softmax))[self.softmax_approximation_layers[2]:self.softmax_approximation_layers[3]+self.gelu_approximation_layers[2]]
        # less_importance_layers_3_softmax = np.argsort(np.array(self.importance_score_softmax))[self.softmax_approximation_layers[3]+self.gelu_approximation_layers[2]:]
        # check the final effect
        # self.get_final_metrics(val_batches_test, less_importance_layers_1_activation, less_importance_layers_2_activation, less_importance_layers_3_activation, less_importance_layers_1_softmax,less_importance_layers_2_softmax,less_importance_layers_3_softmax)


    # calculate the reward based on the time saved (he)
    # or the communication cost saved (mpc)


    def get_final_metrics(self, val_batches=[], selected_layers1_gelu=[], selected_layers2_gelu=[], selected_layers3_gelu=[], selected_layers1_softmax=[], selected_layers2_softmax=[], selected_layers3_softmax=[]):
        total_loss_final = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        if len(selected_layers1_gelu) == 0 and len(selected_layers2_gelu) == 0 and len(selected_layers3_gelu) and len(selected_layers1_softmax) == 0 and len(selected_layers2_softmax) == 0 and len(selected_layers3_softmax) == 0:
            self.logs("No layers selected for evaluation.")
        else:
            self.reversible_layer_handler.replace_layer_gelu(layer_indices=selected_layers1_gelu, layer_name=self.layers_attribute, degree=1)
            self.reversible_layer_handler.replace_layer_gelu(layer_indices=selected_layers2_gelu, layer_name=self.layers_attribute, degree=2)
            self.reversible_layer_handler.replace_layer_gelu(layer_indices=selected_layers3_gelu, layer_name=self.layers_attribute, degree=3)

            ## same replace for the softmax
            self.reversible_layer_handler.replace_layer_softmax(layer_indices=selected_layers1_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=2, lower_bound=-4)
            self.reversible_layer_handler.replace_layer_softmax(layer_indices=selected_layers2_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=3, lower_bound=-10)
            self.reversible_layer_handler.replace_layer_softmax(layer_indices=selected_layers3_softmax, layer_name=self.layers_attribute, attention_name="attention.self", degree=4, lower_bound=-13)

        self.reversible_layer_handler.model.eval()
        for val_batch in val_batches:
            with torch.inference_mode():
                self.reversible_layer_handler.model.to("cuda")
                device = next(self.reversible_layer_handler.model.parameters()).device
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                outputs = self.reversible_layer_handler.model(**val_batch)
                #pearson and spearman correlation
                
                logits = outputs.logits.squeeze().detach().cpu().numpy()  # shape: (batch_size,)
                labels = val_batch["labels"].detach().cpu().numpy()
                total_loss_final += outputs.loss.item()

                all_predictions.extend(logits)
                all_labels.extend(labels)
                
                
                # acc and f1 score
                # logits = outputs.logits
                # predictions = torch.argmax(logits, dim=-1)
                # labels = val_batch["labels"]
                # total_loss_final += outputs.loss.item()

                # # 累计统计量
                # total_correct += (predictions == labels).sum().item()
                # total_samples += labels.size(0)
                # all_predictions.extend(predictions.detach().cpu().numpy())
                # all_labels.extend(labels.detach().cpu().numpy())

        # 计算指标
        # accuracy = total_correct / total_samples
        # f1 = f1_score(all_labels, all_predictions, average='macro')
        total_loss_final = total_loss_final / len(val_batches)  # Average loss over the dataset
        
        pearson_corr = pearsonr(all_predictions, all_labels)[0]
        spearman_corr = spearmanr(all_predictions, all_labels)[0]

        # self.reversible_layer_handler.restore_all()
        # self.logs(f"Final Gelu Layer1:{selected_layers1_gelu}; Final Gelu Layer2: {selected_layers2_gelu}; Final Gelu Layer3: {selected_layers3_gelu}; Final Softmax Layer1: {selected_layers1_softmax}; Final Softmax Layer2: {selected_layers2_softmax}; Final Softmax Layer3: {selected_layers3_softmax};")
        # self.logs(f"Final Metrics - Accuracy: {accuracy}, F1 Score: {f1}, Total Loss: {total_loss_final}")
        # return accuracy, f1, total_loss_final

        self.reversible_layer_handler.restore_all()
        self.logs(f"Final Gelu Layer1:{selected_layers1_gelu}; Final Gelu Layer2: {selected_layers2_gelu}; Final Gelu Layer3: {selected_layers3_gelu}; Final Softmax Layer1: {selected_layers1_softmax}; Final Softmax Layer2: {selected_layers2_softmax}; Final Softmax Layer3: {selected_layers3_softmax};")
        self.logs(f"Final Metrics - pearson: {pearson_corr}, spearman: {spearman_corr}, Total Loss: {total_loss_final}")
        return pearson_corr, spearman_corr, total_loss_final
        
    # def get_time_reward(self, degree):
    #     """
    #     Calculate the time reward based on the time saved.
    #     The higher the time saved, the higher the reward.
    #     """
    #     time_reward = []
    #     for i in range(self.rl_action):
    #         # to be adjusted: confirm the relation bewtween the time saved and 
    #         time_reward.append(-1 * self.time_rate * degree[i])
    #     return time_reward
    
    
    def logs(self, result):
        """
        Write the importance scores to a file.
        """
        with open(self.log_path, 'a') as f:
            f.write(f"{result}\n")

    def get_weights(self, num_heads = 12, head_dim = 64, hidden_size = 768) -> list:
        # 用于存放结果： per layer -> {q,k,v}->{W_heads, b_heads}
        per_layer_qkv = []
        with torch.no_grad():
            for i in range(self.total_layers):
                sa = self.model.bert.encoder.layer[i].attention.self  # BertSelfAttention

                layer_dict = {}
                for name in ["query", "key", "value"]:
                    linear = getattr(sa, name)                   # nn.Linear
                    W = linear.weight.data.clone()               # [hidden_size, hidden_size]
                    b = linear.bias.data.clone() if linear.bias is not None else None  # [hidden_size]

                    # 关键：按“输出维度”分头。权重形状 [out_features, in_features]
                    # 先把 out_features=hidden_size 切成 [num_heads, head_dim]
                    W_heads = W.view(num_heads, head_dim, hidden_size)   # [H, Dh, hidden_size]
                    b_heads = b.view(num_heads, head_dim) if b is not None else None  # [H, Dh]

                    # 可选：如果你更习惯做 x @ W^T 的每头权重，可保留转置版本
                    # per-head 矩阵乘法通常是: y_h = x @ W_h^T + b_h
                    W_heads_T = W_heads.transpose(1, 2).contiguous()     # [H, hidden_size, Dh]

                    layer_dict[name] = {
                        "W": W,                       # [hidden_size, hidden_size]
                        "b": b,                       # [hidden_size]
                        "W_heads": W_heads,           # [H, Dh, hidden_size]
                        "b_heads": b_heads,           # [H, Dh]
                        "W_heads_T": W_heads_T,       # [H, hidden_size, Dh] 方便做 x @ W_h^T
                    }

                per_layer_qkv.append(layer_dict)
        return per_layer_qkv
        # —— 使用示例 —— #
        # 第0层、第0个头的 Query 权重（按头切好）形状：
        # print(per_layer_qkv[0]["query"]["W_heads"].shape)   # torch.Size([num_heads, head_dim, hidden_size])
        # print(per_layer_qkv[0]["query"]["b_heads"].shape)   # torch.Size([num_heads, head_dim])

        # # 若想看某一头（比如 head=0）的 Wq：
        # head = 0
        # Wq_h = per_layer_qkv[0]["query"]["W_heads"][head]    # [Dh, hidden_size]
        # bq_h = per_layer_qkv[0]["query"]["b_heads"][head]    # [Dh]
        # print(Wq_h.shape, bq_h.shape)
    
    def compute_importance_matrix_lipschitz(self, layer_indices, layer_name, data_euclidean_bound=1, sequence_length=128, has_mask = False, has_data_info=False):
        """
        Compute the Lipschitz constant for the specified layer.
        Formula:
        q = ||Wq||_2, k = ||Wk||_2, v = ||Wv||_2
        d: hidden size, h: number of heads, L: sequence length, C: data euclidean norm bound
        Q = XWq + 1*bq, K = XWk + 1*bk, V = XWv + 1*bv
            has_mask=false(bert): 
                has_data_info=false: 1/(sqrt(d/h)) * (q * ||K||_2 + k * ||Q||_2)
                has_data_info=true: 1/(sqrt(d/h)) * (q * sqrt(k^2 * C^2 + L * ||bk||_2^2) + k * sqrt(q^2 * C^2 + L * ||bq||_2^2))
            has_mask=true(gpt):
                has_data_info=false:
                has_data_info=true: 
        """
        pass

        
    def compute_self_attention_lipschitz(self, layer_indices, layer_name, data_euclidean_bound=1, sequence_length=128, has_mask = False, has_data_info=False, attention_name=None):
        """
        Compute the Lipschitz constant for the specified layer.
        Formula from: How smooth is Attention? 
        paper url: https://arxiv.org/pdf/2312.14820
        """
        pass
    
    
    def compute_spectral_norm(self,
                            W: torch.Tensor,
                            n_iters: int = 10,
                            eps: float = 1e-12,
                            u_init: torch.Tensor | None = None,
                            v_init: torch.Tensor | None = None,
                            track_gradients: bool = False,
                            return_uv: bool = False):
        """
        计算矩阵 W 的谱范数 (最大奇异值) —— power iteration 版。

        Args:
            W: 形状 (m, n) 的 2D 矩阵 (torch.Tensor)。
            n_iters: 幂迭代次数，通常 5~10 就足够。
            eps: 防止除零的稳定项。
            u_init, v_init: 可选的初始向量 (分别为形状 (m,), (n,))。若为 None 随机初始化。
            track_gradients: 是否让迭代过程参与 autograd。多数场景下 False 更稳定；
                            若希望对 W 的梯度更精确，可设为 True。
            return_uv: 是否返回最终的左右奇异向量近似 (u, v)。

        Returns:
            sigma: 近似的最大奇异值 (谱范数)，标量张量。
            (可选) u, v: 近似的主奇异向量（左、右）。
        """
        assert W.dim() == 2, "W 必须是 2D 矩阵张量。"
        m, n = W.shape
        device, dtype = W.device, W.dtype

        # 初始化 u, v
        if u_init is None:
            u = torch.randn(m, device=device, dtype=dtype)
        else:
            u = u_init.to(device=device, dtype=dtype)

        if v_init is None:
            v = torch.randn(n, device=device, dtype=dtype)
        else:
            v = v_init.to(device=device, dtype=dtype)

        def _normalize(x):
            return x / (x.norm(p=2) + eps)

        # 迭代
        if not track_gradients:
            with torch.no_grad():
                u = _normalize(u)
                v = _normalize(v)
                for _ in range(n_iters):
                    # 右乘得到 v（近似主右奇异向量）
                    v = _normalize(W.t().matmul(u))
                    # 左乘得到 u（近似主左奇异向量）
                    u = _normalize(W.matmul(v))
            # 迭代结束后，再计算一次 sigma（这一步要参与梯度可求导）
            # 注意：这里用上一步的 u,v（视为常量方向）来度量 W 的放大
            sigma = u @ (W @ v)
        else:
            # 完全可微的版本（更耗算、更不稳定一点）
            u = _normalize(u)
            v = _normalize(v)
            for _ in range(n_iters):
                v = _normalize(W.t() @ u)
                u = _normalize(W @ v)
            sigma = u @ (W @ v)

        # 谱范数应为非负；数值波动时取绝对值更稳
        sigma = sigma.abs()

        if return_uv:
            return sigma, u, v
        return sigma.item()
    
    
    
    
    def _tensor_max_l2_norm(self, x: torch.Tensor) -> float:
        """
        计算张量 x 在最后一维上的向量 L2 范数，并返回全局最大值。
        适配形状: (..., d)
        例如:
        - (B, D) → 每个样本向量的范数; 取 max
        - (B, T, D) → 每个 token 向量范数; 取 max
        - (D,) → 单个向量范数
        """
        if x.numel() == 0:
            return 0.0
        # 在最后一维求向量范数
        norms = torch.linalg.vector_norm(x, ord=2, dim=-1)
        # 再在剩余维上取最大
        return norms.max().item()


    def compute_euclidean_norm_bound(
        self,
        data_or_tensor: Union[torch.Tensor, datasets.dataset_dict.Dataset, Iterable],
        input_key: Optional[str] = None,
        to_float: bool = True,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        计算“每层输入向量”的欧几里得范数（L2，沿最后一维）最大值。
        - 若传入的是 torch.Tensor（如中间层激活）：按最后一维求向量范数并取全局最大。
        - 若传入的是 HuggingFace Dataset / 可迭代样本：会从样本中取 `input_key` 字段（或自动猜测）转成 tensor 计算，并不断更新最大值。

        Args:
            data_or_tensor: torch.Tensor，或 HuggingFace Dataset，或可迭代的样本/张量。
            input_key: 当输入为样本字典时，从该 key 取向量/张量；若为 None 将自动尝试常见 key。
            to_float: 是否把数据转为 float 计算范数（推荐 True）。
            device: 若给定，会把张量移到该设备后再计算（如 torch.device("cuda")）。
            batch_size: 对支持 `.with_format("torch")` 的 Dataset，可指定批大小进行批处理（更快）。

        Returns:
            float: 数据中所有向量（沿最后一维）的 L2 范数的最大值。
        """
        # 1) 直接是 Tensor
        if isinstance(data_or_tensor, torch.Tensor):
            x = data_or_tensor
            if to_float:
                x = x.float()
            if device is not None:
                x = x.to(device)
            return self._tensor_max_l2_norm(x)

        # 2) HuggingFace Dataset
        if isinstance(data_or_tensor, datasets.dataset_dict.Dataset):
            ds = data_or_tensor

            # 自动猜测 input_key
            if input_key is None:
                for k in ("hidden_states", "layer_input", "inputs", "input", "input_ids", "x"):
                    if k in ds.features:
                        input_key = k
                        break
                if input_key is None:
                    raise ValueError("请指定 input_key（样本字典中对应输入向量的字段名）。")

            max_norm = 0.0

            # 优先用 torch 格式 + 批处理
            if hasattr(ds, "with_format"):
                ds_t = ds.with_format("torch")
                if batch_size is None:
                    # 逐条遍历（通用但慢）
                    for ex in ds_t:
                        vec = ex[input_key]
                        if not isinstance(vec, torch.Tensor):
                            vec = torch.tensor(vec)
                        if to_float:
                            vec = vec.float()
                        if device is not None:
                            vec = vec.to(device)
                        max_norm = max(max_norm, self._tensor_max_l2_norm(vec))
                else:
                    # 批处理：利用 select+batched=True
                    for i in range(0, len(ds_t), batch_size):
                        batch = ds_t.select(range(i, min(i + batch_size, len(ds_t)))).to_dict()
                        vec = batch[input_key]  # 可能是 list[Tensor] 或 Tensor
                        if isinstance(vec, list):
                            vec = torch.nn.utils.rnn.pad_sequence(
                                [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in vec],
                                batch_first=True
                            )
                        if not isinstance(vec, torch.Tensor):
                            vec = torch.tensor(vec)
                        if to_float:
                            vec = vec.float()
                        if device is not None:
                            vec = vec.to(device)
                        max_norm = max(max_norm, self._tensor_max_l2_norm(vec))
            else:
                # 退化路径：把 Dataset 当作普通可迭代
                for ex in ds:
                    vec = ex[input_key]
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.tensor(vec)
                    if to_float:
                        vec = vec.float()
                    if device is not None:
                        vec = vec.to(device)
                    max_norm = max(max_norm, self._tensor_max_l2_norm(vec))
            return max_norm

        # 3) 其它可迭代（list/tuple/生成器等）：元素可为 Tensor 或 dict
        if isinstance(data_or_tensor, Iterable):
            max_norm = 0.0
            for ex in data_or_tensor:
                if isinstance(ex, torch.Tensor):
                    vec = ex
                elif isinstance(ex, dict):
                    key = input_key
                    if key is None:
                        for k in ("hidden_states", "layer_input", "inputs", "input", "input_ids", "x"):
                            if k in ex:
                                key = k
                                break
                        if key is None:
                            raise ValueError("元素为字典但未找到输入字段；请显式设置 input_key。")
                    vec = ex[key]
                else:
                    # 假定是 array-like
                    vec = torch.tensor(ex)

                if not isinstance(vec, torch.Tensor):
                    vec = torch.tensor(vec)
                if to_float:
                    vec = vec.float()
                if device is not None:
                    vec = vec.to(device)

                max_norm = max(max_norm, self._tensor_max_l2_norm(vec))
            return max_norm

        raise TypeError("不支持的输入类型：请传入 torch.Tensor、HuggingFace Dataset 或可迭代对象。")