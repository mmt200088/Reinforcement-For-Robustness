# src/train/rl_layer.py
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from transformers import BertForSequenceClassification
from function_handler import ReversibleLayerHandler
from sklearn.metrics import f1_score

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

        # eval('self.' + self.layers_attribute)  Dynamically execute to access the self.layers_attribute's actual value
        self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers

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
    def sampling_less_important_selection(self, importance_score, num):
        if num == 0:
            return torch.tensor([], dtype=torch.long)
        prob = (-importance_score).sigmoid()
        select = torch.sort(torch.multinomial(prob, num))[0]
        return select

    def sampling_more_important_selection(self, importance_score, num):
        if num == 0:
            return torch.tensor([], dtype=torch.long)
        prob = (importance_score).sigmoid()
        select = torch.sort(torch.multinomial(prob, num))[0]
        return select

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

        # self.logs("Evaluating origin:")
        # self.get_final_metrics(val_batches_test,[],[],[],[],[],[])
        

        
        # self.logs("Evaluating worst combination 4-4-4-lr15-threshold:")
        # self.get_final_metrics(val_batches_test,[3,2,1,10],[5,4],[9,8,11,6,7,0],[11,8,4,2],[5,3],[7,10,0,1,6,9])
        
        # self.logs("Evaluating best combination 4-4-4-lr15:")
        # self.get_final_metrics(val_batches_test,[4,6,9,10],[7,11,1,2],[5,8,3,0],[9,8,7,4],[0,3,6,10],[11,5,2,1])
        
        # self.logs("Evaluating worst combination 4-4-4-lr15:")
        # self.get_final_metrics(val_batches_test,[5,8,3,0],[7,11,1,2],[4,6,9,10],[11,5,2,1],[0,3,6,10],[9,8,7,4])
        
        
        # self.logs("Evaluating best combination 4-4-4-lr25-threshold:")
        # self.get_final_metrics(val_batches_test,[10,4,0,6],[7,2],[11,3,9,1,5,8],[7,4,8,5],[6,9],[3,2,11,1,0,10])
        
        # self.logs("Evaluating best combination 4-4-4-lr15-threshold:")
        # self.get_final_metrics(val_batches_test,[4,6,9,10],[7,11],[1,2,5,8,3,0],[9,8,7,4],[0,3],[6,10,11,5,2,1])
        
        # self.logs("Evaluating worst combination 4-4-4-lr25-threshold:")
        # self.get_final_metrics(val_batches_test,[9,1,5,8],[11,3],[7,2,10,4,0,6],[11,1,0,10],[3,2],[6,9,7,4,8,5])
        
        # self.logs("Evaluating worst combination 4-4-4-lr15:-threshold:")
        # self.get_final_metrics(val_batches_test,[5,8,3,0],[1,2],[11,7,4,6,9,10],[11,5,2,1],[6,10],[0,3,9,8,7,4])
        
        

        
        # self.logs("Evaluating best combination:")
        # self.get_final_metrics(val_batches_test,[1,0,11],[8,2,7,10,9,3],[6,5,11],[7,0,3,1,9,10])
        # self.logs("Evaluating best combination --- threshold:")
        # self.get_final_metrics(val_batches_test,[1,0,11],[8,2,7,10],[6,5,11],[7,0,3,1])
        # self.logs("Evaluating best combination --- threshold 2:")
        # self.get_final_metrics(val_batches_test,[1,0,11],[8,2,7,10],[6,5,11],[7,0])
        
        # self.logs("Evaluating worst combination:")
        # self.get_final_metrics(val_batches_test,[5,4,6],[3,9,10,7,2,8] ,[8,2,4],[10,9,1,3,0,7])
        # self.logs("Evaluating worst combination -- threshold:")
        # self.get_final_metrics(val_batches_test,[5,4,6],[3,9,10,7] ,[8,2,4],[10,9,1,3])
        # self.logs("Evaluating worst combination -- threshold 2:")
        # self.get_final_metrics(val_batches_test,[5,4,6],[3,9,10,7] ,[8,2,4],[10,9])

        # self.logs("Evaluating random combination:")
        # self.get_final_metrics(val_batches_test, [10,3,7],[1,9,11,2,6,8],[3,4,2],[5,7,10,9,1,8])
        
        # self.logs("Evaluating random combination threshold:")
        # self.get_final_metrics(val_batches_test, [10,3,7],[1,9,11,2],[3,4,2],[5,7,10,9])

        # self.logs("Evaluating random combination threshold 2:")
        # self.get_final_metrics(val_batches_test, [10,3,7],[1,9,11,2],[3,4,2],[5,7])


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
                select1_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.gelu_approximation_layers[1])
                select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.gelu_approximation_layers[2])

                while self.tensor_in_list(selects_activation["selects1"], select1_activation):
                    select1_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.gelu_approximation_layers[1])
                
                while len(set(select1_activation.tolist()) & set(select2_activation.tolist())) > 0:
                    select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.gelu_approximation_layers[2])
                    while self.tensor_in_list(selects_activation["selects2"], select2_activation):
                        select2_activation = self.sampling_less_important_selection(importance_score=self.importance_score_activation, num=self.gelu_approximation_layers[2])
                
                exclude_activation = torch.cat((select1_activation, select2_activation), dim=0)  
                mask_activation = torch.isin(total_layers_index, exclude_activation, invert=True)
                select3_activation = total_layers_index[mask_activation]

                selects_activation["selects1"].append(select1_activation)
                selects_activation["selects2"].append(select2_activation)
                selects_activation["selects3"].append(select3_activation)
                
                ''' softmax selection '''
                # todo: classify the approximation degree
                select1_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.softmax_approximation_layers[2])
                select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.softmax_approximation_layers[3])
                
                while self.tensor_in_list(selects_softmax["selects2"], select1_softmax):
                    select1_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.softmax_approximation_layers[2])
                
                while len(set(select1_softmax.tolist()) & set(select2_softmax.tolist())) > 0:
                    select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.softmax_approximation_layers[3])
                    while self.tensor_in_list(selects_softmax["selects3"], select2_softmax):
                        select2_softmax = self.sampling_less_important_selection(importance_score=self.importance_score_softmax, num=self.softmax_approximation_layers[3])
                
                
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
                        self.importance_score_activation[i] += rewards[k] * prob_activation[i] * (1 - prob_activation[i]) * self.rl_lr * 1
                        if i not in selects_activation["selects2"][k]:
                            self.importance_score_activation[i] += rewards[k] * prob_activation[i] * (1 - prob_activation[i]) * self.rl_lr * 1.5
                    # softmax_importance score update
                    if i not in selects_softmax["selects2"][k]:
                        self.importance_score_softmax[i] += rewards[k] * prob_softmax[i] * (1 - prob_softmax[i]) * self.rl_lr * 1
                        if i not in selects_softmax["selects3"][k]:
                            self.importance_score_softmax[i] += rewards[k] * prob_softmax[i] * (1 - prob_softmax[i]) * self.rl_lr * 1.5
                    # else:
                    #     self.importance_score[i] -= rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr

        self.logs(f"Final Activation Importance Scores: {self.importance_score_activation.tolist()}")
        self.logs(f"Final Softmax Importance Scores: {self.importance_score_softmax.tolist()}")
        less_importance_layers_1_activation = np.argsort(np.array(self.importance_score_activation))[:self.gelu_approximation_layers[1]]
        less_importance_layers_2_activation = np.argsort(np.array(self.importance_score_activation))[self.gelu_approximation_layers[1]:self.gelu_approximation_layers[2]+self.gelu_approximation_layers[1]]
        less_importance_layers_3_activation = np.argsort(np.array(self.importance_score_activation))[self.gelu_approximation_layers[2]+self.gelu_approximation_layers[1]:]

        less_importance_layers_1_softmax = np.argsort(np.array(self.importance_score_softmax))[:self.softmax_approximation_layers[2]]
        less_importance_layers_2_softmax = np.argsort(np.array(self.importance_score_softmax))[self.softmax_approximation_layers[2]:self.softmax_approximation_layers[3]+self.gelu_approximation_layers[2]]
        less_importance_layers_3_softmax = np.argsort(np.array(self.importance_score_softmax))[self.softmax_approximation_layers[3]+self.gelu_approximation_layers[2]:]
        # check the final effect
        self.get_final_metrics(val_batches_test, less_importance_layers_1_activation, less_importance_layers_2_activation, less_importance_layers_3_activation, less_importance_layers_1_softmax,less_importance_layers_2_softmax,less_importance_layers_3_softmax)


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
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = val_batch["labels"]
                total_loss_final += outputs.loss.item()

                # 累计统计量
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        # 计算指标
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_predictions, average='macro')
        total_loss_final = total_loss_final / len(val_batches)  # Average loss over the dataset

        self.reversible_layer_handler.restore_all()
        self.logs(f"Final Gelu Layer1:{selected_layers1_gelu}; Final Gelu Layer2: {selected_layers2_gelu}; Final Gelu Layer3: {selected_layers3_gelu}; Final Softmax Layer1: {selected_layers1_softmax}; Final Softmax Layer2: {selected_layers2_softmax}; Final Softmax Layer3: {selected_layers3_softmax};")
        self.logs(f"Final Metrics - Accuracy: {accuracy}, F1 Score: {f1}, Total Loss: {total_loss_final}")
        return accuracy, f1, total_loss_final

    def get_time_reward(self, degree):
        """
        Calculate the time reward based on the time saved.
        The higher the time saved, the higher the reward.
        """
        time_reward = []
        for i in range(self.rl_action):
            # to be adjusted: confirm the relation bewtween the time saved and 
            time_reward.append(-1 * self.time_rate * degree[i])
        return time_reward
    
    def logs(self, result):
        """
        Write the importance scores to a file.
        """
        with open(self.log_path, 'a') as f:
            f.write(f"{result}\n")
