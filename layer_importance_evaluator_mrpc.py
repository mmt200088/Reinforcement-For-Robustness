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
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=100, degree=2, device='cpu'):
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
        self.reversible_gelu_handler = ReversibleLayerHandler(self.model)
        
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
        self.importance_score = torch.zeros(self.total_layers)

        ### hyper parameters reinforecement learning step/rate
        # number of candidate networks(RL joint action) in each RL step
        self.rl_action = 3
        # updating rate of importance score
        self.rl_lr = rl_lr
        self.update_importance_steps = 20
        self.log_path = f"./importance_scores_lr{self.rl_lr}_steps{self.update_importance_steps}_degree{self.degree}_actions{self.rl_action}_model{model_class_name}.txt" # Path to save importance scores

        self.train_batch_number = 52 # Number of training batches to evaluate importance scores
        self.update_importance_interval_steps = 1 # Set to 0 for continuous evaluation

        # hyper parameters for layer importance evaluation
        # take the saved time and the declined accuracy into account 
        self.time_rate = 0.25 # benefit for reward 
        self.accurate_rate = 0.25 # harmful to reward
        
        # number of layers to be updated in each RL step
        self.gelu_approximation_factor = [0, 0, 1, 0, 0] # degree 0,1,2,3,4 polynomial approximation facto
        # number of layers to be updated with degree 0,1,2,3,4 polynomial approximation
        self.gelu_approximation_layers = [round(x*self.total_layers) for x in self.gelu_approximation_factor]



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
        layers = eval('self.' + self.layers_attribute)

        
        # for idx in range(self.total_layers):
        #     for name, module in layers[idx].named_modules():
        #         if hasattr(module, 'scaling'):
        #             self.raw_scaling = module.scaling
        #         if hasattr(module, 'adapter_scaling'):
        #             self.raw_scaling = module.adapter_scaling
        #         if hasattr(module, 'disable_adapters'):
        #             for name, param in module.named_parameters():
        #                 if param.requires_grad and name not in self.trainable_module_name:
        #                     self.trainable_module_name.append(name)
        

        # if self.raw_scaling is not None:
        #     print(f'default scaling is {self.raw_scaling}')
        # else:
        #     raise NotImplementedError

    # sample layers index based on importance score
    def sampling_less_important_selection(self, num):
        if num == 0:
            return torch.tensor([], dtype=torch.long)
        prob = (-self.importance_score).sigmoid()
        select = torch.sort(torch.multinomial(prob, num))[0]
        return select

    def sampling_more_important_selection(self, num):
        if num == 0:
            return torch.tensor([], dtype=torch.long)
        prob = (self.importance_score).sigmoid()
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

        # loss test
        # self.get_final_loss(val_batches_test, np.array([23,0,8,21,2]), np.array([14,18,22,15,20,19]))
        # self.get_final_loss(val_batches_test, np.array([14,17,7,1,6]), np.array([9,23,0,8,21,2])) 
        # self.get_final_loss(val_batches_test, np.array([9,5,16,4,10]), np.array([13,14,17,7,1,6])) 
        # self.get_final_loss(val_batches_test, np.array([3,8,5,7,21]), np.array([23,12,14,6,1,9])) 
       

        # accuracy test
        self.logs("Evaluating worst:")
        self.get_final_loss(val_batches_test,[], [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
        self.get_final_metrics(val_batches_test,[], [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        # self.logs("Evaluating middle:")
        # self.get_final_loss(val_batches_test, [9,6,8], [7,2,4])
        # self.get_final_metrics(val_batches_test, [9,6,8], [7,2,4])

        # self.logs("Evaluating best:")
        # self.get_final_loss(val_batches_test, [0,11,10], [9,6,8])
        # self.get_final_metrics(val_batches_test, [0,11,10], [9,6,8])


        for _ in range(self.update_importance_steps):
            self.logs(f"In the {_}'th rl step, updating importance scores...")
            self.logs(f"Current Importance Scores: {self.importance_score.tolist()}")

            selects = {
                "selects1": [],
                "selects2": [],
                "selects3": [],
                "selects4": []
            }
            # Record the losses for each action
            rets = []
            # Calculate rewards based on the losses
            rewards = []

            # try:
            #     val_batch = next(self.dataloader)
            # except:
            #     self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
            #                                     collate_fn=self.data_collator)
            #     self.dataloader = iter(self.dataloader)
            #     val_batch = next(self.dataloader)

            # Move batch to device (cpu/cuda)
            # val_batch = {k: v.to(self.device) for k, v in val_batch.items()}

            for k in range(self.rl_action):
                # todo: classify the approximation degree
                # now: degree 1 polynomial approximation
                select1 = self.sampling_less_important_selection(num=self.gelu_approximation_layers[1])
                select2 = self.sampling_less_important_selection(num=self.gelu_approximation_layers[2])
                
                while self.tensor_in_list(selects["selects1"], select1):
                    select1 = self.sampling_less_important_selection(num=self.gelu_approximation_layers[1])
                
                while len(set(select1.tolist()) & set(select2.tolist())) > 0:
                    select2 = self.sampling_less_important_selection(num=self.gelu_approximation_layers[2])
                    while self.tensor_in_list(selects["selects2"], select2):
                        select2 = self.sampling_less_important_selection(num=self.gelu_approximation_layers[2])
                

                selects["selects1"].append(select1)
                selects["selects2"].append(select2)
                
                # self.switch_active_adapter(select)

                # Evaluate the model with the selected layers
                # Replace the GELU function in the selected layers with degree 1 polynomial approximation
                # to do: divide into 4 groups according to importance

                self.reversible_gelu_handler.replace_layer_gelu(layer_indices=select1, layer_name=self.layers_attribute, degree=1)
                self.reversible_gelu_handler.replace_layer_gelu(layer_indices=select2, layer_name=self.layers_attribute, degree=2)
                
                self.reversible_gelu_handler.model.eval()
                
                # do actior to get rewards
                # inference mode is used to avoid gradient calculation
                total_loss = 0
                now_batch_num = 0
                for val_batch in val_batches:
                    if now_batch_num < self.train_batch_number:
                        now_batch_num += 1
                        print(f"Evaluating with batch: {val_batch}")
                        with torch.inference_mode():
                            outputs = self.reversible_gelu_handler.model(**val_batch)
                            total_loss += outputs.loss.item()
                    else:
                        break    
                
                total_loss = total_loss / self.train_batch_number  # Average loss over the dataset
                # self.model.train()
                self.logs(f"RL Action {k}, Selected1 Layers: {select1}, Selected2 Layers: {select2}, Loss: {total_loss}")
                rets.append(total_loss)

                # todo recover the original gelu function
                self.reversible_gelu_handler.restore_all() 

            self.logs(f"All action Losses: {rets}")

            for i in range(self.rl_action):
                rewards.append(math.exp(-rets[i]))

            _mean = np.mean(rewards)

            # rewards = np.array([(r - _mean) for r in rewards]).tolist() + \
            #     self.get_time_reward(rewards)


            # without time reward term
            rewards = np.array([(r - _mean) for r in rewards]).tolist()
            self.logs(f"Rewards: {rewards}")
            prob = self.importance_score.sigmoid()
            self.logs(f"Probs:{prob}")
            for k in range(self.rl_action):
                for i in range(self.total_layers):
                    if i not in selects["selects1"][k]:
                        self.importance_score[i] += rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr * 2
                    if i not in selects["selects2"][k]:
                        self.importance_score[i] += rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr
                    # else:
                    #     self.importance_score[i] -= rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr

        self.logs(f"Final Importance Scores: {self.importance_score.tolist()}")
        less_importance_layers_1 = np.argsort(np.array(self.importance_score))[:self.gelu_approximation_layers[1]]
        less_importance_layers_2 = np.argsort(np.array(self.importance_score))[self.gelu_approximation_layers[1]:self.gelu_approximation_layers[2]+self.gelu_approximation_layers[1]]
        
        # check the final effect
        self.get_final_loss(val_batches_test, less_importance_layers_1, less_importance_layers_2)
        self.get_final_metrics(val_batches_test, less_importance_layers_1, less_importance_layers_2)

        
    
    # calculate the reward based on the time saved (he)
    # or the communication cost saved (mpc)
    
    def get_final_loss(self, val_batches=[], selected_layers1=[], selected_layers2=[]):
        # check the final effect
        total_loss_final = 0

        if len(selected_layers1) == 0 and len(selected_layers2) == 0:
            self.logs("No layers selected for evaluation.")
        else:
            self.reversible_gelu_handler.replace_layer_gelu(layer_indices=selected_layers1, layer_name=self.layers_attribute, degree=1)
            self.reversible_gelu_handler.replace_layer_gelu(layer_indices=selected_layers2, layer_name=self.layers_attribute, degree=2)
        
        self.reversible_gelu_handler.model.eval()

        for val_batch in val_batches:
            print(f"Evaluating with batch: {val_batch}")
            with torch.inference_mode():
                outputs = self.reversible_gelu_handler.model(**val_batch)
                total_loss_final += outputs.loss.item()
        total_loss_final = total_loss_final / len(val_batches)  # Average loss over the dataset
        self.reversible_gelu_handler.restore_all()  # Restore the original GELU function
        self.logs(f"Final Selected Layers 1: {np.sort(selected_layers1)}, Selected Layers 2: {selected_layers2}, Loss: {total_loss_final}")
        return total_loss_final


    def get_final_metrics(self, val_batches=[], selected_layers1=[], selected_layers2=[]):
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        if len(selected_layers1) == 0 and len(selected_layers2) == 0:
            self.logs("No layers selected for evaluation.")
        else:
            self.reversible_gelu_handler.replace_layer_gelu(layer_indices=selected_layers1, layer_name=self.layers_attribute, degree=1)
            self.reversible_gelu_handler.replace_layer_gelu(layer_indices=selected_layers2, layer_name=self.layers_attribute, degree=2)
            
        self.reversible_gelu_handler.model.eval()

        for val_batch in val_batches:
            with torch.inference_mode():
                outputs = self.reversible_gelu_handler.model(**val_batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = val_batch["labels"]
                
                # 累计统计量
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        self.reversible_gelu_handler.restore_all()
        self.logs(f"Layers1: {selected_layers1}, Layers2: {selected_layers2} | "
                f"Accuracy: {accuracy:.4f}, Macro-F1: {f1:.4f}")
        return accuracy, f1



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
    
    # choose active layers of approximation in each step 
    # use inference to calculate the loss
    # def active_all_adapter(self):
    #     self.model.train()
    #     layers = eval('self.' + self.layers_attribute)
    #     for idx in range(self.total_layers):
    #         for name, module in layers[idx].named_modules():
    #             if hasattr(module, 'scaling'):
    #                 module.scaling = self.raw_scaling
    #             if hasattr(module, 'adapter_scaling'):
    #                 module.adapter_scaling = self.raw_scaling

    # def switch_active_adapter(self, select):
    #     layers = eval('self.' + self.layers_attribute)
    #     for idx in range(self.total_layers):
    #         if idx in select:  # disable lora
    #             for name, module in layers[idx].named_modules():
    #                 if hasattr(module, 'scaling'):
    #                     module.scaling = self.raw_scaling * self.response_suppression_factor
    #                 if hasattr(module, 'adapter_scaling'):
    #                     module.adapter_scaling = self.raw_scaling * self.response_suppression_factor
    #         else:
    #             for name, module in layers[idx].named_modules():
    #                 if hasattr(module, 'scaling'):
    #                     module.scaling = self.raw_scaling
    #                 if hasattr(module, 'adapter_scaling'):
    #                     module.adapter_scaling = self.raw_scaling

    # def switch_active_layers(self):
    #     # First, disable gradients for all layers
    #     self.freeze_all_layers()

    #     # Randomly select n_layers to activate
    #     layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
    #     self.active_layers_indices = self.sampling_more_important_selection(self.n_layers_updated)
    #     print(
    #         f"Total layers: {self.total_layers}, Activating layers at indices: {self.active_layers_indices} for the next steps.",
    #         flush=True)

    #     # Enable gradients only for the selected layers
    #     for idx in self.active_layers_indices:
    #         for name, module in layers[idx].named_modules():
    #             if hasattr(module, 'disable_adapters'):
    #                 for name, param in module.named_parameters():
    #                     if name in self.trainable_module_name:
    #                         param.requires_grad = True
    def logs(self, result):
        """
        Write the importance scores to a file.
        """
        with open(self.log_path, 'a') as f:
            f.write(f"{result}\n")
