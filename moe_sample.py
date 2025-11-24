import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络
        self.experts = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts)])
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        gate_logits = self.gate(x)  # 计算门控网络的分数
        gate_probs = F.softmax(gate_logits, dim=-1)  # 归一化
        
        top_k_values, top_k_indices = torch.topk(gate_probs, self.top_k)  # 选择Top-K专家
        top_k_experts = [self.experts[i](x) for i in top_k_indices[0]]  # 计算选中专家的输出
        
        # 按权重加权求和
        output = sum(w * e for w, e in zip(top_k_values[0], top_k_experts))
        return output

# 测试
moe_model = MoE(input_dim=16, num_experts=4, top_k=2)
x = torch.randn(1, 16)  # 输入数据
output = moe_model(x)
print(output.shape)  # 输出维度


# importance matrix
class Importance:
    def __init__(self, layer_dim, expert_dim, layer_importance = [], expert_importance = []):
        self.layer_importance = layer_importance
        self.expert_importance = expert_importance
    
# Dynamic Importance
# Importance dynamicly change according to the imputs
class Dynamic_Importance(Importance):
    def __init__(self, layer_dim, expert_dim, layer_importance=[], expert_importance=[]):
        super().__init__(layer_dim, expert_dim, layer_importance, expert_importance)
    
    # round is the input round that take into consideration, the importance is the round-th Markov chain
    def update(self, ciphertext_input, round, change_interval):
        return

# Static Importance
# Importance is independent from the inputs, and has been calculated before online inference
class Static_Importance(Importance):
    def __init__(self, layer_dim, expert_dim, layer_importance=[], expert_importance=[]):
        super().__init__(layer_dim, expert_dim, layer_importance, expert_importance)

