import math
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention
import copy
from torch import Tensor
from typing import Optional


# GELU approximation coeff
# GELU_COEEF[i][0]-positive; GELU_COEEF[i][1]-negative (-2.7, 2.7)
GELU_COEEF = {
            # degree 0: same coefficients as degree 1, but applied without piecewise comparison
            0: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],
            # todo: change the pivot point of degree 1
            # pivot point: use SEAF- -2.5, -0.75, 0 , 0.5, 2.5?
            1: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],
            2: [[-0.12136484, 0.94386247, 0.04261206],[-0.12136484, -0.44386247+0.5, 0.04261206]],
            # relative error: -0.75 pivot point
            # 1: [[5.9839183235390844e-05, 0.6170698026386807], [-0.2052886977665538, -0.0759420475301809]],
            # 2: [[4.680008304412681e-06, 0.4740042074483325, 0.29206518457930236],[-0.3773006655410396, -0.25069817033674346, -0.04239126463806122]],
            
            
            3: [[-0.01524885, 0.57426473, 0.35500657, -0.07415983], [-0.01524885, -0.07426473+0.5, 0.35500657, 0.07415983]],
            4: [[0.00746413, -0.07087454+0.5, 0.58960402, -0.20949432, 0.02540485], [ 0.00746413, 0.07087454+0.5, 0.58960402, 0.20949432, 0.02540485]]
            # 4: [[0.00162080853184154, -0.03798164612714154+0.5, 0.5410550166368381, -0.18352506127082727, 0.020848611754127593], [0.00162080853184154, 0.03798164612714154+0.5, 0.5410550166368381, 0.18352506127082727, 0.020848611754127593]]
}

# SiLU approximation coeff (Bumblebee piecewise)
SiLU_COEEF = {
#             1: [[0.7618737346704126, 0.5000893434026534],[-0.10118073891975127,-0.013543261873265973]],
            1: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.10118073891975127,-0.013543261873265973]],
            2: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.2932427892002413,-0.07801652478737445,-0.005269243960262952]],
            3: [[0.14241236482342567, 0.4999863582405589, 0.12920235286785606, 0],[-0.4233567569791515,-0.14755599495248886,-0.017365847597972207,-0.0006859293250386277]],
            4: [[0.03284668051202981,0.5000000914210826,0.19746490458050728,0,-0.005281681095454781],[-0.49057828462086733,-0.02757518199120323,0.05336178194846048,0.011409101768158705,0.0006606624719387583]]
}


# Exponential approximation Taylor bound
Exp_bound = {
    1:-2,
    2:-4,
    3:-8,
    4:-12,
    5:-13,
    6:-13
}

# Transformer-layer input noise table.
# Values are variances sigma^2 for N(0, sigma^2).
# Current x-noise injection uses the "fresh" column.
INPUT_NOISE_VARIANCE_TABLE = {
    10: {"encoding": 6.510416e-04, "fresh": 1.310800e+03, "rescale": 5.333984e+00},
    12: {"encoding": 4.069010e-05, "fresh": 8.192500e+01, "rescale": 3.333740e-01},
    14: {"encoding": 2.543131e-06, "fresh": 5.120312e+00, "rescale": 2.083588e-02},
    16: {"encoding": 1.589457e-07, "fresh": 3.200195e-01, "rescale": 1.302242e-03},
    18: {"encoding": 9.934107e-09, "fresh": 2.000122e-02, "rescale": 8.139014e-05},
    20: {"encoding": 6.208817e-10, "fresh": 1.250076e-03, "rescale": 5.086884e-06},
    22: {"encoding": 3.880511e-11, "fresh": 7.812977e-05, "rescale": 3.179302e-07},
    24: {"encoding": 2.425319e-12, "fresh": 4.883110e-06, "rescale": 1.987064e-08},
    26: {"encoding": 1.515824e-13, "fresh": 3.051944e-07, "rescale": 1.241915e-09},
    28: {"encoding": 9.473903e-15, "fresh": 1.907465e-08, "rescale": 7.761969e-11},
    30: {"encoding": 5.921189e-16, "fresh": 1.192166e-09, "rescale": 4.851231e-12},
    32: {"encoding": 3.700743e-17, "fresh": 7.451035e-11, "rescale": 3.032019e-13},
    34: {"encoding": 2.312965e-18, "fresh": 4.656897e-12, "rescale": 1.895012e-14},
    36: {"encoding": 1.445603e-19, "fresh": 2.910561e-13, "rescale": 1.184382e-15},
    38: {"encoding": 9.035018e-21, "fresh": 1.819100e-14, "rescale": 7.402390e-17},
    40: {"encoding": 5.646886e-22, "fresh": 1.136938e-15, "rescale": 4.626494e-18},
    42: {"encoding": 3.529304e-23, "fresh": 7.105861e-17, "rescale": 2.891559e-19},
    44: {"encoding": 2.205815e-24, "fresh": 4.441163e-18, "rescale": 1.807224e-20},
    46: {"encoding": 1.378634e-25, "fresh": 2.775727e-19, "rescale": 1.129515e-21},
    48: {"encoding": 8.616464e-27, "fresh": 1.734829e-20, "rescale": 7.059470e-23},
}

INPUT_NOISE_ALLOWED_SCALING_FACTORS = (20, 22, 24, 26, 28, 30)
INPUT_NOISE_DEFAULT_SCALING_FACTOR = 30
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS = (10, 12, 14, 16, 18, 20, 22)
WEIGHT_NOISE_DEFAULT_SCALING_FACTOR = 22


def get_input_noise_variance(scaling_factor: int, distribution: str = "fresh") -> float:
    if scaling_factor not in INPUT_NOISE_VARIANCE_TABLE:
        raise ValueError(
            f"Unsupported scaling factor {scaling_factor}. "
            f"Supported values: {sorted(INPUT_NOISE_VARIANCE_TABLE)}"
        )
    distribution_key = str(distribution).lower()
    if distribution_key not in INPUT_NOISE_VARIANCE_TABLE[scaling_factor]:
        raise ValueError(
            f"Unsupported input-noise distribution '{distribution}'. "
            "Use one of: encoding, fresh, rescale."
        )
    return float(INPUT_NOISE_VARIANCE_TABLE[scaling_factor][distribution_key])


def add_gaussian_input_noise(
        hidden_states: Tensor,
        scaling_factor: int,
        distribution: str = "fresh"
        ) -> Tensor:
    variance = get_input_noise_variance(scaling_factor, distribution=distribution)
    if variance <= 0.0:
        return hidden_states
    std = math.sqrt(variance)
    noise = torch.randn_like(hidden_states) * std
    return hidden_states + noise


def _make_input_noise_forward(original_forward, scaling_factor: int, distribution: str = "fresh"):
    def noisy_forward(hidden_states, *args, **kwargs):
        if hidden_states is None:
            return original_forward(hidden_states, *args, **kwargs)
        noisy_hidden_states = add_gaussian_input_noise(
            hidden_states,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        return original_forward(noisy_hidden_states, *args, **kwargs)
    return noisy_forward


def add_gaussian_weight_noise(
        weight: Tensor,
        scaling_factor: int,
        distribution: str = "encoding"
        ) -> Tensor:
    variance = get_input_noise_variance(scaling_factor, distribution=distribution)
    if variance <= 0.0:
        return weight
    std = math.sqrt(variance)
    noise = torch.randn_like(weight) * std
    return weight + noise


def _make_noisy_linear_forward(linear_module: nn.Linear, scaling_factor: int, distribution: str = "encoding"):
    def noisy_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        noisy_weight = add_gaussian_weight_noise(
            linear_module.weight,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        return nn.functional.linear(hidden_states, noisy_weight, linear_module.bias)
    return noisy_forward

# Tanh approximation coeff
Tanh_COEEF = {
            1: [[0.5, 0.5], [0.5, 0.5]],
            2: [[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]],
            3: [[0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5]],
            4: [[0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5]]
}          

# Less than coeff (to be done, not sure can be approximated in mpc evaluation)
Less_than_COEEF = {
            1: [[0.5, 0.5], [0.5, 0.5]],
            2: [[0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]],
            3: [[0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5]],
            4: [[0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5]]
}

# Sqrt 1/rootsq
ReSqrt_COEFF = {

}



# millionaire approximation used in protocol type: 0->mpc; 1->HE
def less_than_approximaion (
        x: torch.Tensor, 
        coeff: Optional[list] = None, 
        sign: int = 1, 
        protocol_type: 
        int = 0
        ) -> torch.Tensor:
    pass
# tensor polynomial approximation
def polynomial(x, coeff, sign):
    # x: Tensor, 可能在 cuda:0 或 cpu
    device = x.device
    dtype  = x.dtype

    # 1. 生成 x 的幂
    powers = torch.stack([x.pow(i) for i in range(len(coeff[sign]))], dim=-1)

    # 2. 在同一设备上创建系数 Tensor
    coeff_tensor = torch.tensor(
        coeff[sign],
        device=device,
        dtype=dtype
    )

    # 3. 按维度相乘求和
    return (powers * coeff_tensor).sum(dim=-1)

class PolynomialGELU(nn.Module):
    """可逆的三次多项式GELU近似"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeff = GELU_COEEF[degree]  # 正向系数
        self.degree = degree
        
    def forward(self, x: Tensor) -> Tensor:

        if self.degree == 0:
            # Degree 0: skip piecewise comparison, directly use [-2.7, 0] interval polynomial
            return polynomial(x, self.coeff, 1)

        y0 = torch.zeros_like(x, dtype=x.dtype, device=x.device) 
        y1 = polynomial(x, self.coeff, 1)
        y2 = polynomial(x, self.coeff, 0)
        y3 = x
        
        # 创建与x相同设备和类型的输出张量
        
        if(self.degree == 1 or self.degree == 2):
            # degree 1, use the Bumblebee piecewise
            mask_low = x < -2.7
            mask_neg = (x >= -2.7) & (x < 0)
            mask_pos = (x >= 0) & (x <= 2.7)
            mask_high = x > 2.7
        else:
            mask_low = x < -2.7
            mask_neg = (x >= -2.7) & (x < 0)
            mask_pos = (x >= 0) & (x <= 2.7)
            mask_high = x > 2.7
        
        # 分段处理
        # print(f"y0 : {y0}, y1 : {y1}, y2 : {y2}, y3 : {y3}")
        out = torch.where(mask_low, y0, torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, y3, out)

        # print(f"X : {x}, Y : {out}, OriginGelu: {origin}")
        return out
    
class PolynomiaTanh(nn.Module):
    """可逆的三次多项式GELU近似"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeff = GELU_COEEF[degree]  # 正向系数

        
    def forward(self, x: Tensor) -> Tensor:

        y0 = torch.zeros_like(x, dtype=x.dtype, device=x.device) 
        y1 = polynomial(x, self.coeff, 1)
        y2 = polynomial(x, self.coeff, 0)
        y3 = x
        
        # 创建与x相同设备和类型的输出张量
        mask_low = x < -2.7
        mask_neg = (x >= -2.7) & (x < 0)
        mask_pos = (x >= 0) & (x <= 2.7)
        mask_high = x > 2.7
        
        # 分段处理
        # print(f"y0 : {y0}, y1 : {y1}, y2 : {y2}, y3 : {y3}")
        out = torch.where(mask_low, y0, torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, y3, out)
        return out

# change BertsdpaAttention to normal self attention and change its softmax
class BertSelfAttentionWithAproximation(BertSelfAttention):
    """BertSelfAttention with softmax approximation"""
    def __init__(self, config, degree, lower_bound):
        super().__init__(config)
        self.degree = degree 
        self.lower_bound = lower_bound

    def approximation_exponential(self, x: torch.Tensor) -> torch.Tensor:
        """近似计算指数函数""" # degree = 1,2,3,4,5,6 
        x = torch.pow(1 + x / (2 ** self.degree), 2 ** self.degree) 
        return x


    # do approximation softmax
    def approximation_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """使用指数近似计算softmax"""
        # print("do approximation softmax")
        # 计算指数近似,  < lower bound的exp值为0 
        # need to be optimized
        # (degree, lower_bound) -> (1, -2), (2, -4), (3, -10), (4, -13), (5, -13), (6, -13)
        x = x - x.max(dim=-1, keepdim=True)[0] + 1e-9  # 数值稳定处理
        # print(f"This is x: {x}")
        # print(torch.isnan(x).any(), torch.isinf(x).any())  # 检测异常值
        # print(x.abs().max())  # 确认数值量级

        exp_approx = self.approximation_exponential(x)
        exp_out = torch.where(x < self.lower_bound, torch.zeros_like(x), exp_approx)
        sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9
        # print(f"this is exp_out: {exp_out}; this is sum_exp: {sum_exp}")
        return exp_out / sum_exp  # 统一使用掩码后结果

    # error construction
    # def error_construction(self, scales: torch.Tensor) -> torch.Tensor:
    #     absolute_error = torch.
        
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
       # the original BertSelfAttention forward function
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Changed Softmax approximation
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.approximation_softmax(attention_scores)

        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class PerturbedLiner(nn.Module):
    """可逆的三次多项式GELU近似"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeff = GELU_COEEF[degree]  # 正向系数
        
    def forward(self, x: Tensor) -> Tensor:

        y0 = torch.zeros_like(x, dtype=x.dtype, device=x.device) 
        y1 = polynomial(x, 1)
        y2 = polynomial(x, 0)
        y3 = x
        
        # 创建与x相同设备和类型的输出张量
        mask_low = x < -2.7
        mask_neg = (x >= -2.7) & (x < 0)
        mask_pos = (x >= 0) & (x <= 2.7)
        mask_high = x > 2.7
        
        # 分段处理
        # print(f"y0 : {y0}, y1 : {y1}, y2 : {y2}, y3 : {y3}")
        out = torch.where(mask_low, y0, torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, y3, out)

        origin = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        # print(f"X : {x}, Y : {out}, OriginGelu: {origin}")
        return out
    
class ReversibleLayerHandler:
    """管理GELU函数替换/恢复的工具类"""
    def __init__(self, model):
        self.model = model
        self.original_gelu = {}
        self.original_attention = {}
        self.original_input_noise = {}
        self.original_projection_noise = {
            "query": {},
            "key": {},
            "value": {},
            "wo": {},
            "wffn1": {},
            "wffn2": {},
        }
        self.backup_model = copy.deepcopy(model)  # 完整模型备份
    
    def replace_layer_gelu(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """替换指定层的GELU函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        # 'act_fn': layer.mlp.act_fn
                        'act_fn': layer.intermediate.intermediate_act_fn
                    }
                
                # 应用新函数
                # layer.mlp.act_fn = PolynomialGELU(degree=degree)
                layer.intermediate.intermediate_act_fn = PolynomialGELU(degree=degree)
                # layer.output.activation = PolynomialGELU(degree=degree)
        
        print(f"已替换 {len(layer_indices)} 层的GELU函数")
    
    def replace_layer_norm(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """替换指定层的LayerNorm函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        # 'act_fn': layer.mlp.act_fn
                        'act_fn': layer.intermediate.intermediate_act_fn
                    }
                
                # 应用新函数
                # layer.mlp.act_fn = nn.LayerNorm(layer.mlp.hidden_size)
                layer.intermediate.intermediate_act_fn = nn.LayerNorm(layer.intermediate.intermediate_size)
                # layer.output.activation = nn.LayerNorm(layer.output.size)
    
    def replace_layer_tanh(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """替换指定层的Tanh函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        # 'act_fn': layer.mlp.act_fn
                        'act_fn': layer.intermediate.intermediate_act_fn
                    }
                
                # 应用新函数
                # layer.mlp.act_fn = nn.Tanh()
                layer.intermediate.intermediate_act_fn = nn.Tanh()
                # layer.output.activation = nn.Tanh()

    def replace_layer_linear(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        pass

    def replace_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention", degree=1):
        """替换指定层的Softmax函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_attention:
                    self.original_attention[i] = {
                        'attention': eval("layer."+ attention_name)
                    }
                
                # 应用新函数
                orig_sd = layer.attention.self.state_dict()
                new_attn = BertSelfAttentionWithAproximation(self.model.config, degree=degree, lower_bound=Exp_bound[degree])
                new_attn.load_state_dict(orig_sd, strict=False)
                layer.attention.self = new_attn
        
        print(f"已替换 {len(layer_indices)} 层的Softmax函数")
    
    def replace_layer_input_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="fresh"
            ):
        """Inject x-noise on transformer-layer inputs: x + N(0, sigma^2)."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            stored_forward = self.original_input_noise.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not layer:
                self.original_input_noise[i] = {
                    "forward": layer.forward,
                }

            original_forward = self.original_input_noise[i]["forward"]
            layer.forward = _make_input_noise_forward(
                original_forward,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            self.original_input_noise[i]["scaling_factor"] = int(scaling_factor)
            self.original_input_noise[i]["distribution"] = str(distribution).lower()

        print(
            f"Input noise enabled for {len(selected)} layers "
            f"(scaling_factor={int(scaling_factor)}, distribution={distribution})"
        )

    def _replace_attention_projection_noise(
            self,
            projection_name,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        """Temporarily use (W + We) inside Q/K/V projection without mutating the stored weight."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        projection_store = self.original_projection_noise.setdefault(projection_name, {})
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            projection_module = getattr(layer.attention.self, projection_name)
            stored_forward = projection_store.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not projection_module:
                projection_store[i] = {
                    "forward": projection_module.forward,
                }

            projection_module.forward = _make_noisy_linear_forward(
                projection_module,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            projection_store[i]["scaling_factor"] = int(scaling_factor)
            projection_store[i]["distribution"] = str(distribution).lower()

        print(
            f"{projection_name} noise enabled for {len(selected)} layers "
            f"(scaling_factor={int(scaling_factor)}, distribution={distribution})"
        )

    def _replace_layer_linear_module_noise(
            self,
            store_key,
            module_path,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        """Temporarily use (W + We) inside a layer Linear module without mutating the stored weight."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        projection_store = self.original_projection_noise.setdefault(store_key, {})
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            linear_module = eval("layer." + module_path)
            stored_forward = projection_store.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not linear_module:
                projection_store[i] = {
                    "forward": linear_module.forward,
                }

            linear_module.forward = _make_noisy_linear_forward(
                linear_module,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            projection_store[i]["scaling_factor"] = int(scaling_factor)
            projection_store[i]["distribution"] = str(distribution).lower()

        print(
            f"{store_key} noise enabled for {len(selected)} layers "
            f"(scaling_factor={int(scaling_factor)}, distribution={distribution})"
        )

    def replace_layer_query_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "query",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_key_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "key",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_value_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "value",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_attention_output_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wo",
            "attention.output.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_ffn1_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wffn1",
            "intermediate.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_ffn2_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wffn2",
            "output.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def restore_layer_gelu(self, layer_indices=None, layer_name="model.model.layers"):
        """恢复指定层的原始GELU函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_gelu:
                # layer.mlp.act_fn = self.original_gelu[i]['act_fn']
                layer.intermediate.intermediate_act_fn = self.original_gelu[i]['act_fn']
                # layer.output.activation = self.original_gelu[i]['output']
        
        print(f"已恢复 {len(layer_indices)} 层的原始GELU函数")
    
    def restore_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention"):
        """恢复指定层的原始Softmax函数"""
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_attention:
                # eval("layer."+ attention_name) = self.original_attention[i]['attention']
                layer.attention.self = self.original_attention[i]['attention']

   
    def restore_layer_input_noise(self, layer_indices=None, layer_name="model.model.layers"):
        """Restore original transformer-layer inputs for selected layers."""
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i in selected and i in self.original_input_noise:
                original_forward = self.original_input_noise[i]["forward"]
                if getattr(original_forward, "__self__", None) is layer:
                    layer.forward = original_forward
                del self.original_input_noise[i]

    def _restore_attention_projection_noise(
            self,
            projection_name,
            layer_indices=None,
            layer_name="model.model.layers"
            ):
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        projection_store = self.original_projection_noise.get(projection_name, {})
        for i, layer in enumerate(layers):
            if i in selected and i in projection_store:
                projection_module = getattr(layer.attention.self, projection_name)
                original_forward = projection_store[i]["forward"]
                if getattr(original_forward, "__self__", None) is projection_module:
                    projection_module.forward = original_forward
                del projection_store[i]

    def _restore_layer_linear_module_noise(
            self,
            store_key,
            module_path,
            layer_indices=None,
            layer_name="model.model.layers"
            ):
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        projection_store = self.original_projection_noise.get(store_key, {})
        for i, layer in enumerate(layers):
            if i in selected and i in projection_store:
                linear_module = eval("layer." + module_path)
                original_forward = projection_store[i]["forward"]
                if getattr(original_forward, "__self__", None) is linear_module:
                    linear_module.forward = original_forward
                del projection_store[i]

    def restore_layer_query_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "query",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_key_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "key",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_value_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "value",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_attention_output_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wo",
            "attention.output.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn1_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn1",
            "intermediate.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn2_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn2",
            "output.dense",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_all(self):
        """完全恢复原始模型状态"""
        self.model = copy.deepcopy(self.backup_model)
        self.original_gelu = {}
        self.original_attention = {}
        self.original_input_noise = {}
        self.original_projection_noise = {
            "query": {},
            "key": {},
            "value": {},
            "wo": {},
            "wffn1": {},
            "wffn2": {},
        }
        print("已完全恢复原始模型状态")
