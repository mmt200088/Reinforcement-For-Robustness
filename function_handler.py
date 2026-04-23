import math
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as _GPT2Conv1D
except Exception:  # pragma: no cover - transformers always ships this, but be defensive
    _GPT2Conv1D = None
try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as _GPT2Attention
except Exception:  # pragma: no cover
    _GPT2Attention = None
import copy
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers shared by BERT and GPT-2 code paths.
# ---------------------------------------------------------------------------
def _get_attr_path(obj, path):
    """Resolve a dotted attribute path starting from ``obj``."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_attr_path(obj, path, value):
    """Set a dotted attribute path starting from ``obj``."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _is_gpt2_conv1d(module) -> bool:
    return _GPT2Conv1D is not None and isinstance(module, _GPT2Conv1D)


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

INPUT_NOISE_ALLOWED_SCALING_FACTORS = (22, 24, 26, 28, 30)
INPUT_NOISE_DEFAULT_SCALING_FACTOR = 30
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS = (14, 16, 18, 20, 22)
WEIGHT_NOISE_DEFAULT_SCALING_FACTOR = 22
WFFN1_NOISE_ALLOWED_SCALING_FACTORS = (16, 18, 20, 22, 24)
WFFN1_NOISE_DEFAULT_SCALING_FACTOR = 24
SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS = tuple(sorted(INPUT_NOISE_VARIANCE_TABLE))
SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR = max(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS)


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


def _format_noise_distribution_label(distribution: str) -> str:
    distribution_key = str(distribution).lower()
    distribution_labels = {
        "fresh": "新采样（fresh）",
        "encoding": "编码分布（encoding）",
        "rescale": "重缩放（rescale）",
    }
    return distribution_labels.get(distribution_key, str(distribution))


def _format_noise_target_label(target_key: str) -> str:
    target_labels = {
        "input": "输入噪声（Input noise）",
        "query": "查询投影噪声（query noise）",
        "key": "键投影噪声（key noise）",
        "value": "值投影噪声（value noise）",
        "wo": "注意力输出投影噪声（wo noise）",
        "wffn1": "前馈网络第一层噪声（wffn1 noise）",
        "wffn2": "前馈网络第二层噪声（wffn2 noise）",
    }
    return target_labels.get(str(target_key).lower(), f"{target_key} 噪声")


def _format_noise_enable_message(
        target_key: str,
        layer_count: int,
        scaling_factor: int,
        distribution: str
        ) -> str:
    return (
        f"已为 {int(layer_count)} 层启用{_format_noise_target_label(target_key)}，"
        f"缩放因子（scaling_factor）={int(scaling_factor)}，"
        f"分布（distribution）={_format_noise_distribution_label(distribution)}"
    )


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


def _apply_softmax_value_noise(attention_probs: Tensor, value_layer: Tensor, owner) -> tuple:
    """Apply fresh tensor noise to attention_probs and value_layer before attention matmul."""
    state = getattr(owner, "_softmax_value_noise_state", None)
    if not state:
        return attention_probs, value_layer

    distribution = str(state.get("distribution", "fresh")).lower()
    softmax_scaling_factor = state.get("softmax_scaling_factor")
    value_scaling_factor = state.get("value_scaling_factor")

    noisy_attention_probs = attention_probs
    noisy_value_layer = value_layer
    if softmax_scaling_factor is not None:
        noisy_attention_probs = add_gaussian_input_noise(
            attention_probs,
            scaling_factor=int(softmax_scaling_factor),
            distribution=distribution,
        )
    if value_scaling_factor is not None:
        noisy_value_layer = add_gaussian_input_noise(
            value_layer,
            scaling_factor=int(value_scaling_factor),
            distribution=distribution,
        )
    return noisy_attention_probs, noisy_value_layer


def _make_noisy_linear_forward(linear_module: nn.Linear, scaling_factor: int, distribution: str = "encoding"):
    def noisy_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        noisy_weight = add_gaussian_weight_noise(
            linear_module.weight,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return nn.functional.linear(hidden_states, noisy_weight, bias)
    return noisy_forward


def _make_noisy_conv1d_forward(conv1d, scaling_factor: int, distribution: str = "encoding"):
    """Weight-noise forward for HuggingFace GPT-2 ``Conv1D`` (weight shape ``[in, out]``)."""
    def noisy_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        noisy_weight = add_gaussian_weight_noise(
            conv1d.weight,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = conv1d.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        size_out = hidden_states.size()[:-1] + (conv1d.nf,)
        out = torch.addmm(
            bias,
            hidden_states.view(-1, hidden_states.size(-1)),
            noisy_weight,
        )
        return out.view(size_out)
    return noisy_forward


def _make_noisy_projection_forward(module, scaling_factor: int, distribution: str = "encoding"):
    """Dispatch to the right noisy-forward builder depending on module type."""
    if _is_gpt2_conv1d(module):
        return _make_noisy_conv1d_forward(module, scaling_factor, distribution)
    return _make_noisy_linear_forward(module, scaling_factor, distribution)

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
    def __init__(self, config, degree, lower_bound, position_embedding_type=None, layer_idx=None):
        try:
            super().__init__(
                config,
                position_embedding_type=position_embedding_type,
                layer_idx=layer_idx,
            )
        except TypeError:
            try:
                super().__init__(
                    config,
                    position_embedding_type=position_embedding_type,
                )
            except TypeError:
                super().__init__(config)
        if position_embedding_type is not None:
            self.position_embedding_type = position_embedding_type
        self.layer_idx = layer_idx
        self.degree = degree 
        self.lower_bound = lower_bound
        self._softmax_value_noise_state = None

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
        
    
    def _looks_like_attention_mask(self, value) -> bool:
        return torch.is_tensor(value) and value.dim() >= 2

    def _looks_like_cache_position(self, value) -> bool:
        return (
            torch.is_tensor(value)
            and value.dim() <= 1
            and value.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.long,
            )
        )

    def _looks_like_cache(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, (tuple, list)):
            return True
        return any(
            hasattr(value, attr)
            for attr in ("update", "is_updated", "layers", "self_attention_cache", "cross_attention_cache")
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        *args,
        **kwargs,
    ):
       # Follow the current transformers BERT attention flow and replace
       # only the softmax step with the approximation variant.
        encoder_attention_mask = kwargs.pop("encoder_attention_mask", None)
        past_key_value = kwargs.pop("past_key_value", None)
        past_key_values = kwargs.pop("past_key_values", None)
        output_attentions = kwargs.pop("output_attentions", False)
        cache_position = kwargs.pop("cache_position", None)

        tail = list(args)

        if isinstance(past_key_value, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_value
            past_key_value = None
        if isinstance(past_key_values, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_values
            past_key_values = None

        if tail and self._looks_like_cache_position(tail[-1]) and cache_position is None:
            cache_position = tail.pop()

        if tail and isinstance(tail[-1], bool):
            output_attentions = tail.pop()

        if encoder_hidden_states is not None and tail:
            first = tail[0]
            if encoder_attention_mask is None and (first is None or self._looks_like_attention_mask(first)):
                encoder_attention_mask = tail.pop(0)

        if past_key_value is None and past_key_values is None and tail:
            candidate = tail.pop(0)
            if isinstance(candidate, bool):
                if output_attentions in (False, None):
                    output_attentions = candidate
                candidate = None
            elif (
                encoder_hidden_states is None
                and encoder_attention_mask is None
                and self._looks_like_attention_mask(candidate)
            ):
                # Some legacy positional paths may still include a placeholder
                # encoder-attention mask slot even for encoder-only BERT.
                encoder_attention_mask = candidate
                candidate = tail.pop(0) if tail else None
            past_key_value = candidate

        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        elif past_key_values is None and past_key_value is not None:
            past_key_values = past_key_value

        if isinstance(past_key_value, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_value
            past_key_value = None
            past_key_values = None

        batch_size, _, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        curr_past_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "is_updated"):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        if is_cross_attention and curr_past_key_value is not None and is_updated:
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
            value_layer = self.value(current_states)
            value_layer = value_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

            if curr_past_key_value is not None:
                if hasattr(curr_past_key_value, "update"):
                    cache_position = cache_position if not is_cross_attention else None
                    key_layer, value_layer = curr_past_key_value.update(
                        key_layer,
                        value_layer,
                        self.layer_idx,
                        {"cache_position": cache_position},
                    )
                    if is_cross_attention and hasattr(past_key_value, "is_updated"):
                        past_key_value.is_updated[self.layer_idx] = True
                elif self._looks_like_cache(curr_past_key_value):
                    key_layer = torch.cat([curr_past_key_value[0], key_layer], dim=2)
                    value_layer = torch.cat([curr_past_key_value[1], value_layer], dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_value is not None:
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

        context_attention_probs, context_value_layer = _apply_softmax_value_noise(
            attention_probs,
            value_layer,
            self,
        )
        context_layer = torch.matmul(context_attention_probs, context_value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# ---------------------------------------------------------------------------
# GPT-2 Softmax 近似: 通过 monkey-patch eager_attention_forward 实现
# ---------------------------------------------------------------------------

def _approx_exponential(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Taylor 展开近似 exp(x), degree 控制精度."""
    return torch.pow(1 + x / (2 ** degree), 2 ** degree)


def _approx_softmax(x: torch.Tensor, degree: int, lower_bound: float) -> torch.Tensor:
    """使用指数近似计算 softmax, 与 BertSelfAttentionWithAproximation 保持一致."""
    x = x - x.max(dim=-1, keepdim=True)[0] + 1e-9
    exp_approx = _approx_exponential(x, degree)
    exp_out = torch.where(x < lower_bound, torch.zeros_like(x), exp_approx)
    sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9
    return exp_out / sum_exp


def _make_gpt2_approx_attn_forward(attn_module, degree: int, lower_bound: float):
    """构造一个替代 GPT2Attention.forward 的函数, 将 softmax 替换为近似版本.

    该函数完整复制 HuggingFace eager_attention_forward 的计算逻辑,
    唯一区别是把 ``nn.functional.softmax(attn_weights, dim=-1)`` 换成
    ``_approx_softmax(attn_weights, degree, lower_bound)``.
    """
    original_forward = attn_module.forward

    def _approx_eager_attention(module, query, key, value,
                                attention_mask, head_mask=None, **kwargs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if module.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5,
                dtype=attn_weights.dtype, device=attn_weights.device,
            )
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            attn_weights = attn_weights / float(module.layer_idx + 1)
        if not module.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = module.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full(
                [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device,
            )
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        # ----- 核心替换: 使用近似 softmax -----
        attn_weights = _approx_softmax(attn_weights, degree, lower_bound)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = module.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        context_attn_weights, context_value = _apply_softmax_value_noise(
            attn_weights,
            value,
            module,
        )
        attn_output = torch.matmul(context_attn_weights, context_value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    def patched_forward(hidden_states, *args, **kwargs):
        """替换 GPT2Attention.forward, 强制使用带近似 softmax 的 eager attention."""
        # 保存原始 _attn_implementation, 临时强制 eager 模式
        orig_impl = attn_module.config._attn_implementation
        orig_reorder = attn_module.reorder_and_upcast_attn
        attn_module.config._attn_implementation = "eager"
        attn_module.reorder_and_upcast_attn = False
        # 注入自定义 attention 函数
        import transformers.models.gpt2.modeling_gpt2 as _gpt2_mod
        _saved_fn = _gpt2_mod.eager_attention_forward
        _gpt2_mod.eager_attention_forward = _approx_eager_attention
        try:
            result = original_forward(hidden_states, *args, **kwargs)
        finally:
            _gpt2_mod.eager_attention_forward = _saved_fn
            attn_module.config._attn_implementation = orig_impl
            attn_module.reorder_and_upcast_attn = orig_reorder
        return result

    return patched_forward


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
    """管理GELU/Softmax/噪声替换与恢复的工具类.

    支持两类 backbone:
      * BERT 家族 (bert-base / bert-large, roberta): 依赖 ``attention.self.{query,key,value}``
        / ``attention.output.dense`` / ``intermediate.dense`` / ``output.dense``
        / ``intermediate.intermediate_act_fn`` 的模块路径.
      * GPT-2 家族 (openai-community/gpt2): 使用融合的 ``attn.c_attn`` (Conv1D) +
        ``attn.c_proj`` / ``mlp.c_fc`` / ``mlp.c_proj`` / ``mlp.act``.
        由于 c_attn 把 Q/K/V 融合成一个 Conv1D, 这里通过一次性包装 c_attn.forward,
        在单层上按需累加 q/k/v 各自的权重噪声.
    """

    # Layer-local path tables (relative to a single transformer block).
    _BERT_PATHS = {
        "gelu_act": "intermediate.intermediate_act_fn",
        "wo_dense": "attention.output.dense",
        "wffn1_dense": "intermediate.dense",
        "wffn2_dense": "output.dense",
    }
    _GPT2_PATHS = {
        "gelu_act": "mlp.act",
        "wo_dense": "attn.c_proj",
        "wffn1_dense": "mlp.c_fc",
        "wffn2_dense": "mlp.c_proj",
    }

    @staticmethod
    def _detect_arch(model) -> str:
        """Return ``'gpt2'`` or ``'bert'`` based on top-level module layout."""
        # GPT-2 style: has .transformer.h (list of GPT2Block)
        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return "gpt2"
        # BERT / RoBERTa style
        if hasattr(model, "bert") or hasattr(model, "roberta"):
            return "bert"
        # Fallback: inspect first layer attributes
        for attr in ("bert", "roberta", "transformer"):
            sub = getattr(model, attr, None)
            if sub is None:
                continue
            layers = getattr(sub, "h", None) or getattr(getattr(sub, "encoder", None), "layer", None)
            if layers is None or len(layers) == 0:
                continue
            first = layers[0]
            if hasattr(first, "attn") and hasattr(first.attn, "c_attn"):
                return "gpt2"
            if hasattr(first, "attention") and hasattr(first.attention, "self"):
                return "bert"
        return "bert"

    def __init__(self, model):
        self.model = model
        self._arch = self._detect_arch(model)
        self._paths = self._GPT2_PATHS if self._arch == "gpt2" else self._BERT_PATHS
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
        self.original_softmax_value_noise = {}
        # GPT-2 fused Q/K/V state: {layer_idx: {"query"/"key"/"value": (sf, distribution)}}
        self._gpt2_qkv_state = {}
        # Wrapped c_attn registry so we install the proxy forward only once per layer.
        self._gpt2_qkv_wrapped = {}
        self.backup_model = copy.deepcopy(model)  # 完整模型备份
    
    def replace_layer_gelu(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """替换指定层的GELU函数 (BERT: intermediate.intermediate_act_fn; GPT-2: mlp.act)"""
        act_path = self._paths["gelu_act"]
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        "act_fn": _get_attr_path(layer, act_path),
                    }
                orig_act = _get_attr_path(layer, act_path)
                orig_training = getattr(orig_act, "training", layer.training)
                new_act = PolynomialGELU(degree=degree)
                new_act.train(bool(orig_training))
                _set_attr_path(layer, act_path, new_act)

        print(f"已替换 {len(layer_indices)} 层的GELU函数（GELU function）")
    
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
        """替换指定层的Softmax函数 (BERT: 替换 BertSelfAttention; GPT-2: monkey-patch forward)"""
        if self._arch == "gpt2":
            lb = Exp_bound.get(degree)
            if lb is None:
                print(f"[ReversibleLayerHandler] 警告: degree={degree} 没有对应的 Exp_bound, 跳过 softmax 近似.")
                return
            for i, layer in enumerate(eval("self." + layer_name)):
                if i in layer_indices:
                    if i not in self.original_attention:
                        self.original_attention[i] = {
                            'attention_forward': layer.attn.forward,
                        }
                    layer.attn.forward = _make_gpt2_approx_attn_forward(
                        layer.attn, degree=degree, lower_bound=lb,
                    )
            print(f"已替换 {len(layer_indices)} 层的Softmax函数（GPT-2 approximate softmax, degree={degree}）")
            return
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_attention:
                    self.original_attention[i] = {
                        'attention': eval("layer."+ attention_name)
                    }

                # 应用新函数
                orig_self = layer.attention.self
                orig_sd = orig_self.state_dict()
                new_attn = BertSelfAttentionWithAproximation(
                    self.model.config,
                    degree=degree,
                    lower_bound=Exp_bound[degree],
                    position_embedding_type=getattr(orig_self, "position_embedding_type", None),
                    layer_idx=getattr(orig_self, "layer_idx", None),
                )
                new_attn.load_state_dict(orig_sd, strict=False)
                new_attn = new_attn.to(
                    device=orig_self.query.weight.device,
                    dtype=orig_self.query.weight.dtype,
                )
                new_attn.train(orig_self.training)
                layer.attention.self = new_attn

        print(f"已替换 {len(layer_indices)} 层的Softmax函数（Softmax function）")
    
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

        print(_format_noise_enable_message("input", len(selected), scaling_factor, distribution))

    def _get_attention_core_module(self, layer):
        if self._arch == "gpt2":
            return layer.attn
        return layer.attention.self

    def replace_layer_softmax_value_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            softmax_scaling_factor=SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            value_scaling_factor=SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="fresh",
            ):
        """Inject fresh noise as (softmax + e1) @ (V + e2) in attention."""
        _ = get_input_noise_variance(int(softmax_scaling_factor), distribution=distribution)
        _ = get_input_noise_variance(int(value_scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        state = {
            "softmax_scaling_factor": int(softmax_scaling_factor),
            "value_scaling_factor": int(value_scaling_factor),
            "distribution": str(distribution).lower(),
        }
        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            attn_module = self._get_attention_core_module(layer)
            self.original_softmax_value_noise.setdefault(
                i,
                getattr(attn_module, "_softmax_value_noise_state", None),
            )
            setattr(attn_module, "_softmax_value_noise_state", dict(state))

        print(
            "Enabled softmax/V attention-product noise for "
            f"{len(selected)} layers "
            f"(softmax sf={int(softmax_scaling_factor)}, "
            f"V sf={int(value_scaling_factor)}, distribution={str(distribution).lower()})"
        )

    def _ensure_gpt2_qkv_wrapper(self, layer_idx, layer):
        """Install a single proxy forward on this layer's ``attn.c_attn``.

        The proxy reads ``self._gpt2_qkv_state[layer_idx]`` each forward pass and
        adds per-slice weight noise (query / key / value) on top of the untouched
        base Conv1D output. This keeps the three projection noises independent
        even though GPT-2 stores them as a single fused Conv1D.
        """
        if layer_idx in self._gpt2_qkv_wrapped:
            return
        c_attn = layer.attn.c_attn  # HuggingFace Conv1D
        original_forward = c_attn.forward
        handler = self
        hidden_size = c_attn.nf // 3

        def proxy_forward(hidden_states, *args, **kwargs):
            base = original_forward(hidden_states, *args, **kwargs)
            state = handler._gpt2_qkv_state.get(layer_idx)
            if not state:
                return base
            result = base.clone()
            in_dim = hidden_states.size(-1)
            for slot_name, slot_idx in (("query", 0), ("key", 1), ("value", 2)):
                params = state.get(slot_name)
                if params is None:
                    continue
                sf, dist = params
                variance = get_input_noise_variance(int(sf), distribution=dist)
                if variance <= 0.0:
                    continue
                std = math.sqrt(variance)
                noise_w = torch.randn(
                    in_dim, hidden_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ) * std
                noise_out = torch.matmul(hidden_states, noise_w)
                start = slot_idx * hidden_size
                end = start + hidden_size
                result[..., start:end] = result[..., start:end] + noise_out
            return result

        c_attn.forward = proxy_forward
        self._gpt2_qkv_wrapped[layer_idx] = {
            "c_attn": c_attn,
            "forward": original_forward,
        }

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

        # GPT-2 fused c_attn path: use a per-layer proxy and accumulate state.
        if self._arch == "gpt2":
            for i, layer in enumerate(layers):
                if i not in selected:
                    continue
                self._ensure_gpt2_qkv_wrapper(i, layer)
                state = self._gpt2_qkv_state.setdefault(i, {})
                state[projection_name] = (int(scaling_factor), str(distribution).lower())
            print(_format_noise_enable_message(projection_name, len(selected), scaling_factor, distribution))
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

        print(_format_noise_enable_message(projection_name, len(selected), scaling_factor, distribution))

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

            linear_module.forward = _make_noisy_projection_forward(
                linear_module,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            projection_store[i]["scaling_factor"] = int(scaling_factor)
            projection_store[i]["distribution"] = str(distribution).lower()

        print(_format_noise_enable_message(store_key, len(selected), scaling_factor, distribution))

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
            self._paths["wo_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_ffn1_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wffn1",
            self._paths["wffn1_dense"],
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
            self._paths["wffn2_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def restore_layer_gelu(self, layer_indices=None, layer_name="model.model.layers"):
        """恢复指定层的原始GELU函数"""
        act_path = self._paths["gelu_act"]
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_gelu:
                _set_attr_path(layer, act_path, self.original_gelu[i]["act_fn"])

        print(f"已恢复 {len(layer_indices)} 层的原始GELU函数（original GELU function）")
    
    def restore_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention"):
        """恢复指定层的原始Softmax函数"""
        if self._arch == "gpt2":
            for i, layer in enumerate(eval("self." + layer_name)):
                if i in layer_indices and i in self.original_attention:
                    original_fwd = self.original_attention[i].get('attention_forward')
                    if original_fwd is not None:
                        layer.attn.forward = original_fwd
                    del self.original_attention[i]
            return
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_attention:
                current_training = layer.attention.self.training
                restored_attention = self.original_attention[i]['attention']
                restored_attention.train(bool(current_training))
                layer.attention.self = restored_attention

   
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

    def restore_layer_softmax_value_noise(self, layer_indices=None, layer_name="model.model.layers"):
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
            attn_module = self._get_attention_core_module(layer)
            previous_state = self.original_softmax_value_noise.pop(i, None)
            setattr(attn_module, "_softmax_value_noise_state", previous_state)

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

        if self._arch == "gpt2":
            for i in list(selected):
                state = self._gpt2_qkv_state.get(i)
                if state is None:
                    continue
                state.pop(projection_name, None)
                if not state:
                    self._gpt2_qkv_state.pop(i, None)
                    # All three slots cleared — restore the base Conv1D forward
                    wrapped = self._gpt2_qkv_wrapped.pop(i, None)
                    if wrapped is not None:
                        wrapped["c_attn"].forward = wrapped["forward"]
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
            self._paths["wo_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn1_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn1",
            self._paths["wffn1_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn2_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn2",
            self._paths["wffn2_dense"],
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
        self.original_softmax_value_noise = {}
        self._gpt2_qkv_state = {}
        self._gpt2_qkv_wrapped = {}
        print("已完全恢复原始模型状态")
