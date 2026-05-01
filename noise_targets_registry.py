"""
BERT 前向中所有“乘法操作”的噪声注入候选注册表（仅做梳理，不做注入）。

设计目的
--------
1. **梳理**：把 BERT (BertForSequenceClassification) 前向路径里
   所有 **乘法相关运算** 与它们的 **每一个参与操作数** 都登记在 ``NOISE_TARGETS``
   里 —— 不论该乘法是否已经被加噪声、不论是矩阵乘还是标量乘。
2. **可维护 / 可扩展**：要新增一个噪声候选，直接在 ``NOISE_TARGETS`` 末尾
   追加一条 dict 即可；要登记非乘法的候选（比如未来 LayerNorm 的加法），
   只需扩充 ``op_type`` 取值集合 + 加新条目。
3. **可选择**：``select(...)`` 给出按 stage / op_type / 现有噪声状态 / id 的筛选。
   以后真正写注入器时，传入 id 列表即可。

字段 schema
-----------
每个 entry 是一个 dict：

    id              str   全局唯一稳定 id，命名 = "{stage}.{component}.{op}"
    stage           str   高层归属：
                          "embeddings" | "encoder.attn" | "encoder.ffn"
                          | "pooler" | "head"
    per_layer       bool  True = 每个 transformer block 重复一次；
                          False = 全模型只出现一次
    module_path     str   该乘法所在模块路径
                          - per_layer=True 时相对于一个 BertLayer
                          - per_layer=False 时相对于 model.bert / model
    op_type         str   乘法/相关运算分类，见 OP_TYPES
    operands        list  每个操作数 = {"name", "role", "shape", "source"}
                          role ∈ OPERAND_ROLES
    current_noise   None|str  当前是否已加噪声；如已加，给出现行 NOISE_KEYS 中的键
                              (来自 config.constants.NOISE_KEYS +
                               function_handler 的 softmax/value 钩子)
    notes           str   人话说明（什么场景才生效、HF 源代码行号、坑点等）


常量速查（与 transformers 4.5x 的 BertModel 对应）
-------------------------------------------------
    H   = config.hidden_size            (bert-base = 768)
    I   = config.intermediate_size      (bert-base = 3072)
    A   = config.num_attention_heads    (bert-base = 12)
    Dh  = H // A                        (bert-base = 64)
    L   = config.num_hidden_layers      (bert-base = 12)
    V   = config.vocab_size
    Tt  = config.type_vocab_size        (=2)
    Pmx = config.max_position_embeddings(=512)

噪声键的真相来源
----------------
    config/constants.py:80   NOISE_KEYS = ("x","wq","wk","wv","wo","wffn1","wffn2")
    function_handler.py      新增的 softmax/V 注入: ("softmax_probs","value_after_softmax")

只要你后面想再扩展，比如想给 attention_scores / sqrt(d_k) 注入一个新噪声键
"qk_scale"，做法是：① 在 NOISE_KEYS 里加这个键；② 把本文件对应条目的
``current_noise`` 改成 "qk_scale"；③ 在新的 noise injector 里实现这个键的注入。
"""

from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# 取值枚举（仅作约束 / 文档；不做强校验）
# ---------------------------------------------------------------------------

OP_TYPES = (
    "linear_mm",        # Y = X @ W (+b)，nn.Linear / GPT-2 Conv1D 的 forward
    "activation_mm",    # Y = A1 @ A2，两个操作数都是运行时激活（Q@K^T、probs@V）
    "scalar_mul",       # Y = X * c，c 是编译期标量（1/sqrt(d_k)、1/2^d、多项式系数 …）
    "elementwise_mul",  # Y = A ⊙ B，逐元素相乘（head_mask、LayerNorm 的 *gamma …）
    "stat_div",         # Y = X / s(X)，分母来自 X 自身统计量；等价于 *(1/s)
                        # 用于 LayerNorm 的 /sqrt(var+eps)、softmax 的 /sum_exp
    "self_power",       # Y = X**k，由若干次 X*X 累乘构成（多项式 GELU 的 x^i、
                        # exp 近似的 (1+x/2^d)**(2^d)）
    "embedding_lookup", # 等价于 OneHot(idx) @ W；不是矩阵乘但语义上是一种线性乘
)

OPERAND_ROLES = (
    "activation",       # 运行时张量（hidden_states、Q、K、V、probs、context …）
    "weight",           # nn.Linear.weight / Conv1D.weight，可学习
    "bias",             # nn.Linear.bias，可学习；不参与乘法但列出便于完整性
    "embedding_table",  # nn.Embedding.weight，本质等价 weight
    "param_scale",      # LayerNorm.gamma / Tanh 等可学习的逐特征缩放
    "param_bias",       # LayerNorm.beta，仅做加法；不参与乘法但列出便于完整性
    "scalar_const",     # 编译期常量（1/sqrt(d_k)、1/2^d、polynomial coeff、Exp_bound）
    "statistic",        # 由 X 派生的运行时统计量（mean、var、sum_exp、x.max）
    "mask",             # head_mask / attention_mask
    "input_index",      # input_ids / token_type_ids / position_ids（embedding lookup 的下标）
)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
# 顺序按 BERT 前向时间序：
#   embeddings → (encoder.attn → encoder.ffn) × L → pooler → head
# ---------------------------------------------------------------------------

NOISE_TARGETS = [

    # =====================================================================
    # Stage 0: BertEmbeddings  (modeling_bert.py:127, forward 在 :149)
    # =====================================================================
    # 三个 embedding lookup + 1 个 LayerNorm。三个 lookup 的输出做加法
    # (embeddings = inputs_embeds + token_type_embeddings + position_embeddings)
    # —— 加法不在本注册表范围内，只登记其中的“乘法当量”。

    {
        "id": "emb.word_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "module_path": "bert.embeddings.word_embeddings",
        "op_type": "embedding_lookup",
        "operands": [
            {"name": "input_ids",       "role": "input_index",     "shape": "[B, S]"},
            {"name": "W_word",          "role": "embedding_table", "shape": "[V, H]"},
        ],
        "current_noise": None,
        "notes": "embedding lookup ≡ one_hot(input_ids) @ W_word；"
                 "若把它视作线性乘法，操作数就是 one_hot 与 W_word。",
    },
    {
        "id": "emb.token_type_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "module_path": "bert.embeddings.token_type_embeddings",
        "op_type": "embedding_lookup",
        "operands": [
            {"name": "token_type_ids",  "role": "input_index",     "shape": "[B, S]"},
            {"name": "W_token_type",    "role": "embedding_table", "shape": "[Tt, H]"},
        ],
        "current_noise": None,
        "notes": "type_vocab_size 一般 = 2。",
    },
    {
        "id": "emb.position_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "module_path": "bert.embeddings.position_embeddings",
        "op_type": "embedding_lookup",
        "operands": [
            {"name": "position_ids",    "role": "input_index",     "shape": "[1, S]"},
            {"name": "W_position",      "role": "embedding_table", "shape": "[Pmx, H]"},
        ],
        "current_noise": None,
        "notes": "仅当 position_embedding_type == 'absolute' 时启用（默认成立）。",
    },
    {
        "id": "emb.layernorm.stat_div",
        "stage": "embeddings",
        "per_layer": False,
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "stat_div",
        "operands": [
            {"name": "embeddings - mean", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/sqrt(var+eps)",   "role": "statistic",  "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm 第 1 个乘法：归一化阶段，等价于 X * (1/sqrt(var+eps))。",
    },
    {
        "id": "emb.layernorm.scale_mul",
        "stage": "embeddings",
        "per_layer": False,
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "elementwise_mul",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma",      "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm 第 2 个乘法：缩放阶段。后面还有 +beta（不在本表）。",
    },


    # =====================================================================
    # Stage 1: BertEncoder.layer[i]  (×L 次)
    # ---------------------------------------------------------------------
    # 当前 input X 的噪声 (NOISE_KEYS[0]='x') 是包在 layer.forward 上的，
    # 影响下面 attn.q_proj / attn.k_proj / attn.v_proj 三个 Linear 的 X 操作数。
    # =====================================================================

    # ---- 1A: BertSelfAttention (forward 在 modeling_bert.py:220, 这里被
    #         BertSelfAttentionWithAproximation 替换；逻辑等价) -------------

    {
        "id": "attn.q_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.query",
        "op_type": "linear_mm",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_q", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_q", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wq",
        "notes": "X 同时受 NOISE_KEYS[0]='x' 影响（在 layer.forward 入口处一次性注入）。",
    },
    {
        "id": "attn.k_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.key",
        "op_type": "linear_mm",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_k", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_k", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wk",
        "notes": "cross-attention 时输入是 encoder_hidden_states；BERT 默认是 self-attn。",
    },
    {
        "id": "attn.v_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.value",
        "op_type": "linear_mm",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_v", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_v", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wv",
        "notes": "",
    },

    {
        "id": "attn.qk_matmul",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self  (line ~540 of WithAproximation.forward)",
        "op_type": "activation_mm",
        "operands": [
            {"name": "Q",  "role": "activation", "shape": "[B, A, S, Dh]"},
            {"name": "Kt", "role": "activation", "shape": "[B, A, Dh, S]"},
        ],
        "current_noise": None,
        "notes": "attention_scores = matmul(Q, K^T)；两个操作数均为激活。",
    },
    {
        "id": "attn.qk_scale_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self  (line ~564 of WithAproximation.forward)",
        "op_type": "scalar_mul",
        "operands": [
            {"name": "attention_scores", "role": "activation",   "shape": "[B, A, S, S]"},
            {"name": "1/sqrt(Dh)",       "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "attention_scores / sqrt(attention_head_size)；除法等价 *1/sqrt(Dh)。",
    },

    # ---- 1A-softmax 内部展开（仅当用了近似 softmax 时成立。当前默认走的就是
    #      BertSelfAttentionWithAproximation，所以这些条目都是“活”的） -----

    {
        "id": "attn.softmax.expapprox.scalar_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.approximation_exponential",
        "op_type": "scalar_mul",
        "operands": [
            {"name": "x_shifted", "role": "activation",   "shape": "[B, A, S, S]"},
            {"name": "1/2**deg",  "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "exp 近似第一步：x / 2**degree（degree ∈ {1..6}）。",
    },
    {
        "id": "attn.softmax.expapprox.power",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.approximation_exponential",
        "op_type": "self_power",
        "operands": [
            {"name": "(1 + x/2**deg)", "role": "activation",   "shape": "[B, A, S, S]"},
            {"name": "exponent=2**deg","role": "scalar_const", "shape": "scalar(int)"},
        ],
        "current_noise": None,
        "notes": "torch.pow(.,2**degree)，本质是 2**degree 次自乘；对噪声分析需要分解。",
    },
    {
        "id": "attn.softmax.norm_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self.approximation_softmax",
        "op_type": "stat_div",
        "operands": [
            {"name": "exp_out", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "1/(sum_exp+1e-9)", "role": "statistic", "shape": "[B, A, S, 1]"},
        ],
        "current_noise": None,
        "notes": "softmax 的归一化除法；分母是逐 row 统计量。",
    },

    # ---- 1A 余下：head_mask、probs@V、Wo --------------------------------

    {
        "id": "attn.head_mask_mul",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self  (line ~580 of WithAproximation.forward)",
        "op_type": "elementwise_mul",
        "operands": [
            {"name": "attention_probs", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "head_mask",       "role": "mask",       "shape": "[A] (broadcastable)"},
        ],
        "current_noise": None,
        "notes": "仅当 head_mask is not None 时执行；BERT GLUE 任务通常不传，默认 None。",
    },
    {
        "id": "attn.probs_v_matmul",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.self  (line ~587 of WithAproximation.forward)",
        "op_type": "activation_mm",
        "operands": [
            {"name": "attention_probs", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "V",               "role": "activation", "shape": "[B, A, S, Dh]"},
        ],
        "current_noise": "softmax_probs / value_after_softmax",
        "notes": "现在通过 _apply_softmax_value_noise 同时给 attention_probs 和 V 加 fresh 噪声后再 matmul。"
                 "若以后想分别控制，softmax_probs 与 value_after_softmax 已经是两个独立 scaling factor。",
    },
    {
        "id": "attn.o_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.output.dense",
        "op_type": "linear_mm",
        "operands": [
            {"name": "context", "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_o",     "role": "weight",     "shape": "[H, H]"},
            {"name": "b_o",     "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": "wo",
        "notes": "context 由前面 reshape 得到；当前只噪声了 W_o，没噪声 context。",
    },

    # ---- 1B: BertSelfOutput LayerNorm（attention 残差后的 LN）-----------

    {
        "id": "attn.layernorm.stat_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.output.LayerNorm",
        "op_type": "stat_div",
        "operands": [
            {"name": "(hidden + residual) - mean", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/sqrt(var+eps)",            "role": "statistic",  "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm(dropout(o_proj_out) + X_residual) 第 1 步。",
    },
    {
        "id": "attn.layernorm.scale_mul",
        "stage": "encoder.attn",
        "per_layer": True,
        "module_path": "attention.output.LayerNorm",
        "op_type": "elementwise_mul",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma_attn", "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm 第 2 步。",
    },


    # =====================================================================
    # Stage 2: BertIntermediate + BertOutput (FFN)
    # =====================================================================

    {
        "id": "ffn.intermediate_proj",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "intermediate.dense",
        "op_type": "linear_mm",
        "operands": [
            {"name": "attention_output", "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_ffn1",           "role": "weight",     "shape": "[I, H]"},
            {"name": "b_ffn1",           "role": "bias",       "shape": "[I]"},
        ],
        "current_noise": "wffn1",
        "notes": "I = intermediate_size（bert-base = 3072）。",
    },

    # ---- GELU 内部展开（仅当走多项式近似时成立。当前 ReversibleLayerHandler
    #      会把 intermediate_act_fn 替换为 PolynomialGELU，所以这些条目通常活）

    {
        "id": "ffn.gelu.power",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "intermediate.intermediate_act_fn (PolynomialGELU)",
        "op_type": "self_power",
        "operands": [
            {"name": "x",        "role": "activation",   "shape": "[B, S, I]"},
            {"name": "i ∈ [0..degree]", "role": "scalar_const", "shape": "int"},
        ],
        "current_noise": None,
        "notes": "polynomial(x, coeff, sign) 里的 x.pow(i) for i in range(degree+1)；"
                 "degree=4 时会出现 x^0..x^4 共 5 个幂运算。",
    },
    {
        "id": "ffn.gelu.coeff_mul",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "intermediate.intermediate_act_fn (PolynomialGELU)",
        "op_type": "scalar_mul",
        "operands": [
            {"name": "powers (x^i)",      "role": "activation",   "shape": "[B, S, I, degree+1]"},
            {"name": "GELU_COEEF[deg][s]","role": "scalar_const", "shape": "[degree+1]"},
        ],
        "current_noise": None,
        "notes": "powers * coeff_tensor，逐元素再 .sum(dim=-1)；coeff 由分段（正/负）选择。",
    },

    {
        "id": "ffn.output_proj",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "output.dense",
        "op_type": "linear_mm",
        "operands": [
            {"name": "gelu_out", "role": "activation", "shape": "[B, S, I]"},
            {"name": "W_ffn2",   "role": "weight",     "shape": "[H, I]"},
            {"name": "b_ffn2",   "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": "wffn2",
        "notes": "FFN 第二个 Linear。",
    },

    {
        "id": "ffn.layernorm.stat_div",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "output.LayerNorm",
        "op_type": "stat_div",
        "operands": [
            {"name": "(ffn_out + residual) - mean", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/sqrt(var+eps)",             "role": "statistic",  "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm(dropout(ffn2_out) + attention_output) 第 1 步。",
    },
    {
        "id": "ffn.layernorm.scale_mul",
        "stage": "encoder.ffn",
        "per_layer": True,
        "module_path": "output.LayerNorm",
        "op_type": "elementwise_mul",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma_ffn",  "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "LayerNorm 第 2 步。",
    },


    # =====================================================================
    # Stage 3: BertPooler（仅 [CLS] token 走这一支；序列分类用得到）
    # =====================================================================

    {
        "id": "pooler.dense",
        "stage": "pooler",
        "per_layer": False,
        "module_path": "bert.pooler.dense",
        "op_type": "linear_mm",
        "operands": [
            {"name": "hidden_states[:, 0]", "role": "activation", "shape": "[B, H]"},
            {"name": "W_pool",              "role": "weight",     "shape": "[H, H]"},
            {"name": "b_pool",              "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "pooler.dense；之后接 Tanh。",
    },
    # Tanh 当前是 nn.Tanh()，逐元素非线性，没有“可选操作数”意义上的乘法；
    # 如果以后把它替换成多项式近似，再补一组 (pooler.tanh.power, pooler.tanh.coeff_mul)。


    # =====================================================================
    # Stage 4: 分类头  (BertForSequenceClassification.classifier)
    # =====================================================================

    {
        "id": "head.classifier",
        "stage": "head",
        "per_layer": False,
        "module_path": "classifier",
        "op_type": "linear_mm",
        "operands": [
            {"name": "pooled_dropout", "role": "activation", "shape": "[B, H]"},
            {"name": "W_clf",          "role": "weight",     "shape": "[num_labels, H]"},
            {"name": "b_clf",          "role": "bias",       "shape": "[num_labels]"},
        ],
        "current_noise": None,
        "notes": "MRPC 例子里 num_labels=2。",
    },
]


# ---------------------------------------------------------------------------
# 选择 / 索引工具
# ---------------------------------------------------------------------------

def select(
    *,
    ids: Optional[Iterable[str]] = None,
    stage: Optional[str] = None,
    op_type: Optional[str] = None,
    per_layer: Optional[bool] = None,
    has_noise: Optional[bool] = None,
    operand_role: Optional[str] = None,
):
    """按条件筛选注册表条目。所有参数互相 AND。

    例：
        select(stage="encoder.attn", op_type="linear_mm")     # 取 Q/K/V/Wo
        select(has_noise=False, op_type="linear_mm")          # 还没加噪声的所有 Linear
        select(operand_role="activation", per_layer=True)     # 所有“激活操作数”候选
        select(ids=["attn.qk_matmul", "attn.probs_v_matmul"]) # 直接按 id 取
    """
    id_set = set(ids) if ids is not None else None
    out = []
    for t in NOISE_TARGETS:
        if id_set is not None and t["id"] not in id_set:
            continue
        if stage is not None and t["stage"] != stage:
            continue
        if op_type is not None and t["op_type"] != op_type:
            continue
        if per_layer is not None and t["per_layer"] != per_layer:
            continue
        if has_noise is not None:
            if has_noise != (t["current_noise"] is not None):
                continue
        if operand_role is not None:
            if not any(op.get("role") == operand_role for op in t["operands"]):
                continue
        out.append(t)
    return out


def get(target_id: str) -> dict:
    for t in NOISE_TARGETS:
        if t["id"] == target_id:
            return t
    raise KeyError(target_id)


def list_ids() -> list:
    return [t["id"] for t in NOISE_TARGETS]


__all__ = [
    "OP_TYPES",
    "OPERAND_ROLES",
    "NOISE_TARGETS",
    "select",
    "get",
    "list_ids",
]
