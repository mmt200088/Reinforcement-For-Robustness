"""
BERT 前向中所有"乘法操作"的噪声注入候选注册表（仅梳理；不做注入）。

设计目的
--------
1. **梳理**：把 BertForSequenceClassification 前向路径里所有 **乘法相关运算**
   及其每一个 **参与操作数** 都登记在 ``NOISE_TARGETS`` 中。
2. **可维护 / 可扩展**：要新增噪声候选，直接在 ``NOISE_TARGETS`` 末尾追加 dict。
3. **可选择**：``select(...)`` 支持按 id / stage / op_type / blb_block / blb_N /
   degrees / has_noise 筛选。

字段 schema
-----------
每个 entry 是一个 dict：

    id              str    全局唯一稳定 id；命名 ``"{stage}.{component}.{op}"``
    stage           str    "embeddings" | "encoder.attn" | "encoder.ffn"
                           | "pooler" | "head"
    per_layer       bool   True = 每个 transformer block 重复一次
    blb_block       int|str|None
                           1, 2, 3, 4, 5 = BLB Figure 10 的 5 个 fused block；
                           "embeddings" / "pooler" / "head" = 不在 5-block 循环内；
                           None = 跨 block 或位置待定。
    blb_N           int|None
                           BLB 该 block 用的 CKKS 多项式阶（8192 或 16384），
                           对应 NOISE_VARIANCE_TABLE_BY_N 里的 key。None = 不固定
                           或不在 BLB 5-block 循环里。
    degrees         tuple|None
                           softmax/GELU 这种 degree 相关的 op：列出该 id 在
                           哪些 degree 下生效；None = 与 degree 无关。
    shared_with     list   该 id 必须与表里其它哪些 id 用 **相同** scaling factor
                           / 噪声选择（BLB 协议约束，比如 Q/K 共享）。
    module_path     str    PyTorch 中该乘法所在模块路径（per_layer=True 时相对
                           于一个 BertLayer）。
    op_type         str    乘法分类，见 OP_TYPES。
    blb_op          str    BLB Figure 10 / Table 2 里对应的 CKKS 算子名
                           （matmulcc / matmulcp / smulcp / ewmulcp / ewmulcc / sum 等）。
    operands        list   {"name", "role", "shape"} × N；role ∈ OPERAND_ROLES。
    current_noise   None|str
                           当前是否已加噪声；如已加，给出现行 NOISE_KEYS 中的键
                           （"x", "wq", "wk", "wv", "wo", "wffn1", "wffn2",
                           "softmax_probs", "value_after_softmax"）。
    notes           str    人话说明。


常量速查
--------
    H   = config.hidden_size            (bert-base = 768)
    I   = config.intermediate_size      (bert-base = 3072)
    A   = config.num_attention_heads    (bert-base = 12)
    Dh  = H // A                        (bert-base = 64)
    L   = config.num_hidden_layers      (bert-base = 12)
    V   = config.vocab_size
    Tt  = config.type_vocab_size        (= 2)
    Pmx = config.max_position_embeddings(= 512)
"""

from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# 取值枚举
# ---------------------------------------------------------------------------

OP_TYPES = (
    "linear_mm",        # Y = X @ W (+b)，nn.Linear / Conv1D
    "activation_mm",    # Y = A1 @ A2，两个操作数都是激活（Q@K^T / probs@V）
    "scalar_mul",       # Y = X * c，c 是编译期标量
    "elementwise_mul",  # Y = A ⊙ B，逐元素乘
    "self_power",       # Y = X**k，由若干次 X*X 累乘构成
    "stat_div",         # Y = X / s(X)，分母来自 X 自身统计量
    "embedding_lookup", # 等价于 OneHot · W
)

OPERAND_ROLES = (
    "activation",       # 运行时张量
    "weight",           # nn.Linear.weight / Conv1D.weight
    "bias",
    "embedding_table",
    "param_scale",      # LayerNorm.gamma 之类
    "param_bias",       # LayerNorm.beta（仅加法）
    "scalar_const",     # 1/√Dh、1/2^d、polynomial coeff、Exp_bound
    "statistic",        # mean、var、sum_exp、x.max
    "mask",             # head_mask / attention_mask / BLB 协议里的 mask 明文
    "input_index",      # input_ids 之类
)


# ---------------------------------------------------------------------------
# 注册表：按 BLB Figure 10 时间序排列
#   embeddings → (Block 1 → Block 2 → Block 3 → Block 4 → Block 5) × L
#   → pooler → head
# 注：BLB block 内含跨层融合，PyTorch 时序上每条都属于其 module_path 所在 layer。
# ---------------------------------------------------------------------------

NOISE_TARGETS = [

    # =====================================================================
    # Stage 0: BertEmbeddings (一次性，不在 BLB 5-block 循环内)
    # =====================================================================
    {
        "id": "emb.word_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.word_embeddings",
        "op_type": "embedding_lookup",
        "blb_op": "matmulcp (≡ OneHot·W)",
        "operands": [
            {"name": "input_ids", "role": "input_index",     "shape": "[B, S]"},
            {"name": "W_word",    "role": "embedding_table", "shape": "[V, H]"},
        ],
        "current_noise": None,
        "notes": "词嵌入查表",
    },
    {
        "id": "emb.token_type_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.token_type_embeddings",
        "op_type": "embedding_lookup",
        "blb_op": "matmulcp",
        "operands": [
            {"name": "token_type_ids", "role": "input_index",     "shape": "[B, S]"},
            {"name": "W_token_type",   "role": "embedding_table", "shape": "[Tt, H]"},
        ],
        "current_noise": None,
        "notes": "句子类型嵌入；Tt 一般 = 2",
    },
    {
        "id": "emb.position_lookup",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.position_embeddings",
        "op_type": "embedding_lookup",
        "blb_op": "matmulcp",
        "operands": [
            {"name": "position_ids", "role": "input_index",     "shape": "[1, S]"},
            {"name": "W_position",   "role": "embedding_table", "shape": "[Pmx, H]"},
        ],
        "current_noise": None,
        "notes": "仅 absolute position embedding 时启用（默认）",
    },

    # ----- Embedding LayerNorm (head + tail，与 attn/ffn LN 结构相同) -----
    {
        "id": "emb.layernorm.head.mean_smul",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_x",   "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",     "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "LN head 第 1 步：均值 = (Σx) · (1/D)",
    },
    {
        "id": "emb.layernorm.head.center_ctpt",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× square mask)",
        "operands": [
            {"name": "x",            "role": "activation", "shape": "[B, S, H]"},
            {"name": "square_mask",  "role": "mask",       "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "BLB 协议里的 ct*pt：把 x 与一个'平方掩码'相乘，为接下来的 ct*ct 做准备",
    },
    {
        "id": "emb.layernorm.head.square_ctct",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (square)",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
        ],
        "current_noise": None,
        "notes": "LN head：求 (x − μ)² 的 ct*ct 自乘",
    },
    {
        "id": "emb.layernorm.head.var_smul",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_xc2", "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",     "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "LN head 末步：variance = (Σ(x−μ)²) · (1/D)；之后是 rsqrt 非线性",
    },
    {
        "id": "emb.layernorm.tail.normalize_ctct",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (× 1/std)",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/std",      "role": "activation", "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "LN tail 第 1 步：normalize = (x − μ) · (1/std)；rsqrt 之后",
    },
    {
        "id": "emb.layernorm.tail.scale_ctpt",
        "stage": "embeddings",
        "per_layer": False,
        "blb_block": "embeddings",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.embeddings.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× γ)",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma",      "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "LN tail 第 2 步：× γ；之后 + β（不算乘法）",
    },

    # =====================================================================
    # BLB Block 1: Wffn2 + post-FFN LN head（PyTorch 时序上属于本层 ffn output）
    # 用 N=8192 表
    # =====================================================================
    {
        "id": "ffn.output_proj",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 1,
        "blb_N": 8192,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.dense",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (X · W_ffn2)",
        "operands": [
            {"name": "gelu_out", "role": "activation", "shape": "[B, S, I]"},
            {"name": "W_ffn2",   "role": "weight",     "shape": "[H, I]"},
            {"name": "b_ffn2",   "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": "wffn2",
        "notes": "BLB Block 1 的起点；I = intermediate_size",
    },
    {
        "id": "ffn.layernorm.head.mean_smul",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 1,
        "blb_N": 8192,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_x", "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",   "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "post-FFN LN head：mean 计算（Rotation Sum1: 3 之后的 ct*pt）",
    },
    {
        "id": "ffn.layernorm.head.center_ctpt",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 1,
        "blb_N": 8192,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× square mask)",
        "operands": [
            {"name": "x",           "role": "activation", "shape": "[B, S, H]"},
            {"name": "square_mask", "role": "mask",       "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "post-FFN LN head：BLB 协议里的 ct*pt with square mask",
    },
    {
        "id": "ffn.layernorm.head.square_ctct",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 1,
        "blb_N": 8192,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
        ],
        "current_noise": None,
        "notes": "post-FFN LN head：(x − μ)² 自乘",
    },
    {
        "id": "ffn.layernorm.head.var_smul",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 1,
        "blb_N": 8192,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_xc2", "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",     "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "Block 1 末步：variance；输出 (x−μ)²·1/D 给 rsqrt（Block 1 / Block 2 边界非线性）",
    },

    # =====================================================================
    # BLB Block 2: post-FFN LN tail + Q/K/V 投影 + Q·K^T
    # 用 N=16384 表
    # =====================================================================
    {
        "id": "ffn.layernorm.tail.normalize_ctct",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (× 1/std)",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/std",      "role": "activation", "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "post-FFN LN tail 第 1 步：rsqrt 之后的 ct*ct: x-mean/std",
    },
    {
        "id": "ffn.layernorm.tail.scale_ctpt",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× γ)",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma_ffn",  "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "post-FFN LN tail 第 2 步：× γ；输出 γ·(x−μ)/std 给 Block 2 后续",
    },
    {
        "id": "attn.q_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": ["attn.k_proj"],
        "module_path": "attention.self.query",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (X · W_q)",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_q", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_q", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wq",
        "notes": "BLB 约束：Q/K 必须共享 scaling factor（动作选择一致）",
    },
    {
        "id": "attn.k_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": ["attn.q_proj"],
        "module_path": "attention.self.key",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (X · W_k)",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_k", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_k", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wk",
        "notes": "BLB 约束：与 attn.q_proj 共享 scaling factor",
    },
    {
        "id": "attn.v_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,    # 🚧 TBD：用户稍后补充 V 在 BLB Figure 10 里的精确位置
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self.value",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (X · W_v)",
        "operands": [
            {"name": "X",   "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_v", "role": "weight",     "shape": "[A*Dh, H]"},
            {"name": "b_v", "role": "bias",       "shape": "[A*Dh]"},
        ],
        "current_noise": "wv",
        "notes": "🚧 V 投影的 BLB block 归属待用户补完图后确认；当前默认 Block 2",
    },
    {
        "id": "attn.q.bsgs_mask.step1",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 _block2_q_bsgs_hook 第 1 步)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (Q ⊙ ones-mask_1)",
        "operands": [
            {"name": "Q",          "role": "activation",    "shape": "[B, A, S, Dh]"},
            {"name": "ones_mask_1", "role": "plaintext_mask", "shape": "[B, A, S, Dh]"},
        ],
        "current_noise": None,
        "notes": "BLB BSGS 转置/重排第 1 步；明文模拟版：与全 1 plaintext 按位乘",
    },
    {
        "id": "attn.q.bsgs_mask.step2",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 _block2_q_bsgs_hook 第 2 步)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (Q' ⊙ ones-mask_2)",
        "operands": [
            {"name": "Q_after_mask1", "role": "activation",    "shape": "[B, A, S, Dh]"},
            {"name": "ones_mask_2",   "role": "plaintext_mask", "shape": "[B, A, S, Dh]"},
        ],
        "current_noise": None,
        "notes": "BLB BSGS 转置/重排第 2 步；与第 1 步噪声参数独立",
    },
    {
        "id": "attn.kt.bsgs_mask.step1",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 _block2_kt_bsgs_hook 第 1 步)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (K^T ⊙ ones-mask_1)",
        "operands": [
            {"name": "Kt",          "role": "activation",    "shape": "[B, A, Dh, S]"},
            {"name": "ones_mask_1", "role": "plaintext_mask", "shape": "[B, A, Dh, S]"},
        ],
        "current_noise": None,
        "notes": "BLB BSGS 转置/重排第 1 步；明文模拟版：与全 1 plaintext 按位乘",
    },
    {
        "id": "attn.kt.bsgs_mask.step2",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 _block2_kt_bsgs_hook 第 2 步)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (K^T' ⊙ ones-mask_2)",
        "operands": [
            {"name": "Kt_after_mask1", "role": "activation",    "shape": "[B, A, Dh, S]"},
            {"name": "ones_mask_2",    "role": "plaintext_mask", "shape": "[B, A, Dh, S]"},
        ],
        "current_noise": None,
        "notes": "BLB BSGS 转置/重排第 2 步；与第 1 步噪声参数独立",
    },
    {
        "id": "attn.qk_matmul",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 torch.matmul)",
        "op_type": "activation_mm",
        "blb_op": "matmulcc (Q · K^T)",
        "operands": [
            {"name": "Q",  "role": "activation", "shape": "[B, A, S, Dh]"},
            {"name": "Kt", "role": "activation", "shape": "[B, A, Dh, S]"},
        ],
        "current_noise": None,
        "notes": "BLB Block 2 末端：preprocess QK^T；Q 操作数来自 attn.q_proj 输出（经 BSGS mask 之后）",
    },
    {
        "id": "attn.qkt.merge_mask",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (BertSelfAttentionWithAproximation.forward 内 _block2_qkt_merge_hook)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (Q·K^T ⊙ ones-mask)",
        "operands": [
            {"name": "qkt_result", "role": "activation",    "shape": "[B, A, S, S]"},
            {"name": "ones_mask",  "role": "plaintext_mask", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "Q·K^T 之后的 \"合并 Q,K\" 步：matmul 结果上加 rescale + 一次 ones-mask ewmulcp + 结果 rescale",
    },
    {
        "id": "attn.qk_scale_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 2,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (attention_scores / sqrt(Dh))",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/√Dh)",
        "operands": [
            {"name": "attention_scores", "role": "activation",   "shape": "[B, A, S, S]"},
            {"name": "1/sqrt(Dh)",       "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "标量除法等价 × (1/√Dh)；在我们项目里在 softmax 之前显式做",
    },

    # =====================================================================
    # BLB Block 3: Softmax exp 近似 (degree-aware)
    # softmax2 → N=8192；softmax 3/4/5/6 → N=16384
    # =====================================================================
    {
        "id": "attn.softmax.scalar_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": None,        # softmax2 用 8192；其它用 16384，看 degree
        "degrees": (2, 3, 4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "scalar_mul",
        "blb_op": "ctpt (1 + x/(2^n))",
        "operands": [
            {"name": "x_shifted", "role": "activation",   "shape": "[B, A, S, S]"},
            {"name": "1/(2^n)",   "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "exp 近似的初始 ct*pt：(1 + x/2^degree)；plaintext 1/(2^n) 在 scale 15",
    },
    # exp 近似的 6 个自乘步骤（每个步骤的 degree 激活条件不同）
    {
        "id": "attn.softmax.power.s1",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": None,
        "degrees": (2, 3, 4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (self-square)",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 1 次平方：y → y^2；degree ≥ 2 时生效",
    },
    {
        "id": "attn.softmax.power.s2",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": None,
        "degrees": (2, 3, 4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 2 次平方：y^(2^2)；degree ≥ 2 时生效",
    },
    {
        "id": "attn.softmax.power.s3",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": 16384,
        "degrees": (3, 4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 3 次平方：y^(2^3)；degree ≥ 3 时生效",
    },
    {
        "id": "attn.softmax.power.s4",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": 16384,
        "degrees": (4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 4 次平方：y^(2^4)；degree ≥ 4 时生效",
    },
    {
        "id": "attn.softmax.power.s5",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": 16384,
        "degrees": (5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 5 次平方：y^(2^5)；degree ≥ 5 时生效",
    },
    {
        "id": "attn.softmax.power.s6",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": 16384,
        "degrees": (6,),
        "shared_with": [],
        "module_path": "attention.self.approximation_exponential",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "y_prev", "role": "activation", "shape": "[B, A, S, S]"},
        ],
        "current_noise": None,
        "notes": "exp 近似第 6 次平方：y^(2^6) ≈ exp(x)；仅 degree = 6 时生效",
    },
    {
        "id": "attn.softmax.norm_div",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 3,
        "blb_N": None,
        "degrees": (2, 3, 4, 5, 6),
        "shared_with": [],
        "module_path": "attention.self.approximation_softmax",
        "op_type": "stat_div",
        "blb_op": "rec + smulcc (exp_out / sum_exp)",
        "operands": [
            {"name": "exp_out",          "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "1/(sum_exp+1e-9)", "role": "statistic",  "shape": "[B, A, S, 1]"},
        ],
        "current_noise": None,
        "notes": "softmax 收尾：rec 求倒数（非线性，走 MPC）后乘以 exp_out",
    },

    # =====================================================================
    # BLB Block 4: probs · V + Wo + post-attn LN head
    # 用 N=16384 表
    # =====================================================================
    {
        "id": "attn.head_mask_mul",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (probs * head_mask)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp",
        "operands": [
            {"name": "attention_probs", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "head_mask",       "role": "mask",       "shape": "[A] (broadcast)"},
        ],
        "current_noise": None,
        "notes": "默认 head_mask=None 不触发；BERT GLUE 不传",
    },
    {
        "id": "attn.probs_v_matmul",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.self  (torch.matmul(probs, V))",
        "op_type": "activation_mm",
        "blb_op": "matmulcc (CTCT_MUL: rot_softmax · V) — 主链 + other ct",
        "operands": [
            {"name": "rot_softmax", "role": "activation", "shape": "[B, A, S, S]"},
            {"name": "V",           "role": "activation", "shape": "[B, A, S, Dh]"},
        ],
        "current_noise": "softmax_probs / value_after_softmax",
        "notes": (
            "BLB 里这是特殊的 ctct_rot_softmax_mul_v：rot_softmax 是主链 ct，"
            "V 在 rescale skeleton 里体现为 other_ct_scale_bits。"
            "可独立给主链(rot_softmax)和 V 选 scaling factor，"
            "对应 'softmax_probs' / 'value_after_softmax' 两个噪声键。"
        ),
    },
    {
        "id": "attn.o_proj",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.dense",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (context · W_o)",
        "operands": [
            {"name": "context", "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_o",     "role": "weight",     "shape": "[H, H]"},
            {"name": "b_o",     "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": "wo",
        "notes": "BLB Block 4 中段：AttnOut",
    },
    {
        "id": "attn.layernorm.head.mean_smul",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_x", "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",   "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "post-attn LN head：mean 计算",
    },
    {
        "id": "attn.layernorm.head.center_ctpt",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× square mask)",
        "operands": [
            {"name": "x",           "role": "activation", "shape": "[B, S, H]"},
            {"name": "square_mask", "role": "mask",       "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "post-attn LN head：BLB ct*pt with square mask",
    },
    {
        "id": "attn.layernorm.head.square_ctct",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
        ],
        "current_noise": None,
        "notes": "post-attn LN head：(x − μ)² 自乘",
    },
    {
        "id": "attn.layernorm.head.var_smul",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 4,
        "blb_N": 16384,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "scalar_mul",
        "blb_op": "smulcp (× 1/D)",
        "operands": [
            {"name": "sum_xc2", "role": "activation",   "shape": "[B, S, 1]"},
            {"name": "1/D",     "role": "scalar_const", "shape": "scalar"},
        ],
        "current_noise": None,
        "notes": "Block 4 末步：variance；之后 rsqrt（Block 4 / Block 5 边界）",
    },

    # =====================================================================
    # BLB Block 5: post-attn LN tail + Wffn1 + GELU 多项式
    # gelu1 → N=8192；gelu2 / gelu4 → N=16384
    # =====================================================================
    {
        "id": "attn.layernorm.tail.normalize_ctct",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": None,        # gelu1=8192；gelu2/4=16384
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (× 1/std)",
        "operands": [
            {"name": "x_centered", "role": "activation", "shape": "[B, S, H]"},
            {"name": "1/std",      "role": "activation", "shape": "[B, S, 1]"},
        ],
        "current_noise": None,
        "notes": "post-attn LN tail 第 1 步：rsqrt 之后",
    },
    {
        "id": "attn.layernorm.tail.scale_ctpt",
        "stage": "encoder.attn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "attention.output.LayerNorm",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcp (× γ)",
        "operands": [
            {"name": "normalized", "role": "activation",  "shape": "[B, S, H]"},
            {"name": "gamma_attn", "role": "param_scale", "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "post-attn LN tail 第 2 步：× γ",
    },
    {
        "id": "ffn.intermediate_proj",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "intermediate.dense",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (X · W_ffn1)",
        "operands": [
            {"name": "X",       "role": "activation", "shape": "[B, S, H]"},
            {"name": "W_ffn1",  "role": "weight",     "shape": "[I, H]"},
            {"name": "b_ffn1",  "role": "bias",       "shape": "[I]"},
        ],
        "current_noise": "wffn1",
        "notes": "FFN 第 1 个 Linear；I = intermediate_size",
    },
    {
        "id": "ffn.gelu.power.x2",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": 16384,
        "degrees": (2, 4),
        "shared_with": ["ffn.gelu.power.x3x4"],   # gelu4 时与 x3x4 共享
        "module_path": "intermediate.intermediate_act_fn (PolynomialGELU)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (x · x = x²)",
        "operands": [
            {"name": "x", "role": "activation", "shape": "[B, S, I]"},
            {"name": "x", "role": "activation", "shape": "[B, S, I]"},
        ],
        "current_noise": None,
        "notes": "GELU 多项式：x²；degree ∈ {2, 4} 时生效（degree=1 跳过此步）",
    },
    {
        "id": "ffn.gelu.power.x3x4",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": 16384,
        "degrees": (4,),
        "shared_with": ["ffn.gelu.power.x2"],     # gelu4 内两步共享
        "module_path": "intermediate.intermediate_act_fn (PolynomialGELU)",
        "op_type": "elementwise_mul",
        "blb_op": "ewmulcc (x²·x = x³ 与 x²·x² = x⁴；并行)",
        "operands": [
            {"name": "x²", "role": "activation", "shape": "[B, S, I]"},
            {"name": "x²", "role": "activation", "shape": "[B, S, I]"},
        ],
        "current_noise": None,
        "notes": (
            "gelu4 用一个 ct*ct 节点同时算 x³ 和 x⁴；"
            "BLB 约束：与 ffn.gelu.power.x2 共享 scaling factor"
        ),
    },
    {
        "id": "ffn.gelu.coeff_mul",
        "stage": "encoder.ffn",
        "per_layer": True,
        "blb_block": 5,
        "blb_N": None,        # gelu1=8192；gelu2/4=16384
        "degrees": (1, 2, 4),
        "shared_with": [],
        "module_path": "intermediate.intermediate_act_fn (PolynomialGELU)",
        "op_type": "scalar_mul",
        "blb_op": "ewmulcp (Σ coeff_i · x^i)",
        "operands": [
            {"name": "powers (x^0..x^d)", "role": "activation",   "shape": "[B, S, I, d+1]"},
            {"name": "GELU_COEEF[d]",     "role": "scalar_const", "shape": "[d+1]"},
        ],
        "current_noise": None,
        "notes": (
            "GELU 多项式末步 ct*pt：把各次幂乘以系数后求和。"
            "degree 1 → 2 个系数 (a, b)；degree 2 → 3 个 (a/b/c)；"
            "degree 4 → 5 个 (a/b/c/d/e)。"
        ),
    },

    # =====================================================================
    # Stage 3 / 4: Pooler + Classifier (一次性)
    # =====================================================================
    {
        "id": "pooler.dense",
        "stage": "pooler",
        "per_layer": False,
        "blb_block": "pooler",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "bert.pooler.dense",
        "op_type": "linear_mm",
        "blb_op": "matmulcp ([CLS] · W_pool)",
        "operands": [
            {"name": "hidden_states[:, 0]", "role": "activation", "shape": "[B, H]"},
            {"name": "W_pool",              "role": "weight",     "shape": "[H, H]"},
            {"name": "b_pool",              "role": "bias",       "shape": "[H]"},
        ],
        "current_noise": None,
        "notes": "Pooler；之后 Tanh（非线性，无内置乘法登记）",
    },
    {
        "id": "head.classifier",
        "stage": "head",
        "per_layer": False,
        "blb_block": "head",
        "blb_N": None,
        "degrees": None,
        "shared_with": [],
        "module_path": "classifier",
        "op_type": "linear_mm",
        "blb_op": "matmulcp (pooled · W_clf)",
        "operands": [
            {"name": "pooled_dropout", "role": "activation", "shape": "[B, H]"},
            {"name": "W_clf",          "role": "weight",     "shape": "[num_labels, H]"},
            {"name": "b_clf",          "role": "bias",       "shape": "[num_labels]"},
        ],
        "current_noise": None,
        "notes": "MRPC 时 num_labels = 2",
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
    blb_block=None,                           # 1..5 / "embeddings" / "pooler" / "head"
    blb_N: Optional[int] = None,              # 8192 / 16384
    degree: Optional[int] = None,             # 当前 degree (gelu/softmax)
):
    """按条件筛选注册表条目。所有参数互相 AND。

    例子：
        select(blb_block=2)                          # Block 2 全部
        select(blb_block=3, degree=4)                # softmax degree=4 时 Block 3 激活的所有 id
        select(blb_block=5, degree=4)                # gelu4 时 Block 5 激活的所有 id
        select(has_noise=False, op_type="linear_mm") # 还没加噪的所有 Linear
        select(operand_role="activation", per_layer=True)
        select(ids=["attn.qk_matmul", "attn.probs_v_matmul"])
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
        if blb_block is not None and t["blb_block"] != blb_block:
            continue
        if blb_N is not None and t["blb_N"] != blb_N:
            continue
        if degree is not None:
            ds = t.get("degrees")
            if ds is not None and degree not in ds:
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


def shared_groups() -> list:
    """返回所有"必须共享 scaling factor"的 id 组（去重）。"""
    visited = set()
    groups = []
    for t in NOISE_TARGETS:
        sw = t.get("shared_with") or []
        if not sw:
            continue
        if t["id"] in visited:
            continue
        group = {t["id"], *sw}
        visited.update(group)
        groups.append(sorted(group))
    return groups


__all__ = [
    "OP_TYPES",
    "OPERAND_ROLES",
    "NOISE_TARGETS",
    "select",
    "get",
    "list_ids",
    "shared_groups",
]
