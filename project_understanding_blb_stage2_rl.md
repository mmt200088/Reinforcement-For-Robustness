# Codex 项目理解备忘录：BLB Stage-2 RL 如何为 CKKS scale 和 MPC truncation 搜索最优配置

这份文档是给 Codex 的项目理解文件。它不是单次执行命令，也不是普通代码说明，而是接下来远程连接服务器、继续优化、跑训练、排查结果时必须始终遵守的“项目心智模型”。

最重要的一句话：

**本项目使用强化学习，在 BLB 五个 fused block 的固定计算图中，为每一个必须发生的 CKKS scale 位置选择 scaling factor；同时为 block 末尾的 MPC/CKKS 转换模拟选择 truncation bit。RL 输出的是动作索引 action index，动作索引再解码成真实 scaling factor 或 truncation k。**

项目当前不是真正执行密文推理。它是在明文 Transformer 上模拟 BLB 私有推理：模型仍在 PyTorch 里 forward，但在理论上会发生 CKKS encode、fresh encryption、rescale、rotation、CKKS/MPC conversion、MPC truncation 的位置，根据 RL 动作选择插入模拟噪声或截断，再用 probe/final eval 检查精度、稳定性和开销。

---

## 1. 绝对不能误解的核心事实

### 1.1 RL 不是在选择“操作做不做”

BLB 五个 block 中的 CKKS scale 相关位置都是固定计算图中的必须操作。RL 不能删除这些位置，不能跳过这些位置，不能把某个必须操作改成“不发生”。

RL 只能做这件事：

```text
对每个必须存在的 slot_j，选择一个离散动作索引 a_j。
再把 a_j 解码成 scaling_factor_j 或 truncation_k_j。
```

所以，如果代码中出现 mask，它只能是“index mask”，意思是暂时限制某个槽位允许选择哪些动作索引。例如一个槽位有 5 档，训练早期可以只允许 index 3 和 4，后期再开放 0 到 4。mask 绝不能被理解成 operation mask，不能表示“这个操作不用做”。

### 1.2 RL action 是索引，不是 scale 本身

策略网络输出的是整数动作索引：

```text
a = [a_0, a_1, ..., a_{D-1}]

对每个 slot_j：
    a_j in {0, 1, ..., m_j - 1}
```

之后由解码规则得到真实值：

```text
如果 slot_j 是 CKKS scaling factor 槽位：
    sf_j = DecodeSF_j(a_j)

如果 slot_j 是 truncation 槽位：
    k_j = DecodeK_j(a_j)
```

只有解码后的 `sf_j` 或 `k_j` 才进入 BLB cfg、Rescale_optimizer、明文噪声模拟和最终评估。

不要在代码中把动作索引直接当成 scaling factor 使用，也不要在日志中只保存 action index 而不保存解码后的真实 scale。正确的候选记录必须同时保存：

```text
action_index
slot_id
slot_name
kind
num_levels
level_values / decode rule
decoded_value
N
distribution
block
operation
```

### 1.3 当前优化目标不是 Stage-1，也不是 legacy Stage-2

当前要优化的是 **BLB Stage-2 RL**。

Stage-1 搜索每层 GELU degree 和 Softmax degree。它可以存在于项目中，BLB Stage-2 需要使用 Stage-1 固定好的 degree，但当前优化主目标不是 Stage-1。

`noise_rl_module_v2.py` 是先代 Stage-2 RL 的前身/参考实现，不是当前实际项目要优化的主路径。它的算法经验可以参考，例如 MC repeated evaluation、层级奖励、challenger confirmation、warmstart、entropy schedule、robust advantage normalization，但不要把它当作当前项目中的第二个实际阶段去优化。

如果 repo 里仍有 `legacy_v2`、`noise_rl_module_v2` 入口，可以保留用于对照实验，但当前远程训练和最终优化目标必须聚焦 BLB Stage-2。

### 1.4 项目不是在跑真实密文推理

真实 BLB 论文中，线性 operator 用 CKKS HE，非线性 operator 用 MPC，并设计 CKKS/MPC 安全转换。这个项目没有真正执行 CKKS ciphertext arithmetic，也没有真正运行两方 MPC。

本项目的模拟路径是：

```text
原始 Transformer 明文模型
    -> 安装 Stage-1 多项式近似
    -> 根据 BLB Stage-2 action 安装噪声/截断 wrapper
    -> 明文 forward
    -> 在指定位置加入 Gaussian 噪声或 truncation
    -> 用 probe/final eval 评估精度和稳定性
    -> 用 Rescale_optimizer 估算模数链合法性和开销
```

因此本项目产出的配置是“明文模拟环境下搜索到的 CKKS scale schedule / truncation schedule”。它用于指导和优化 BLB 风格私有推理配置，但它本身不是密码学安全实现。

---

## 2. 项目与 BLB baseline 论文的关系

### 2.1 baseline 论文解决了什么问题

BLB 论文的背景是 hybrid HE/MPC 私有 Transformer 推理。典型方式是：

```text
线性层 / 线性 operator：用 HE 计算
非线性层 / 非线性 operator：用 MPC 计算
```

问题在于，传统 layer-wise 方式会频繁在 HE 和 MPC 之间转换，并且固定点乘法后需要大量 truncation。论文指出 conversion 和 truncation 是通信开销的主要来源。BLB 的核心思路是 breaking the layer barrier：不再按粗粒度 layer 评估，而是把 Transformer layer 拆成更细粒度 operator，并把相邻线性 operator 融合成 fused blocks，以减少 HE/MPC conversion 和 truncation。

### 2.2 BLB 的五个 block 是本项目 Stage-2 的结构基础

BLB 论文 Figure 10 把一个 Transformer layer 拆成 5 个 fused linear operator blocks。你们项目的 BLB Stage-2 RL 就是在这 5 个 block 的固定结构上，为每个必须发生的 CKKS scale 点选择 scaling factor，并为 block 末尾转换模拟选择 truncation bit。

不要把五个 block 理解成 RL 可以选择是否存在的模块。它们是 baseline BLB 计算图的一部分。RL 只选择这些 block 内部各个 scale 点的数值配置。

### 2.3 本项目相对于论文 baseline 的优化点

论文 baseline 给出了一套 BLB framework 和固定/编译出的 CKKS 参数，用来证明 BLB 可以减少通信和延迟。

本项目进一步要做的是：

```text
在明文模拟环境中，自动搜索 BLB 五个 block 内部更细粒度的 scaling factor / truncation 配置。
目标是：
    先保证模型精度；
    再保证模拟噪声稳定性；
    最后尽量降低模数链和 MPC 通信相关开销。
```

也就是说，论文 baseline 是结构和协议基础；本项目的 RL 是在这套结构上做 per-slot scale schedule 搜索。

---

## 3. CKKS scale、rescale、rotation、truncation 的项目语义

### 3.1 scaling factor 表示 CKKS 中的 scale bits

在 CKKS 中，实数通常按 scale 编码。若 scale 为 2 的 s 次方，代码里就常把 `s` 称为 scaling factor 或 scale bits。

在本项目模拟中，`scaling_factor = s` 的主要作用是：

```text
1. 表示该 CKKS 操作点的 scale bits。
2. 用来查噪声方差表。
3. 影响 Rescale_optimizer 对模数链、total_bits、fusion、validity 的判断。
4. 间接影响模型精度和稳定性，因为 scale 越小，模拟噪声通常越大。
```

### 3.2 当前 scaling factor 的链式变化

可以按“当前 scale”来理解每个 block 内的 CKKS scale chain。

例如：

```text
a 做 fresh，选择 scaling factor 30。
b 做 encode，选择 scaling factor 25。
a * b 之后，当前 scale 近似累加为 55。

之后做 rescale，rescale scaling factor 选择 35。
rescale 后当前 scale 被拉回 35。

再乘一个 encode scaling factor 40 的 plaintext。
乘完当前 scale 变成 75。

再做 rescale，rescale scaling factor 选择 20。
rescale 后当前 scale 被拉回 20。
```

RL 的核心作用就是操纵这条 scale chain 上所有必须位置的 scale bits，让模数链既合法又尽量便宜，同时不让模拟噪声破坏模型。

### 3.3 encode 和 fresh 的区别

在项目语义中：

```text
encode scaling factor：
    通常对应 plaintext operand 或模型参数/常量被编码进 CKKS 计算时的 scale。
    例如权重、mask、scalar、GELU 系数、LayerNorm 的 1/D 等。

fresh scaling factor：
    通常对应新进入 CKKS block 的 ciphertext/中间结果的 fresh 噪声 scale。
    例如 block 输入、Softmax 输出 fresh、V fresh、first-input fresh 等。
```

二者都可能参与乘法并影响当前 scale，但它们对应的噪声 distribution 和方差表项不同，不能混用。

### 3.4 rescale 的 scaling factor 是目标 scale

rescale 槽位不是添加一个新的乘法 operand，而是选择“rescale 后当前 scale 应该变成多少”。

如果某次乘法前后 scale 过大，rescale 可以把 scale 拉回某个目标值。目标值越高，通常精度更安全但模数链/bit width 压力更大；目标值越低，可能省开销但噪声更大。

### 3.5 rotation 噪声不应该由 RL 单独自由选择 scale

rotation 是 CKKS 中对 ciphertext slot 做旋转的操作。项目理解中，rotation 噪声的 scale 应由执行 rotation 时的“当前 scaling factor”决定，而不是由 RL 额外选择一个独立 scaling factor。

特别要注意：

```text
如果一个 rotation 紧跟在某个 rescale 后面，
那么该 rotation 噪声应绑定到 rescale 后的当前 scale。

如果前面那个 rescale 没有被 Rescale_optimizer 选择/执行，
那么紧跟它的 rotation 噪声也不应该被添加。
```

这部分逻辑应由 Rescale_optimizer 的 effective rotations 或可调用信息提供。Codex 不要发明独立 rotation action。

### 3.6 truncation bit 是 MPC/CKKS 转换模拟的精度/通信权衡

每个 block 末尾还有一个 truncation 槽位。它不是 CKKS scaling factor，而是模拟 CKKS 和 MPC 转换或 MPC 固定点处理时保留多少小数 bit。

若保留 k 个二进制小数位，明文模拟可以理解为：

```text
Trunc_k(x) = trunc(x * 2^k) / 2^k
```

k 越大，保留精度越高，模型扰动越小，但 MPC 通信/位宽相关开销可能更高。k 越小，通信/位宽更省，但 truncation error 更大。

所以 BLB Stage-2 RL 同时在做两件事：

```text
1. 选择 scaling factor：优化 CKKS 模数链和噪声。
2. 选择 truncation k：优化 MPC/CKKS 转换模拟和 MPC 通信相关代价。
```

---

## 4. 当前 BLB Stage-2 的动作语义

### 4.1 动作向量

一个完整候选配置对应一个 action vector：

```text
action_vec = [a_0, a_1, ..., a_{D-1}]
```

每个 `a_j` 是一个动作索引。每个槽位可能有不同档位数：

```text
a_j in {0, 1, ..., m_j - 1}
```

完整动作空间是 MultiDiscrete：

```text
A = A_0 × A_1 × ... × A_{D-1}
```

### 4.2 当前“59 个槽位”的理解方式

用户确认的项目目标是：BLB Stage-2 有 59 个必须发生的 CKKS scale / truncation 相关槽位。Codex 必须按“59 个必选槽位”理解项目目标。

不过，某些已上传代码版本的 `action_space.py` 字段表可能导出每层 73 个字段，且文件注释里还可能保留旧的 94 维说法。因此不要直接相信注释，也不要直接相信某个旧字段表。必须使用最新仓库中的 registry/export 脚本确认：

```text
scripts/blb_export_action_registry.py
reports/blb_opt/phase1_registry/slot_registry_required59_or_mismatch.md
```

如果 registry 报告显示不是 59，Codex 的任务不是直接删除字段，而是分类：

```text
required slots：项目理论和用户定义的 59 个必选槽位。
effective extra slots：代码中实际影响 cfg/Rescale_optimizer/噪声模拟的额外槽位。
compat extra slots：旧 checkpoint 或旧实现兼容字段。
ineffective slots：存在于 action vector 但不影响最终 cfg 的槽位。
```

只有在确认某字段确实是兼容冗余或无效字段后，才可以提出代码整理方案。绝不能误删用户定义的 59 个必选槽位。

### 4.3 action index 到 scaling factor 的通用解码

若某个槽位是 scaling factor 类型，常见解码形式是：

```text
sf_from(idx, max_sf, levels) = max_sf - 2 * (levels - 1 - idx)
```

例子：

```text
max_sf = 30, levels = 5
idx 0 -> 22
idx 1 -> 24
idx 2 -> 26
idx 3 -> 28
idx 4 -> 30
```

动作索引越大，通常 scale 越高、噪声越小、代价越高。动作索引越小，scale 越低、噪声越大、代价越低。

但是，最终使用的 scale 还可能经过合法表对齐：

```text
sf_decoded = sf_from(...)
sf_actual = snap_to_noise_table(sf_decoded, N)
```

如果某个 `sf_decoded` 不在对应 N 的噪声表里，应选择合法表中小于等于它的最大值；如果没有更小合法值，则退到最小合法值。候选记录中必须保存解码前后两个值。

### 4.4 truncation k 的解码

K 槽位不是用 `sf_from`。它通过当前权威的 `K_LEVELS` 查表：

```text
K_LEVELS = (8, 9, 11, 13, 10, 12, 6, 7)
```

对应：

```text
idx 0 -> k 8
idx 1 -> k 9
idx 2 -> k 11
idx 3 -> k 13
idx 4 -> k 10
idx 5 -> k 12
idx 6 -> k 6
idx 7 -> k 7
```

索引 `0..5` 保留旧动作语义，新增索引 `6..7` 分别表示 K6 和 K7。这个顺序不是单调的；不要假设 index 越大 k 越大，也不要取表尾作为 baseline。all-max baseline 固定为 K13，对应 `idx 3`。

### 4.5 slot kind

当前动作槽位通常可以分为以下 kind：

```text
F：fresh scaling factor。
W：weight encode scaling factor。
M：mask encode scaling factor。
S：scalar encode scaling factor。
R：rescale target scaling factor。
K：block output truncation bit。
```

Codex 在导出 registry、写候选解释、写日志、做 sensitivity scan 时，都必须保留 kind 信息。不同 kind 的档位数、噪声 distribution、成本意义不同，不能混成普通整数。

---

## 5. 五个 BLB block 的项目级理解

下面是概念级理解。具体字段和 slot 顺序必须以最新 `scripts/blb_export_action_registry.py` 的导出为准。

### 5.1 Block 1

Block 1 主要覆盖 post-FFN / GELU output / Wffn2 / LayerNorm mean-variance 相关的 fused linear operators。

典型 scale 点包括：

```text
GELU 输出 fresh。
Wffn2 权重 encode。
LayerNorm mean 的 1/D scalar encode。
LayerNorm variance 的 1/D scalar encode。
Wffn2 结果 rescale。
mean 相关 rescale。
square/variance 相关 rescale。
block 末尾 truncation k。
```

RL 要选择这些位置的 scale/k，而不是选择 Block 1 是否存在。

### 5.2 Block 2

Block 2 主要覆盖 post-FFN LayerNorm tail、Wq/Wk/Wv 投影、QK 前后的 BSGS/mask/merge 相关操作。

典型 scale 点包括：

```text
LayerNorm inv_std fresh。
x_centered fresh。
gamma/mask/scalar encode。
Wq/Wk/Wv encode。
K/Q BSGS masks encode。
QK merge mask encode。
normalize/gamma/Wq/Wk/Wv/mask/QK matmul/merge 相关 rescale。
block 末尾 truncation k。
```

需要特别小心 Wq/Wk 的关系。如果 BLB/Rescale_optimizer 要求某些 Wq/Wk 或 mask scale 共享，Codex 不能随意破坏。若做约束，需要在 registry 中显式标注 tied group。

### 5.3 Block 3

Block 3 主要覆盖 Softmax 指数近似中的线性/乘法链。Softmax 近似常见形式是：

```text
exp(x) approx (1 + x / 2^n)^(2^n)
```

典型 scale 点包括：

```text
Softmax 输入 fresh。
1/(2^n) 或相关 scalar encode。
x * inv_2n 后 rescale。
多次 square 的 rescale chain。
block 末尾 truncation k。
```

如果 Softmax degree 较低，有些 square rescale 槽位可能不实际启用。Codex 必须用 registry/effective 标记确认，不要盲目优化 inactive slot。

### 5.4 Block 4

Block 4 主要覆盖 Softmax 输出、V、Softmax × V、Wo projection、post-attention LayerNorm head。

典型 scale 点包括：

```text
Softmax 输出 fresh。
V fresh。
Softmax/V/SoftmaxV masks encode。
Wo weight encode。
LayerNorm mean/variance scalar encode。
SoftmaxV matmul/mask/Wo/LayerNorm mean-square-var 相关 rescale。
block 末尾 truncation k。
```

### 5.5 Block 5

Block 5 主要覆盖 post-attention LayerNorm tail、Wffn1 projection、GELU 多项式链。

典型 scale 点包括：

```text
inv_std fresh。
x_centered fresh。
gamma encode。
Wffn1 encode。
GELU coefficient encode。
normalize/gamma/Wffn1 rescale。
GELU power chain rescale。
GELU coefficient multiplication rescale。
block 末尾 truncation k。
```

如果 GELU degree 较低，高阶 power rescale 或 coefficient rescale 可能不启用。Codex 必须按 fixed_gelu degree 和 registry/effective 标记处理。

### 5.6 first-input fresh

第 0 层输入来自 embedding，没有上一层 block 输出，所以项目中可能有 first-input fresh 槽位。它不是某个 block 内字段，而是 layer 0 输入处的 fresh 噪声入口。

如果 registry 中包含 first-input，必须把它作为完整 action 的一部分保存、评估和 final eval 安装。

---

## 6. 明文噪声模拟如何工作

### 6.1 NoisePoint

底层执行器通常把一个噪声点表示成：

```text
NoisePoint(distribution, scaling_factor, N)
```

含义：

```text
distribution：噪声类型，例如 fresh、encoding、rescale、rotation 等。
scaling_factor：该点的 scale bits。
N：CKKS polynomial degree 或噪声表维度索引。
```

模拟时会查噪声表：

```text
variance = NOISE_TABLE[N][scaling_factor][distribution]
noise = Gaussian(0, variance)
tensor = tensor + noise
```

Codex 不要把 `distribution` 忽略掉。相同 scaling_factor 在 fresh、encoding、rescale、rotation 下噪声方差可能不同。

### 6.2 明文 forward 中的噪声插入

模型 forward 仍然是明文 PyTorch 计算。被替换后的模块会在指定位置做：

```text
x_noisy = x + epsilon
```

或者对权重/中间结果使用临时噪声：

```text
W_noisy = W + epsilon_W
out = x @ W_noisy
```

这只是模拟真实 CKKS/MPC 过程的数值误差，不是加密。

### 6.3 truncation 模拟

在 block 输出处可能做：

```text
x_trunc = trunc(x * 2^k) / 2^k
```

这模拟 MPC 固定点截断或 CKKS/MPC 转换附近的精度损失。

### 6.4 噪声 RNG

底层噪声采样应使用独立 RNG，避免被外层 `torch.manual_seed` 完全固定。评估时如果需要复现，可以显式 reseed；如果要评估随机稳定性，则必须让多次 trial 采样独立噪声。

---

## 7. Rescale_optimizer 在项目中的作用

Rescale_optimizer 是 BLB Stage-2 的核心外部依赖。它不是普通 cost heuristic；
在 reward 里它只提供 optimizer 成本与 feasibility 诊断，不能替代或跳过真实模型
forward 得到的精度/稳定性信号。

### 7.1 输入

输入应是由 action 解码得到的完整 BLB cfg：

```text
Block1 cfg for each layer
Block2 cfg for each layer
Block3 cfg for each layer
Block4 cfg for each layer
Block5 cfg for each layer
first-input cfg if applicable
```

每个 cfg 中应包含所有必须 scale 点和 truncation 点。

### 7.2 输出

期望输出包括但不限于：

```text
valid / invalid
invalid_chain / message
total_bits
fusion_count
effective rotations
possibly modulus-chain details
```

### 7.3 RL 如何使用它

如果 Rescale_optimizer 判定 invalid，则该 action 不应进入模型 forward，直接给 invalid penalty。

如果 valid，则：

```text
1. 用 effective rotations 更新 cfg 中的 rotation 噪声开关。
2. 安装 BLB 噪声。
3. 跑 probe evaluation。
4. reward 中使用 total_bits、fusion_count、avg_k 等成本信号。
```

### 7.4 不要绕过真实 Rescale_optimizer

正式训练应使用真实 Rescale_optimizer。旧的 heuristic stub 可以用于本地 smoke test，但不能作为最终优化结果依据。服务器长周期训练、best 配置确认和 final eval 必须走真实 Rescale_optimizer。

---

## 8. BLB Stage-2 的 RL 环境流程

当前 BLB Stage-2 是单步 episode。

### 8.1 单步 episode 的意思

单步不是说只优化一个 layer。单步的意思是：策略网络一次性输出整个模型所有层、所有 BLB block、所有 scale/truncation 槽位的完整 action vector。

一个 episode 只有一次 `env.step(action_vec)`，但这一步内部会完成全部流程：

```text
1. reset/build state。
2. policy 采样完整 action_vec。
3. decode action_vec 到 BLB cfg。
4. 调 Rescale_optimizer。
5. 如果 invalid，直接给 penalty。
6. 如果 valid，安装 BLB 噪声和 truncation。
7. 在 probe batches 上跑多次 noise trial。
8. 聚合 loss/metric/std。
9. 计算 reward。
10. 清理噪声，还原模型。
11. done=True。
```

### 8.2 单步优势函数

因为 horizon 为 1，return 就是 reward：

```text
G = r
A = G - V(s) = r - V(s)
```

这不是漏掉 GAE，而是 GAE 在单步 episode 下的退化形式。

### 8.3 和先代 Stage-2 的区别

先代 Stage-2 是多步 episode，每一步按层选择 7 个 legacy noise scale。它需要 GAE 把终端奖励向前分配到每一步。

当前 BLB Stage-2 是单步 episode，一步输出完整配置，不需要标准多步 GAE。

先代 Stage-2 只可作为算法参考，不是当前主执行路径。

---

## 9. Reward 的真实优先级

当前目标不是单纯最大 cost saving。必须按硬优先级理解：

```text
Priority 0：Rescale_optimizer valid。
Priority 1：精度达标。
Priority 2：稳定性达标。
Priority 3：在都达标的候选中降低开销。
```

### 9.1 invalid

如果模数链 invalid、cfg apply 失败、模型 eval 失败，应直接进入 invalid penalty。不要让 cost reward 抵消 invalid。

### 9.2 精度

精度指标必须相对 baseline 或阈值达标。对分类任务通常是 accuracy / F1 / MCC 等，对回归任务可能是 Pearson/Spearman 或负 MSE。

只要精度不达标，候选不能被选为 final best，即使 cost 很低。

### 9.3 稳定性

稳定性来自多次独立噪声 trial。常见指标是：

```text
loss_std
metric_std
metric_min
loss_max
```

如果同一个 action 在不同噪声采样下波动很大，不应被当成稳定最优。

### 9.4 成本

只有精度达标、稳定性达标后，才比较成本。Rescale_optimizer 的
`any_invalid` 属于成本层的 optimizer feasibility 诊断，而不是精度/稳定性之前的
reward gate。

成本信号应来自真实 Rescale_optimizer 和 truncation k：

```text
total_bits_sum
fusion_count
avg_k / truncation cost
possibly rotation count or modulus-chain objective
```

如果后续加入更多成本项，必须保持硬优先级，不能让成本抵消精度/稳定性失败。

### 9.5 候选排序应使用字典序 rank key

建议候选排序用类似：

```text
rank_key = (
    accuracy_violation,
    stability_violation,
    optimizer_invalid_flag,
    normalized_cost,
    tie_breakers...
)
```

不要只用一个线性 reward 分数决定最终 best。PPO reward 可以用于训练，但 final best 应经过严格确认。

---

## 10. 策略网络的理解

### 10.1 MultiDiscrete actor

BLB Stage-2 的 actor 输出很多个 Categorical head。每个 slot 一个 head：

```text
slot_j logits -> Categorical distribution over action indices
```

完整动作概率是各槽位概率的乘积：

```text
pi(action_vec | state) = product_j pi_j(a_j | state)
log_prob(action_vec | state) = sum_j log_prob_j(a_j | state)
entropy(action_vec | state) = sum_j entropy_j
```

### 10.2 Critic

Critic 输出当前 state 的标量价值：

```text
V(s)
```

由于是单步 episode，critic 主要学习 reward baseline，用来降低 policy gradient 方差。

### 10.3 warmstart baseline bias

动作空间很大，从均匀随机开始几乎必然采到大量 invalid 或不稳定候选。策略初始化应偏向 all-max baseline：

```text
对每个 slot，把 all-max baseline 对应 action index 的 bias 加大。
```

这不是限制最终搜索，只是让冷启动从安全配置附近开始。

### 10.4 per-slot entropy 很重要

不要只看总 entropy。某些槽位可能已经坍缩，另一些仍在探索，总 entropy 可能掩盖问题。

建议日志和诊断中至少按组输出：

```text
F slots entropy
W slots entropy
M/S slots entropy
R slots entropy
K slots entropy
per block entropy
per layer entropy
critical slot entropy
```

---

## 11. 代码文件角色图

以下是 Codex 进入 repo 后应牢记的文件职责。不同 repo 版本可能文件略有差异，但语义应一致。

### 11.1 `function_handler.py`

底层模型改写与噪声模拟执行器。

职责：

```text
替换 GELU / Softmax 多项式近似。
安装 legacy noise wrapper。
安装 BLB block 噪声 wrapper。
安装 first-input fresh 噪声。
根据 NoisePoint 查表加 Gaussian 噪声。
执行 block output truncation。
支持 restore/clear，保证模型可逆还原。
```

Codex 改这个文件时必须极其谨慎。任何噪声注入位置或 scale 语义错误都会直接让 RL 学错目标。

### 11.2 `blb_stage2_rl/action_space.py`

动作空间定义、slot 字段表、action index 解码、cfg 构建。

职责：

```text
定义 slot kind、levels、K_LEVELS。
定义每个 block 的字段顺序。
把 action_vec 解码成 Block1-5 NoiseConfig。
生成 all-max baseline action。
生成动作解释信息。
```

Codex 在这里最容易犯错：不要把字段删成 59，除非已经通过 registry 和用户定义确认哪些字段确实应合并/无效。正确做法是先导出 registry 和 mismatch report。

### 11.3 `blb_stage2_rl/env.py`

单步 RL 环境。

职责：

```text
reset/build_state。
step(action_vec)。
调用 action_space 解码。
调用 Rescale_optimizer。
安装 BLB 噪声。
跑 probe 多 trial evaluation。
计算 metrics。
调用 reward。
清理模型。
```

如果训练结果大量 invalid、apply_error、eval_error，要优先看这里。

### 11.4 `blb_stage2_rl/reward.py`

三层优先级奖励。

职责：

```text
invalid penalty。
accuracy constraint penalty。
stability constraint penalty。
cost reward。
baseline stats。
reward weight calibration。
```

Codex 不要把 reward 改成简单线性组合，除非明确保留硬优先级。

### 11.5 `blb_stage2_rl/policy.py`

BLB Stage-2 actor-critic。

职责：

```text
state encoder。
per-layer shared head。
first-input head。
value head。
sample_action。
evaluate_action。
PPO update。
RolloutBuffer。
```

如果要加入 per-slot mask、per-slot entropy、BC warmstart，需要改这里。

### 11.6 `blb_stage2_rl/runner.py`

主训练入口。

职责：

```text
读取 evaluator 参数。
应用 fixed_gelu/fixed_softmax。
构造 probe。
构造 RescaleOptimizerBridge。
估计 all-max baseline。
preflight baseline。
训练 PPO。
保存 checkpoint/report/best cfg。
返回兼容旧版 final eval 的结果。
```

特别注意：final eval 不能静默只评估 legacy all-max baseline。真正的 BLB best 必须能被读取、解码、安装、评估。

### 11.7 `blb_stage2_rl/persistence.py`

状态板、曲线、报告、候选解释落盘。

职责：

```text
blb_stage2_status.json。
episode trace csv。
training curves。
best action markdown/json。
crash report。
```

Codex 应尽量增强报告，而不是只靠 stdout。

### 11.8 `layer_importance_evaluator.py`

统一调度器、数据、模型、评估和 Stage-1/Stage-2 路由。

职责：

```text
加载数据集。
持有模型和 tokenizer。
Stage-1 degree 搜索。
Stage-2 路由到 BLB runner。
提供 evaluate/final evaluation 支撑。
管理 run_output_dir 和 metadata。
```

Codex 不要把这个文件里的 legacy 路径和当前 BLB 路径混淆。

### 11.9 `noise_rl_module_v2.py`

先代 Stage-2 RL 参考实现。

职责：

```text
逐层选择 legacy 7 类 noise scaling factor。
使用 GTrXL。
中间 reward 稀疏，终端 MC evaluation。
层级 reward。
challenger confirmation。
warmstart/entropy/robust advantage 技巧。
```

当前优化可借鉴它的训练技巧，但不要把它当作当前 BLB Stage-2 实际目标。

### 11.10 最近新增的旁路工具

Codex 上一轮已经新增：

```text
scripts/blb_phase0_preflight.py
scripts/blb_export_action_registry.py
blb_stage2_rl/candidate_store.py
scripts/blb_eval_action.py
```

这些工具的定位是：

```text
preflight：确认入口、文件、环境、数据、关键路径。
action registry：导出完整/effective slot，确认 required 59 与代码实现是否一致。
candidate_store：保存候选 action、hash、metrics、rank key、fidelity 记录。
blb_eval_action：离线评估某个 action 候选，支持 optimizer-only 或更高 fidelity。
```

这些是优化工作流的旁路工具，不应破坏原 launcher。

---

## 12. 远程服务器运行前 Codex 应检查什么

### 12.1 数据集加载

MRPC/GLUE 可能因为网络不稳加载失败。当前 `rl_tune.py` 已支持：

```text
GLUE_LOCAL_DATASET_DIR
GLUE_DATASET_DIR
DatasetDict.save_to_disk(...) 本地目录
本地 parquet split
DownloadConfig(local_files_only=True) HF 缓存 fallback
```

服务器上如果远端 `nyu-mll/glue` 和 parquet mirror 不稳定，先准备本地 GLUE 数据，再运行：

```bash
export GLUE_LOCAL_DATASET_DIR=/path/to/local_glue
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl
```

### 12.2 action registry

必须先运行并阅读：

```bash
python scripts/blb_export_action_registry.py ...
```

重点确认：

```text
required slots 是否为 59。
代码实际 action length 是多少。
哪些 slots 是 effective。
哪些 slots 是 compat extra。
哪些 slots 是 inactive due to degree。
first-input 是否计入。
K slots 是否计入。
rotation 是否独立存在 action，如果存在要警惕。
```

### 12.3 all-max baseline preflight

必须确认全 max baseline：

```text
Rescale_optimizer valid。
能够安装 BLB 噪声。
probe eval 通过。
精度不低于阈值。
稳定性不超过阈值。
```

如果 all-max baseline 都失败，先修环境/数据/模型/bridge/optimizer，不要开始长训练。

### 12.4 final eval 路径

必须确认最终评估不是误用 legacy all-max baseline。真正的 final eval 应该：

```text
读取 best action indices。
解码为 BLB cfg。
调用真实 Rescale_optimizer。
安装 BLB 噪声和 truncation。
跑最终评估。
报告 best BLB 配置的真实指标。
```

如果 runner 为了兼容旧模块返回了 legacy-compatible baseline noise config，不能把这个当成 BLB best。

---

## 13. 结果验证分层

Codex 后续执行时不要一次性把训练跑到底才看结果。应分 fidelity 验证。

### 13.1 F0：optimizer-only

只解码 action 并调用 Rescale_optimizer，不跑模型 forward。

用途：

```text
检查 action 是否能解码。
检查模数链是否合法。
快速收集 total_bits / fusion_count / invalid_chain。
适合 registry、sensitivity scan、候选过滤。
```

### 13.2 F1：small probe, low K

在小 probe 上跑少量噪声 trial。

用途：

```text
快速看精度是否明显崩。
快速筛掉明显不稳定候选。
给 greedy / CEM / PPO 提供 cheap signal。
```

### 13.3 F2：medium probe, higher K

中等 probe 和更多 trial。

用途：

```text
验证 F1 中看起来好的候选不是偶然。
检查 loss_std 和 metric_min。
准备进入 challenger confirmation。
```

### 13.4 F3：confirmation

大 probe，更多 K，多 seed。

用途：

```text
确认候选可以作为 incumbent。
防止 lucky noise trial 被选成 best。
```

### 13.5 F4：final evaluation

完整或接近完整验证集，真实 BLB 安装，固定报告。

用途：

```text
最终声明 best 配置。
生成论文/报告可用数据。
```

---

## 14. 先代 Stage-2 可迁移经验

先代 Stage-2 不再作为当前路径，但它的设计值得迁移到 BLB Stage-2。

### 14.1 MC repeated evaluation

同一个 action 要跑多次独立噪声 trial。不要用单次 trial 判断 best。

### 14.2 层级 reward

先 metric-ok，再 stability-ok，再 cost。不要让 cost reward 抵消硬约束失败。

### 14.3 challenger confirmation

候选超过 incumbent 后，不要立即替换。先用更大 probe / 更多 K / 多 seed confirmation。

### 14.4 warmstart baseline bias

大动作空间必须从安全 baseline 附近开始。all-max baseline bias 是合理的。

### 14.5 entropy schedule / per-slot entropy

不能让所有 slot 太早坍缩到单一选择。建议按 block/kind/slot 监控 entropy。

### 14.6 robust advantage normalization

如果 reward 分布有极端 invalid penalty 或 confirmation penalty，advantage 标准化前可做 outlier clipping，避免 PPO 更新被极端样本支配。

### 14.7 noise debt feature

可以为候选计算 expected noise debt：

```text
variance_debt(slot) roughly proportional to 2^(-2 * scaling_factor)
```

它可作为候选诊断、reward shaping 或搜索排序辅助，但不能替代真实 probe eval。

---

## 15. Codex 后续改代码时的禁止事项

1. 不要把 59 个必须槽位当成可删除操作。
2. 不要把 action index 当成真实 scaling factor。
3. 不要为 rotation 发明独立自由 scaling factor action。
4. 不要绕过真实 Rescale_optimizer 产出最终结果。
5. 不要只用单次噪声 trial 选 best。
6. 不要让 cost reward 覆盖精度/稳定性失败。
7. 不要静默把 final eval 回退成 legacy all-max baseline。
8. 不要相信旧注释中的动作维度，必须以 registry/export 为准。
9. 不要改动 launcher 入口导致原脚本不可运行；新增工具应旁路兼容。
10. 不要在训练结果未知时大改多个模块；每次改动后要用 F0/F1/F2 分层验证。

---

## 16. Codex 后续应主动补强的理解型报告

远程运行时，Codex 应持续生成和维护这些报告，让人不看代码也能判断训练是否正确：

```text
reports/phase0_entrypoints.md
reports/blb_opt/phase1_registry/slot_registry_required59_or_mismatch.md
reports/blb_opt/candidate_store_summary.md
reports/blb_opt/baseline_preflight.md
reports/blb_opt/sensitivity_single_slot.md
reports/blb_opt/search_progress.md
reports/blb_opt/ppo_diagnostics.md
reports/blb_opt/challenger_confirmation.md
reports/blb_opt/final_blb_eval.md
```

每个报告至少包含：

```text
运行命令。
使用的 git commit / timestamp。
关键环境变量。
数据集来源。
模型和 checkpoint。
slot count。
best action hash。
valid/invalid 数量。
metric mean/std/min/max。
cost breakdown。
是否通过硬约束。
下一步建议。
```

---

## 17. 当前项目的一句话验收标准

当 Codex 声称“项目已经优化完成”时，必须能证明：

```text
1. 已确认 59 个必须槽位和代码 registry 的对应关系。
2. 所有候选 action 都保存 action index 和 decoded scale/k。
3. all-max baseline 通过 Rescale_optimizer、probe 和稳定性 preflight。
4. 搜索过程没有删除必须 CKKS scale 操作。
5. best 候选经过多 fidelity、多 trial、confirmation。
6. final evaluation 真实安装了 BLB best action，不是 legacy all-max baseline。
7. best 配置满足：valid、精度达标、稳定性达标。
8. 在满足 7 的前提下，相比 baseline 降低了 total_bits / fusion / truncation cost 中的目标开销。
9. 报告中能逐 slot 解释 best action：哪个 block、哪个 operation、action index、decoded scaling factor/k、为什么它有效。
```

---

## 18. 给 Codex 的最终心智模型

你现在接手的不是一个普通神经网络训练项目，而是一个“用强化学习搜索 BLB 私有 Transformer 推理配置”的项目。

模型权重已经训练好。RL 不训练模型权重。

BLB 五个 block 的计算图已经确定。RL 不选择 block 是否存在。

每个 CKKS scale 位置都是必须操作。RL 不选择操作是否执行。

RL 做的是：

```text
在每个必须 scale slot 上选择一个 action index。
把 action index 解码成 CKKS scaling factor。
在每个 block 末尾选择 truncation k。
把完整配置交给 Rescale_optimizer 检查模数链和成本。
把配置安装到明文 Transformer 中做噪声/截断模拟。
用多 trial probe 评估精度和稳定性。
在 valid、精度、稳定性都满足后，尽量降低开销。
```

如果你后续做的任何代码修改偏离了这段话，就应停下来重新检查。
