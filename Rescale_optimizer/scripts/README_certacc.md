# CertAcc 诊断脚本使用说明

`diagnose_certacc.py` 用来快速诊断当前 noise upper-bound 配置在 BERT-base/MRPC 上能否给出 **非空 certified set**，
对应 `accuracy_preservation_render.md` v2 (IBP / margin-based) framing。

## 公式

$$
\beta_{\text{sim}}(x) \;=\; L_{\text{input}}(x) \cdot \frac{c_\sigma \cdot \sigma_{\text{ct,total}}}{\Delta_{\text{working}}}
$$

其中:

- `σ_ct_total = √(num_stages) · √(N·h/12)` —— sub-Gaussian RMS 累加 (rescale + KS)
- `c_σ ≈ 6` —— sub-Gaussian → high-prob ℓ∞ (δ₀ ≈ 1e-9)
- `Δ_working = 2^delta_bits` —— 工作 scale, 把 ct 噪声转回 logit space
- `L_input(x) = ‖∂F/∂embedding‖_(2→∞)` —— input-level Jacobian, 用 random-direction finite-diff 采 4 次取最大

`CertAcc = #{x : pred(x)=y(x), γ(x,y) > 2·β_sim(x)} / N`，对应 §3 主定理。

## 依赖

```bash
pip install transformers datasets scikit-learn matplotlib
```

> 注意：默认从 HuggingFace Hub 下载 `textattack/bert-base-uncased-MRPC`，需要外网。
> 如果在沙箱里没有外网，可以提前下到本地，把 `--model` 指向本地路径。

## 用法

### 1. 默认 (200 样本快速诊断)

```bash
cd /var/tmp/root-home/Rescale_optimizer
python scripts/diagnose_certacc.py
```

输出会打印:

- `plaintext accuracy` —— 明文 baseline (BERT-MRPC 应该 ~84%)
- `CertAcc` —— 当前 β_sim 下的 certified accuracy
- `median(ratio) = γ / (2β)` —— 关键诊断量
  - **≥ 100**: ✅ STRONG, β 远小于 γ, v2 framing 完美
  - **[1, 100]**: ⚠️ MARGINAL, 同量级, 需要精细化 β
  - **< 1**: ❌ WEAK, β 过保守, CertAcc 接近 0; 切回 v1 sub-Gaussian 或 X2 framing

### 2. 不同 noise budget 扫描

```bash
# A. 当前优化器选出的最优配置 (假设 144 stages, Δ=2^30)
python scripts/diagnose_certacc.py --num-stages 144 --delta-bits 30

# B. 更大 Δ (例如优化器倾向于把 working scale 推到 2^40)
python scripts/diagnose_certacc.py --num-stages 144 --delta-bits 40

# C. 极端保守 (假设 num_stages=500, Δ=2^25)
python scripts/diagnose_certacc.py --num-stages 500 --delta-bits 25
```

期望: A/B 应该 ratio≥1, C 大概率 ratio<1 (CertAcc=0)。

### 3. 全验证集 + 更精细 Jacobian

```bash
python scripts/diagnose_certacc.py \
    --max-samples -1 \
    --num-jacobian-dirs 16 \
    --output-dir ./diagnose_certacc_full
```

### 4. 其他 GLUE 任务

```bash
python scripts/diagnose_certacc.py --task rte  --model textattack/bert-base-uncased-RTE
python scripts/diagnose_certacc.py --task wnli --model textattack/bert-base-uncased-WNLI
python scripts/diagnose_certacc.py --task sst2 --model textattack/bert-base-uncased-SST-2
```

## 输出文件

- `diagnose_certacc_output/diagnose_mrpc_validation.json` —— 完整记录
  - 全局: `plain_accuracy`, `cert_accuracy`, `median_ratio`, `verdict`
  - 每样本: `idx, label, pred, gamma, L_input, beta_sim, ratio, certified`
- `diagnose_certacc_output/diagnose_mrpc_validation.png` —— 直方图
  - 左: γ vs 2β 分布对比
  - 右: log10(ratio) 直方图

## 调参指南 (出现 verdict = WEAK 时)

按优先级 tighten β_sim 的来源:

1. **`--c-sigma 3.0`**: 把 sub-Gaussian tail 放宽 (1e-3 confidence) —— β 缩 2x
2. **`--num-stages` 写实际值**: 不要全用最坏路径, 按计算图 critical path 数。
3. **per-layer 累加 (TODO)**: 当前 `√N` RMS 累加假设各层独立; 实际 transformer 残差结构会让 L_layer 集中在前几层, 用 layer-wise Jacobian 可降一个量级。
4. **`--delta-bits` 调大**: 但这是优化器输出, 不能随便改。
5. 若以上都救不回来 → v2 framing vacuous, 在论文里就只用 v1 (sub-Gaussian Lipschitz) 作为 motivation, CertAcc 章节降级为 "robustness analogy" 而非 hard certificate。

## 与论文 framing 的对接

- **如果 CertAcc 在主优化器给出的配置上 > 50%**:
  → 在论文里写 "our optimizer's noise upper bounds yield non-trivial certified accuracy",
  把这张表/图放进 §6 实验部分, 作为 motivation theorem 的实证支撑。
- **如果 CertAcc 在主配置上 ≈ 0 但 ratio 集中在 0.1~1**:
  → 论文里只放 v1 (sub-Gaussian) framing, CertAcc 不当主指标; 仍然可以放 ratio histogram
  说明 "noise vs. margin 处于同量级, 实证密文准确率验证是必要的"。
- **如果 ratio ≪ 0.01**:
  → β 估计太松, 不要在论文里放; 先 tighten (上面调参指南), 再决定。
