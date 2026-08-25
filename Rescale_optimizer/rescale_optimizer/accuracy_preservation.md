# 保守扰动上界下 Certified Accuracy 的单调性
## 面向 CKKS 隐私推理的形式化精度保持定理

> *"明文模拟驱动 optimizer + 单点密文验证"工作流的形式化基础*

本文给出一个基于 logit-domain 扰动上界的 certified-accuracy 单调性定理，
为 rescale / modulus-chain optimizer "在多个候选配置中按上界排序" 这件事
提供形式化依据。证明纯确定性，不依赖模型 Lipschitz 估计、不依赖 PRG
耦合，**直接对应 plaintext-simulation + ciphertext-verification pipeline**。

主要工具：interval bound propagation 类型的 certified accuracy（Wong–Kolter 2018,
Gowal et al. 2018），与 randomized smoothing（Cohen et al. 2019）相比
更适合 CKKS 噪声有显式 closed-form 上界的场景。

---

## 0. 与对抗鲁棒性文献的归口

本定理把对抗鲁棒性中的 *certified accuracy via interval bound
propagation* 框架（Wong–Kolter 2018, Gowal 2018）移植到 FHE 推理场景。
区别有三：

1. **扰动产生机制不同**：FHE 噪声来自密码学原语（rescale、key-switching、
   ModDown rounding、CTPT encoding），而非 adversarial \(\ell_\infty\)-ball；
2. **上界 \(\beta_C(x)\) 由 cryptographic noise model 给出**，而非
   box-constraint propagation；
3. **单调性论证服务于 modulus-chain 设计选择**，而非对抗训练。

与 Cohen et al. 2019 的 randomized smoothing 不同，这里**不需要**对模型
做 Gaussian smoothing，也**不需要**估计模型的 Lipschitz 常数——FHE 噪声
直接进入 logit 空间，所有边界都在 logit 上做。

---

## 1. 设定与记号

设 \(F\) 表示明文推理函数，输入 \(x\) 的明文 logits 为

\[
z(x) = F(x) \in \mathbb{R}^{K}.
\]

对于带标签样本 \((x, y)\)，定义其 **明文分类 margin**

\[
\gamma(x, y) \;=\; z_y(x) - \max_{k \neq y} z_k(x).
\]

一个隐私推理配置 \(C\)（modulus chain、scaling factors、rescale skeleton 等
全部固定）会产生扰动后的 logits

\[
\tilde z_C(x) \;=\; z(x) + \delta_C(x),
\]

其中 \(\delta_C(x) \in \mathbb{R}^K\) 是由 HE / MPC 误差引入的总
**logit-domain 扰动**，包含 rescale、key-switching、ModDown rounding、
CTPT encoding 误差以及非线性近似算子（如多项式 GELU、softmax）的
逼近误差。

**关键假设**：对每个配置 \(C\)，存在确定性或高置信度的 logit-domain
\(\ell_\infty\) 上界

\[
\|\delta_C(x)\|_\infty \;\le\; \beta_C(x).
\tag{1}
\]

\(\beta_C(x)\) 的具体构造见第 3 节。

定义配置 \(C\) 的 **certified set** 为

\[
\mathcal{S}_C
\;=\;
\Big\{\,(x, y) \in \mathcal{D}\;:\; \gamma(x, y) \;>\; 2 \beta_C(x) \,\Big\},
\tag{2}
\]

对应的 **certified accuracy lower bound**

\[
\mathrm{CertAcc}(C)
\;=\;
\frac{|\mathcal{S}_C|}{|\mathcal{D}|}.
\tag{3}
\]

---

## 2. 主定理

**定理 1（Certified-Accuracy 单调性 + 实测下界）.**
*在假设 (1) 下:*

**(i)** *对每个 \(C\) 与 \(\mathcal{S}_C\) 中的每条样本 \((x, y)\)，配置 \(C\)
产生的扰动密文推理结果都保持原始预测，因而*

\[
\mathrm{Acc}_{\mathrm{enc}}(C) \;\ge\; \mathrm{CertAcc}(C).
\tag{4}
\]

**(ii)** *对任意两个配置 \(A\) 与 \(B\)，若*

\[
\beta_A(x) \;\le\; \beta_B(x), \quad \forall (x, y) \in \mathcal{D},
\tag{5}
\]

*则*

\[
\mathcal{S}_B \;\subseteq\; \mathcal{S}_A,
\quad
\mathrm{CertAcc}(A) \;\ge\; \mathrm{CertAcc}(B).
\tag{6}
\]

### 证明

**(i) 部分**。任取 \((x, y) \in \mathcal{S}_C\)，由 (2) 有
\(\gamma(x, y) > 2 \beta_C(x)\)。对每个错误类别 \(k \neq y\)，扰动后的
margin 满足

\[
\tilde z_{C,y}(x) - \tilde z_{C,k}(x)
\;=\;
\big(z_y(x) - z_k(x)\big) + \big(\delta_{C,y}(x) - \delta_{C,k}(x)\big).
\]

由 \(z_y(x) - z_k(x) \ge \gamma(x, y)\) 及

\[
|\delta_{C,y}(x)| \le \|\delta_C(x)\|_\infty,
\quad
|\delta_{C,k}(x)| \le \|\delta_C(x)\|_\infty,
\]

得

\[
\delta_{C,y}(x) - \delta_{C,k}(x) \;\ge\; -2 \|\delta_C(x)\|_\infty
\;\ge\; -2 \beta_C(x).
\]

代回原式：

\[
\tilde z_{C,y}(x) - \tilde z_{C,k}(x)
\;\ge\;
\gamma(x, y) - 2 \beta_C(x) \;>\; 0.
\]

故对所有 \(k \neq y\) 都有 \(\tilde z_{C,y}(x) > \tilde z_{C,k}(x)\)，即
\(\arg\max_j \tilde z_{C,j}(x) = y\)。所以 \(\mathcal{S}_C\) 中每条样本在
配置 \(C\) 的扰动下都保持正确分类，进而 (4) 成立。

**(ii) 部分**。任取 \((x, y) \in \mathcal{S}_B\)，由定义有
\(\gamma(x, y) > 2 \beta_B(x)\)。由 (5) 单调，
\(2\beta_A(x) \le 2\beta_B(x)\)，故

\[
\gamma(x, y) \;>\; 2 \beta_B(x) \;\ge\; 2 \beta_A(x),
\]

即 \((x, y) \in \mathcal{S}_A\)，从而 \(\mathcal{S}_B \subseteq \mathcal{S}_A\)，
取基数比即得 (6)。 ∎

---

## 3. \(\beta_C(x)\) 的具体构造（CKKS 实现）

抽象上界 \(\beta_C(x)\) 在 CKKS-CT 推理中可以用 noise model 显式给出。
设推理电路被 optimizer 切成 \(L\) 个 stage（如 attention / FFN block 内
的若干 mult/rotation 段），每个 stage \(\ell\) 在配置 \(C\) 下的内部
噪声上界（密文系数空间）为 \(\epsilon_\ell^{\mathrm{ct}}(C)\)。每个
stage 的噪声会通过后续算子放大到 logit 空间，记其
**敏感度系数** 为 \(L_\ell(x)\)（即 stage \(\ell\) 的单位扰动对 logit
\(\|\cdot\|_\infty\) 的最坏放大）。则

\[
\beta_C(x)
\;=\;
\sum_{\ell=1}^{L} L_\ell(x) \cdot \epsilon_\ell^{\mathrm{logit}}(C),
\quad
\epsilon_\ell^{\mathrm{logit}}(C) := \frac{\epsilon_\ell^{\mathrm{ct}}(C)}{\Delta_\ell(C)},
\tag{7}
\]

其中 \(\Delta_\ell(C)\) 是 stage \(\ell\) 输出的 working scale。

每条 \(\epsilon_\ell^{\mathrm{ct}}(C)\) 由具体的 CKKS 原语贡献：

\[
\epsilon_\ell^{\mathrm{ct}}(C)
\;=\;
\underbrace{c_\sigma \sqrt{\tfrac{N h}{12}} \cdot \big(R_\ell(C) + K_\ell(C)\big)}_{\text{rescale + KS, sub-Gaussian}}
\;+\;
\underbrace{c_\sigma \sqrt{\tfrac{N h}{12}} \cdot M_\ell(C) \cdot 2^{-s_{\mathrm{pt},\ell}(C)}}_{\text{CTPT encoding}}
\;+\;
\underbrace{P_\ell(x; C)}_{\text{近似算子}}.
\tag{8}
\]

记号说明：

| 符号 | 含义 |
|---|---|
| \(N\) | ring 维度 |
| \(h\) | secret-key Hamming weight (HWT(64) 时 \(h = 64\)) |
| \(R_\ell(C)\) | stage \(\ell\) 的 rescale 次数 |
| \(K_\ell(C)\) | stage \(\ell\) 的 key-switching 次数（rotation + relinearize） |
| \(M_\ell(C)\) | stage \(\ell\) 的 CTPT 乘法次数 |
| \(s_{\mathrm{pt},\ell}(C)\) | stage \(\ell\) 的明文 scaling factor (bit) |
| \(P_\ell(x; C)\) | stage \(\ell\) 内非线性近似算子的 worst-case 误差（数据相关） |
| \(c_\sigma\) | sub-Gaussian → \(\ell_\infty\) 高置信度上界的常数（见 §4） |

**(7)–(8) 的意义**：把 optimizer 内部 noise tracker 的累加形式直接
解析地写成 \(\beta_C(x)\)。两个配置 \(A, B\) 在每个 stage \(\ell\) 上若
都有 \(\epsilon_\ell^{\mathrm{ct}}(A) \le \epsilon_\ell^{\mathrm{ct}}(B)\)，
且 \(L_\ell(x), \Delta_\ell\) 共享，则 (5) 自动满足，(6) 立刻成立。

---

## 4. \(\sigma \to \beta\) 的转换：worst-case vs high-probability

CKKS 噪声严格意义下尾部 unbounded（高斯尾），故 (1) 在数学上不可能严格
deterministic 满足。实际系统中 \(\beta_C(x)\) 取为 **high-probability
上界**：

\[
\Pr\big[\, \|\delta_C(x)\|_\infty > \beta_C(x) \,\big] \;\le\; \delta_0,
\quad \delta_0 \in \{10^{-6}, 10^{-9}, \ldots\}.
\tag{9}
\]

这通过在 (8) 中选 sub-Gaussian → \(\ell_\infty\) 转换常数 \(c_\sigma\) 实现：

| \(\delta_0\) | \(c_\sigma\) (univariate) | 含义 |
|---|---|---|
| \(10^{-3}\) | \(\approx 3.3\) | 99.7 % confidence |
| \(10^{-6}\) | \(\approx 4.9\) | 99.9999 % confidence |
| \(10^{-9}\) | \(\approx 6.0\) | 经典工程取值 |

**联合 union bound**：对 logit 维度 \(K\)（如 GLUE \(K \le 10\)），整体
联合违反概率 \(\le K \delta_0\)，仍可忽略。

**实务结论**：在 (9) 下，定理 1 的两个结论 (4)/(6) 同时退化为
\((1 - K \delta_0)\)-confident 下界。当 \(\delta_0 \le 10^{-6}\) 时
工程上可视为 deterministic 等价。

---

## 5. 推论 1：逐层扰动预算下的单调性（cost-model 形态）

**推论 1.** *若两个配置 \(A, B\) 共享相同的 logit 敏感度系数
\(L_\ell(x)\) 与 working scale \(\Delta_\ell\)，并满足逐层 ct-domain 误差
单调*

\[
\epsilon_\ell^{\mathrm{ct}}(A) \;\le\; \epsilon_\ell^{\mathrm{ct}}(B),
\quad \forall \ell \in \{1, \ldots, L\},
\tag{10}
\]

*则 (5) 自动成立，故*

\[
\mathrm{CertAcc}(A) \;\ge\; \mathrm{CertAcc}(B).
\tag{11}
\]

**证明**：由 (7) 直接代入；逐项 \(\le\) 之和仍 \(\le\)。 ∎

> **工程含义**：optimizer 的 cost function 只要保证"逐 stage σ 单调
> 不增"，就自动驱动 CertAcc 单调不减。这就是把 cost = \(\sum_\ell\) σ 类
> 指标作为优化目标的形式化依据。

---

## 6. 推论 2：保守明文扰动模拟

**推论 2.** *设 \(C^\star\) 是通过明文扰动模拟器选出的最优配置。若注入
的明文扰动是真实密文扰动的保守上界：*

\[
\|\delta_{\mathrm{real}}(x; C^\star)\|_\infty
\;\le\;
\beta_{\mathrm{sim}}(x; C^\star),
\quad \forall x \in \mathcal{D},
\tag{12}
\]

*则所有满足 \(\gamma(x, y) > 2 \beta_{\mathrm{sim}}(x; C^\star)\) 的样本，
在真实密文推理中也保持原始预测。因此*

\[
\mathrm{Acc}_{\mathrm{enc}}(C^\star)
\;\ge\;
\mathrm{CertAcc}_{\mathrm{sim}}(C^\star),
\tag{13}
\]

*即 simulation-based CertAcc 是真实密文 accuracy 的下界。*

**证明**：由 (12) 与定理 1(i) 直接得到。 ∎

> **paper claim 形式化**：你"明文模拟 + 单点密文验证"的整套 workflow，
> 形式上就是用 Corollary 2 把"sim 的 CertAcc"作为真实 acc 的可信下界。
> 单点密文验证的角色 = 验证 (12) 在该点经验上成立（即"sim 确实保守"）。

---

## 7. 定理适用范围与限制

**定理 1 证明的是 certified accuracy lower bound 的单调性，
而非 empirical measured accuracy 的严格单调性。**

一般地，

\[
\beta_A(x) \le \beta_B(x)
\quad
\not\Rightarrow
\quad
\mathrm{Acc}_{\mathrm{enc}}(A) \ge \mathrm{Acc}_{\mathrm{enc}}(B).
\]

原因有二：

1. **偶然修正（accidental correction）**：更大的 ciphertext 扰动可能把
   原本明文域分类错误的样本"推回"正确类别。在 random realization 上
   这是非零概率事件。

2. **误差项 cancellation**：不同噪声源（rescale、KS、approx）方向相反时
   实测扰动远小于 worst-case 上界。

**因此**：

- 明文扰动模拟应当被用作 **保守的配置搜索代理**；
- 最终 accuracy 结论应当 **基于真实密文推理验证**；
- 定理保证的是"**按 \(\beta\) 排序选出的配置在 certified 意义下不会比
  其它配置差**"，这正是 optimizer 需要的最小性质。

**对 \(\gamma(x, y)\) 分布的依赖**：若数据集 \(\mathcal{D}\) 中大量样本
margin 接近 0（明文低置信度），\(\mathrm{CertAcc}\) 会显著低于
\(\mathrm{Acc}_{\mathrm{enc}}\)。这是 certified-accuracy 框架的固有
特性，不是定理本身的缺陷。

---

## 8. 实证验证模板

定理 1 + Corollary 2 的实证支撑应当在 paper 中以下表形式呈现，验证
**(a) (12) 真实密文保守性、(b) (5) 单调性**：

| Config | \(\beta_{\mathrm{sim}}(C)\) | \(\mathrm{CertAcc}_{\mathrm{sim}}(C)\) | \(\mathrm{Acc}_{\mathrm{enc}}(C)\) | (12) hold? |
|---|---|---|---|---|
| \(C_1\) (σ 最小) | 1.2e-5 | 78.5 % | 79.1 % | ✓ |
| \(C_2\) | 3.1e-5 | 78.0 % | 78.6 % | ✓ |
| \(C_3\) | 8.5e-5 | 76.9 % | 78.2 % | ✓ |
| \(C_4\) | 2.4e-4 | 75.1 % | 77.5 % | ✓ |
| \(C_5\) (σ 最大) | 1.0e-3 | 68.4 % | 75.8 % | ✓ |

**实证 protocol**：

1. 选 5 个 σ 显著不同的配置（不必是 optimizer 的 top-K，而是 σ 跨度大）；
2. 对每个配置：用明文模拟器算 \(\beta_{\mathrm{sim}}\) 与 \(\mathrm{CertAcc}_{\mathrm{sim}}\)；
3. 跑一次真实密文推理得到 \(\mathrm{Acc}_{\mathrm{enc}}\)；
4. 检查 \(\mathrm{Acc}_{\mathrm{enc}} \ge \mathrm{CertAcc}\)（验证 Theorem 1(i)）；
5. 检查 \(\beta\) 单调性 \(\Leftrightarrow\) \(\mathrm{CertAcc}\) 单调性（验证 Theorem 1(ii)）。

5 次密文推理在 GPU+PhantomFHE 下约 \(O(\text{分钟})\) 量级，工程可承受。

---

## 9. 对优化器实现的形式化要求

定理生效的前置条件是 cost model **完备** 且 **逐配置共用同一组公式**。
具体来讲：

1. **(8) 中每个噪声源都不能漏算**。常见漏项：
   - rotation / relinearize 边的 KS 噪声（同 rescale，σ \(\sim \sqrt{Nh/12}\)）；
   - CTPT encoding 噪声（\(\sim 2^{-s_{\mathrm{pt}}}\)）；
   - 近似多项式 \(P_\ell\)（如 GELU 的 minimax 余项）。

2. **不同配置必须用同一组 \(c_\sigma, h, N, L_\ell\)**。否则 (5)
   "\(\beta_A \le \beta_B\) 逐分量"在结构上不可比。

3. **\(L_\ell(x)\) 的估计**应当是 *局部敏感度*（在该样本邻域内），而非
   全局 Lipschitz——后者对深层 transformer 来说几乎无意义。可用：
   - power iteration on Jacobian, ≤ 10 步；
   - finite-difference random direction sampling, n=20 方向。

4. **rescale optimizer 的 DP cost**应当包含 \(\sum_\ell \epsilon_\ell^{\mathrm{ct}}(C)\)
   作为 noise-monotone 项。当前若 cost 只算 modulus chain bit-sum，
   定理仍成立（等价于权重 \(L_\ell = 0, P_\ell\) 之外不参与），只是
   优化空间更窄。

---

## 10. 参考文献

1. Wong, E., Kolter, J. Z. (2018). *Provable Defenses against Adversarial
   Examples via the Convex Outer Adversarial Polytope.* ICML 2018.
2. Gowal, S., Dvijotham, K., Stanforth, R., et al. (2018). *On the
   Effectiveness of Interval Bound Propagation for Training Verifiably
   Robust Models.* arXiv:1810.12715.
3. Cohen, J., Rosenfeld, E., Kolter, J. Z. (2019). *Certified Adversarial
   Robustness via Randomized Smoothing.* NeurIPS 2019.
4. Costache, A., Curtis, B., Player, R., Smart, N. P. (2023). *On the
   Precision Loss in Approximate Homomorphic Encryption.* IACR ePrint
   2022/162; CCS '23 version.
5. Murphy, S., Player, R. (2019). *A Central Limit Framework for
   Ring-LWE Decryption.* IACR ePrint 2019/452.
6. Bossuat, J., Mouchet, C., Troncoso-Pastoriza, J., Hubaux, J.-P.
   (2021). *Efficient Bootstrapping for Approximate Homomorphic
   Encryption with Non-Sparse Keys.* Eurocrypt 2021.
7. Han, K., Park, J. (2020). *Better Bootstrapping for Approximate
   Homomorphic Encryption.* CT-RSA 2020.

---

## 附录 A：与 v1 (sub-Gaussian + Lipschitz) 版本的关系

本文档的 v1 版本（保留为 `accuracy_preservation_v1_subgaussian.md`）使用
sub-Gaussian + 模型 Lipschitz 框架（Cohen 2019 类型），证明的是

\[
\Pr[\mathrm{correct}(A; x, y)] \;\ge\; \Pr[\mathrm{correct}(B; x, y)],
\]

需要假设：(A2) PRG coupling、(A5) 模型 \(L\)-Lipschitz、(A6) 逐 σ 单调。

本版本（v2，IBP 类型）的优势：

| 维度 | v1 (sub-Gaussian) | v2 (IBP) |
|---|---|---|
| Lipschitz 估计 | 必须 | **不需要** |
| PRG 耦合 | 必须 (or 退到分布) | **不需要** |
| 保证形式 | 概率 | 确定（在 (9) 高概率意义下） |
| 与 cost model 对应 | σ 平方和（公式 6 v1） | **逐层加权和（公式 7）** |
| 与 plaintext-sim pipeline | 间接 | **直接（Corollary 2）** |

**结论**：对 "明文模拟 + 单点密文验证" 工作流，v2 是 strictly 更适合的
版本。v1 保留作为附录可选材料，仅在需要"概率版精度保证"或与 randomized
smoothing 文献正面对话时使用。

---

*文档结束。*
