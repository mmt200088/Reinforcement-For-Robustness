# CKKS 模数链设计的精度保持定理

> *形式化论证：在 rescale 优化过程中，最小化噪声上界永远不会（在期望意义下）降低端到端推理精度。*

本文给出一个基于 coupling 的单调性定理，把优化器的设计指标（每个原语的
sub-Gaussian 噪声参数）直接连到加密神经网络的实际分类精度。该定理是对
下面这个问题的形式化回答：

> *"我的噪声上界可能会高估实际误差。我怎么知道选择上界更小的设计永远
> 不会损害精度？"*

证明由三块经典材料拼装而成—— sub-Gaussian 集中不等式、概率耦合、神经
网络的 Lipschitz 鲁棒性——并针对 rescale / 模数链优化器的"设计对比"
场景做了专门的剪裁。

---

## 1. 记号与设定

设 `F : R^d -> R^c` 是一个固定的（明文域）推理函数，例如有 `c` 个类别
的分类器。在输入 `x` 上执行加密推理产生扰动输出

```
y_hat(x; A) = F(x) + eta_total(x; A)        (1)
```

其中 `A` 是一个*设计选择*（模数链、scaling factors、rescale skeleton
等），`eta_total(x; A)` 是累积的密文噪声映射回消息域后的总误差。
我们要比较两个设计

```
A, B：电路拓扑、密钥分布、PRG seed 全部一致。
```

`A` 与 `B` 仅在 optimizer 可调的部分（模数链、rescale 位置、scaling
factor 等）上不同。

对电路里每个原语操作 `i`（CTPT mul、CTCT mul、rotation、rescale 等），
记其在消息域注入的噪声分量为 `eta_i`。CKKS / RNS-HKS 的标准噪声分析
给出

```
eta_i  =  sigma_i  *  Z_i                                   (2)
```

其中

* `sigma_i = sigma_i(A)` 是关于设计 `A` 的确定性函数（依赖 rescale
  之后的 scale `Delta`、special prime `P`、ring 维度 `N`、密钥
  Hamming weight `h`，以及对 `ct*pt` 而言的明文 bit 宽 `sf`）；
* `Z_i` 是均值为 0、单位 sub-Gaussian 的*新鲜*随机变量，编码所有
  密码学随机性（密钥、switching key 噪声、舍入方向）。**关键：
  `Z_i` 不依赖 `A`。**

形如 (2) 的分解正是 SEAL 噪声估计器和 Costache 等人 sub-Gaussian
框架所产生的。这一分解是后续一切论证的基石。

输出端的*总* 消息域噪声是一个线性泛函

```
eta_total(x; A)  =  sum_i  W_i(x)  *  sigma_i(A)  *  Z_i     (3)
```

其中 `W_i(x)` 是依赖数据但不依赖设计的"传播权重"，由电路拓扑、输入和
密钥共同决定。

---

## 2. 假设

全文假设：

**(A1)** *统一的噪声框架*。两设计 `A`、`B` 对每个原语都用**同一个**
`sigma_i(.)` 公式——也就是说 sub-Gaussian 噪声模型是共享的，不为每
个设计单独推导。

**(A2)** *PRG 耦合*。`A` 与 `B` 共享同一份密码学随机性 `(Z_i)_i`；
等价地，对 `A` 与 `B` 的 benchmark 用同一个 RNG seed。

**(A3)** *Sub-Gaussian 性*。每个 `Z_i` 是单位 sub-Gaussian，即
`E[exp(t Z_i)] <= exp(t^2 / 2)` 对所有 `t in R` 成立。这对离散
Gaussian 舍入误差、RNS rescale 舍入、ModDown 舍入、以及（在 `N` 个
系数上做中心极限平滑后的）switching-key 残余噪声均成立——它们恰好
是 CKKS 的标准噪声源。

**(A4)** *独立性*。`(Z_i)_i` 在不同原语间相互独立。这在新鲜密钥、
新鲜 secret 假设下成立，是 CKKS 噪声分析的标准启发式。

**(A5)** *Lipschitz 推理*。`F` 在测试分布相关的数据流形上是
`L`-Lipschitz：

```
|| F(x + eta) - F(x) ||_2  <=  L * || eta ||_2     对所有 eta 成立。   (4)
```

对深层模型，可以是局部 `L`：只要 (4) 在 `||eta||` 不超过该样本的
分类 margin 的邻域内成立，下面的定理就够用。

**(A6)** *逐原语 σ 单调*。优化器选 `A` 使得

```
sigma_i(A)  <=  sigma_i(B)        对每个原语 i 成立。              (5)
```

这就是 optimizer 在压缩的设计指标。**注意它不是对 bound 的假设**：
它是 rescale 优化器在用同一份噪声公式比较两个候选解时所返回的
"解的结构性质"。

---

## 3. 引理 1（路径级耦合不等式）

**引理 1.** *在 (A1)–(A2) 与 (A6) 下，对密码学随机性 `omega` 的每一
个具体取值都有*

```
|| eta_total(x; A) (omega) ||_2  <=  || eta_total(x; B) (omega) ||_2.
```

**证明。** 固定任意 `omega`。由 (3),

```
eta_total(x; A)  =  sum_i  W_i(x) * sigma_i(A) * Z_i(omega)
eta_total(x; B)  =  sum_i  W_i(x) * sigma_i(B) * Z_i(omega).
```

两个求和共享相同的 `W_i(x)`（电路与密钥相同）以及相同的 `Z_i(omega)`
（耦合假设 A2）。由 (A6),

```
sigma_i(A)  <=  sigma_i(B)   逐分量。
```

对输出向量的每个坐标 `k`,

```
| eta_total(x; A)_k |  =  | sum_i  W_i(x)_k * sigma_i(A) * Z_i(omega) |
                       <=  | sum_i  W_i(x)_k * sigma_i(B) * Z_i(omega) |
                       =  | eta_total(x; B)_k |
```

成立的前提是每一项被求和的 `W_i(x)_k * Z_i(omega)` 的符号在两边相同
（事实上确实相同：`sigma_i` 只改变幅度，符号由 `W_i(x)_k * Z_i(omega)`
决定）。对 `k` 求平方和即得结论。∎

> **注记。** 这里得到的是逐路径的不等式，是最强意义下的随机占优；它
> **不**依赖任何 tail bound 或 sub-Gaussian 假设。Sub-Gaussian 性
> 只在第 5 节的概率版本里用到。

---

## 4. 引理 2（Sub-Gaussian 尾占优）

**引理 2.** *在 (A1)–(A4) 下，`eta_total(x; A)` 是 sub-Gaussian，
其参数*

```
sigma_total(x; A)^2  =  sum_i  || W_i(x) ||_2^2  *  sigma_i(A)^2.   (6)
```

*特别地，*

```
Pr[ || eta_total(x; A) ||_2  >  t ]
  <=  2 * exp( - t^2 / (2 * sigma_total(x; A)^2) ).                  (7)
```

**证明。** 由 (A4)，`Z_i` 是相互独立的单位 sub-Gaussian。独立单位
sub-Gaussian 的线性组合 `sum_i a_i Z_i` 是参数为
`(sum_i a_i^2)^{1/2}` 的 sub-Gaussian（标准结果，见 Vershynin
*High-Dimensional Probability* 命题 2.6.1）。对每个输出坐标取
`a_i = W_i(x)_k * sigma_i(A)`，再对坐标做联合界即得 (7)。∎

由 (A6) 与 (6),  `sigma_total(x; A)  <=  sigma_total(x; B)`,  故 (7)
对 `A` 给出的右端不超过对 `B` 给出的右端：

```
Pr[ || eta_total(x; A) || > t ]  <=  Pr[ || eta_total(x; B) || > t ]
                                          对所有 t > 0 成立。         (8)
```

这是噪声范数尾分布意义下的随机占优。

---

## 5. 主定理

记 `Acc(A)` 为设计 `A` 下加密分类器的测试精度。等价地，对一条样本
`(x, y_true)`，记 `correct(A; x, y_true) = 1` 当且仅当
`argmax_k F(x + eta_total(x;A))_k = y_true`。
对每条样本，定义其明文 margin

```
m(x)  =  F(x)_{y_true}  -  max_{k != y_true} F(x)_k.                (9)
```

**定理（精度保持）.**
*在假设 (A1)–(A6) 下,*

```
Pr[ correct(A; x, y_true) ]  >=  Pr[ correct(B; x, y_true) ]        (10)
```

*对每条样本逐点成立，其中概率取自密码学随机性。对测试分布求期望，*

```
E[ Acc(A) ]  >=  E[ Acc(B) ].                                       (11)
```

**证明。** 固定一条 margin 严格为正的样本 `(x, y_true)`（明文下被
误分类的样本对两个设计同等贡献，可忽略）。由 Lipschitz 假设 (A5),

```
correct(A; x, y_true)  被  L * || eta_total(x;A) ||_2  <  m(x) / 2  蕴含。
```

（其中 `1/2` 因子是保守取法：扰动幅度小于 `m(x)/(2L)` 时无法把
预测类移走。）定义事件

```
G_A(omega)  =  { L * || eta_total(x;A)(omega) ||_2  <  m(x) / 2 }
G_B(omega)  =  { L * || eta_total(x;B)(omega) ||_2  <  m(x) / 2 }.
```

由引理 1，对*每一个* `omega`,

```
|| eta_total(x;A)(omega) ||  <=  || eta_total(x;B)(omega) ||,
```

故 `G_B(omega)` 蕴含 `G_A(omega)`，即作为概率空间中的事件
`G_B subseteq G_A`。

更关键的是，在样本级 realization 上直接得出蕴含关系：

> 若 `correct(B; x, y_true)(omega) = 1`，则由 Lipschitz 与引理 1，
> 也有 `correct(A; x, y_true)(omega) = 1`，即

```
{ correct(B; x, y_true) }  subseteq  { correct(A; x, y_true) }      (12)
```

作为概率空间中的事件（在 `1/2` 因子的保守意义下）。两边取概率即
得 (10)。对测试分布取期望即得 (11)。∎

> **注记 1（`1/2` 因子的紧致性）.** 主定理中的 `1/2` 可以替换为
> 任意常数 `c < 1`，只需把 `c` 吸收到 Lipschitz 常数里。渐近不
> 等式 (11) 不依赖 `c`。
>
> **注记 2（worst-case 与 average-case bound 的取舍）.** 本定理
> *不*对上界 `B_A` 做任何陈述。它只用到 sub-Gaussian *参数*
> `sigma_i(A)`。因此任何 worst-case bound 的松弛**毫无影响**——
> 重要的只是 optimizer 的指标在结构上是同一组 `sigma_i` 的上界。
>
> **注记 3（为何耦合是关键）.** 没有 (A2)，我们只有较弱的尾占优
> (8)，进而 `Pr[||eta_A||>t] <= Pr[||eta_B||>t]`，仅给出*分布
> 意义下* `Pr[correct(A)] >= Pr[correct(B)]`；realization 级别
> 的事件包含 (12) 必须依靠耦合。在工程上 (A2) 几乎无需付出代价
> ——把同一个 RNG seed 在两次 benchmark 之间共用即可。

---

## 6. 不依赖耦合的概率版本

如果实现层无法保证 (A2)——例如 `A` 与 `B` 在不同时间被 benchmark、
RNG state 不同——仍然可以得到尾占优形式：

**定理（分布意义下的精度保持）.**
*在 (A1)、(A3)–(A6) 下,*

```
Pr[ correct(A; x, y_true) ]
   >=  1  -  2 * exp( - m(x)^2 / (8 * L^2 * sigma_total(x;A)^2) )

   >=  1  -  2 * exp( - m(x)^2 / (8 * L^2 * sigma_total(x;B)^2) ).   (13)
```

右端是 `B` 也满足的下界，故 `A` 的 certified 精度不低于 `B`。
(13) 同时给出一个**显式的 margin 条件**，可直接放进 optimizer 的
feasibility 检查里：

```
sigma_total(x; A)  <  m(x) / ( 2 * L * sqrt( 2 * ln( 2 / delta ) ) )    (14)
```

保证 `Pr[ correct(A; x, y_true) ]  >=  1 - delta`。这与 Cohen 等
（2019）随机平滑里的 certified-radius 公式形式完全一致。

---

## 7. 对优化器的工程含义

定理为优化器中已经实现的若干工程选择给出了形式化依据：

1. **代价函数的正确性。**
   将每段（每尾）的 SNR 上界压低，在 (A1)–(A6) 下保证不会让精度
   下降。无需用经验精度曲线背书 optimizer 的目标。

2. **比较候选 skeleton。**
   当 DP 给出两条候选 skeleton 满足 `sigma_i(A) <= sigma_i(B)`
   逐分量成立，则更便宜的那条（无论按 DP 代价还是按实际 runtime）
   严格优——精度不会损失。

3. **rescale 与不 rescale 的 stage edge。**
   插入一次 rescale 会增大某些 `sigma_i`（增量约 `sqrt(N h / 12)`）
   但减小另一些（`Delta` 得到恢复）。定理告诉 optimizer 的选择
   规则：在传过电路权重 `W_i(x)` 之后，应取*总* `sigma_total`
   更小的那条——这正是 DP 在用合理的 noise 项做
   `lambda_0 * cost + lambda_1 * level` 最小化时的自动行为。

4. **KS 噪声必须正确入帐。**
   `sigma_KS = sqrt(N h / 12)` 这一项应该加给*每条* rotation 与
   每条 CTCT-relinearize 边，**而不是**只加给 rescale 边。否则
   optimizer 可能选出实际 `sigma_total` 更高的设计——只因为某个
   噪声源没被加到聚合里——破坏 (A6)。

5. **验证环节。**
   `validate_cut_points` 应在校准集上检查
   `sigma_total(x;A) < m(x) / (2 L sqrt(2 ln(2/delta)))`。任何
   通过此检查的设计 certified accuracy `>= 1 - delta`。

---

## 8. 实操校准流程

把上述定理落地为可经验背书的判据：

```
Step 1.  在数据流形上估计 L。
         对校准集中每个 x，计算
            L_local(x) = sup_{||v||=1} || JF(x) v ||
         用 power iteration（快、近似）或 random-direction sampling
         （慢、平均）。

Step 2.  在校准集上估计 m(x)。
         直接由明文模型计算。

Step 3.  使用与 optimizer 同源的 noise tracker 计算
         sigma_total(x; A)（在 rescale optimizer 中已自动完成）。

Step 4.  对每个候选设计 A，评估 (14)：
            certified-fraction(A) = #{ x : (14) 在 delta = 1e-3 下成立 }
         与 optimizer 的 cost 一并报告。

Step 5.  （可选）在少量真实加密推理上验证经验精度不低于 certified
         比例。这是唯一消耗 FHE 算力的步骤。
```

step 4 与 step 5 一致即为该 optimizer 的设计指标确实跟踪真实精度
的经验证据。

---

## 9. 参考文献

1. Cohen, J., Rosenfeld, E., Kolter, J. Z. (2019).
   *Certified Adversarial Robustness via Randomized Smoothing.*
   NeurIPS 2019.
2. Costache, A., Curtis, B., Player, R., Smart, N. P. (2023).
   *On the Precision Loss in Approximate Homomorphic Encryption.*
   IACR ePrint 2022/162；CCS '23 版。
3. Murphy, S., Player, R. (2019).
   *A Central Limit Framework for Ring-LWE Decryption.*
   IACR ePrint 2019/452.
4. Bossuat, J., Mouchet, C., Troncoso-Pastoriza, J., Hubaux, J.-P.
   (2021). *Efficient Bootstrapping for Approximate Homomorphic
   Encryption with Non-Sparse Keys.* Eurocrypt 2021.
5. Han, K., Park, J. (2020).
   *Better Bootstrapping for Approximate Homomorphic Encryption.*
   CT-RSA 2020.
6. Lecuyer, M., Atlidakis, V., Geambasu, R., Hsu, D., Jana, S. (2019).
   *Certified Robustness to Adversarial Examples with Differential
   Privacy.* IEEE S&P.
7. Vershynin, R. (2018).
   *High-Dimensional Probability: An Introduction with Applications in
   Data Science.* Cambridge University Press.（§2.5–§2.6 是引理 2
   所用 sub-Gaussian 性质的来源。）
8. Lindvall, T. (2002).
   *Lectures on the Coupling Method.* Dover.（引理 1 中耦合论证的
   参考。）

---

## 10. 局限与开放问题

1. **(A5) 是局部 Lipschitz 而非全局。** 深层 transformer 的全局
   Lipschitz 常数巨大，但在典型输入邻域内的局部常数很小。定理的
   经验内容依赖局部 Lipschitz 估计（校准 step 1）；某条样本 Lipschitz
   单调性失败不影响期望版 (11)，但会削弱样本级保证 (10)。

2. **(A4) 是启发式独立性。** 严格说不同原语的 `Z_i` 通过密钥弱
   依赖；Costache 等（2023）证明对常用 `N >= 2^{14}` 该依赖是次
   导项。

3. **(A6) 要求 optimizer 用一个完备的 noise model。** 若 cost
   model 对某一设计漏算了某个 noise 源、对另一设计未漏算，(A6)
   表面成立但实际 `sigma_total(A) > sigma_total(B)`，定理失效。
   因此审视 cost model 的完备性（如确认 rotation 也带 KS-noise，
   像 rescale 那样）是定理生效的*前置条件*。

4. **不给绝对精度保证。** 定理是*相对*保证：`A` 不比 `B` 差。它
   不约束两者距明文精度多远。要绝对保证，可用 (14) 的 certified
   比例公式——前提是 `m(x)` 与 `L` 估计可靠。

---

*文档结束。*
