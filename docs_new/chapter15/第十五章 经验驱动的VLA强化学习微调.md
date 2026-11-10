# 第十五章 经验驱动的VLA强化学习微调

> **本章导读**：第十四章完成了 π0.5 的部署与微调，模型可以根据视觉、语言和机器人状态输出连续动作块。本章接着解决一个更接近真实机器人落地的问题：**模型已经会做任务，但做得不够稳定，怎样让它利用自己的执行经验继续进步？**
>
> 截止到教程当前发布时间，由于 pi0.6 没有发布权重，且初始 RL 预训练较困难，因此本章的经验学习主线基于 **Evo-RL** 进行整理和复现。该项目给出了从策略 rollout、人类干预数据、Value 训练、Value 推理、优势条件策略训练到下一轮闭环采集的可运行流程。本章没有原样照搬其机器人平台和人工接管接口，而是结合 G2 Omnipicker、Isaac Sim 与第十四章的 π0.5/OpenPI 数据合同重新实现，并进一步加入自动触发、分段 IK 纠正、严格的三态控制权和单帧 `RELEASE` 交接。也就是说，Value、Advantage 条件训练与多轮经验聚合构成基础主线，自动纠正和仿真安全交接则是本章在此基础上的重点改进。
>
> 本章最终要跑通的案例：G2 Omnipicker 在 Isaac Sim 中完成红、绿、蓝三色物块入盒任务，走完“SFT → 带自动纠正的经验采集 → Value 训练 → Advantage 标注 → 优势条件化微调 π0.5 → 无纠正评测”的完整一轮，并对比 SFT 与 RL Round 1 的成功率。

## 本章概览

### 本章要解决的问题

设想你已经用 20 条/色的自动示教把 π0.5 微调好，它在红色和绿色物块上几乎总能成功，但蓝色物块只有约四分之三的成功率，连续完成三个物块时更是只有一半左右。部署过程中机器人产生了大量轨迹：有的成功但绕了远路，有的抓空后停在原地，有的撞飞物块。传统行为克隆（Behavior Cloning, BC）只会继续堆成功示教，而这些“失败与恢复”的经验被白白浪费了；直接上 PPO（Proximal Policy Optimization）一类在线策略梯度方法，又会遇到 flow matching 动作似然难算、采样昂贵、训练不稳定和安全风险高等问题。本章要回答的就是：**如何在不改动 π0.5 训练基础设施的前提下，把成功、失败、纠正这些经验都变成有效的学习信号。**

传统行为克隆主要学习成功示教，能够让机器人获得初始技能，却很难充分利用部署过程中产生的失败轨迹、恢复过程和速度差异。为此，本章实现了一套面向连续动作 VLA 的迭代改进流程：先让策略在 Isaac Sim 中自主执行任务，再用自动纠正模块处理危险动作与停滞状态；随后训练 Value 模型估计每一帧距离成功还有多远，计算动作优势并生成正、负条件标签，最后用条件化监督学习继续微调 π0.5。

### 学习目标

- **解释** 为什么只学习成功示教不够，以及为什么本章不直接使用 PPO，而采用优势条件化（Advantage-Conditioned Policy, ACP）。
- **推导** 本章的稀疏奖励、逐帧 Value target、稠密奖励和 50 步 Advantage 公式，并能用小例子手算。
- **解释** `POLICY / ACTIVE / RELEASE` 三态控制权和单帧 `RELEASE` 交接为什么是数据正确性的关键。
- **实现/读懂** `value_math.py` 中 `value_targets()`、`two_hot()`、`dense_rewards()`、`n_step_advantage()`、`acp_labels()` 的实现，并把它们与公式一一对应。
- **运行** 从示教采集、SFT、经验采集、Value 训练、Advantage 标注到优势条件化训练和无纠正评测的完整流程。
- **区分** 本章实现与 π0.6 / π\*0.6（RECAP）在模型、Value、条件注入与纠正来源上的差异。

### 前置知识

- 第十四章：π0.5 / OpenPI 的数据合同、三路相机输入、WebSocket 策略服务、LoRA 微调与 checkpoint 结构。
- 第十三章及之前：VLA 的基本概念、flow matching 动作生成、G2 仿真场景与数值 IK。
- 数学：期望、条件期望、分位数、交叉熵、softmax；不需要事先学过强化学习，第二部分会从 MDP 讲起。
- 编程：Python、NumPy、PyTorch 基础；能读懂 argparse 命令行脚本和 dataclass。

### 知识地图

![图 15-1 本章知识地图](assets/new_fig15_01_knowledge_map.svg)

<p align="center"><b>图 15-1</b> 本章知识地图：上行是原理（第一至第六部分），下行是对应代码模块；第五部分得到的新策略回到第三部分开始下一轮采集</p>

### 对应代码

`code/code_chapter15/`：在 Isaac Sim 中实现 G2 三色物块任务的“自主 rollout + 自动纠正 → 分布式 Value → 50 步 Advantage → 优势条件化 π0.5 微调 → 无干预评测”完整闭环；OpenPI 源码位于 `code/code_chapter15/third_party/openpi`。

在进入正文之前，需要先明确四个边界：

1. **本章实现采用 π0.5，而不是直接部署 π0.6。** 标准 π0.6 升级了模型主干、动作专家、预训练数据和离散动作预测机制；在此基础上加入优势条件的经验学习版本通常记作 π0.6\*（也写作 π\*0.6）。截至本章代码整理与验证时，没有公开可直接接入当前 OpenPI/G2 工程链路的 π0.6/π0.6\* 训练实现和 checkpoint；当前内嵌 OpenPI 明确提供并支持的是 π0、π0-FAST 与 π0.5。因此，本章选择公开可获得、且已在第十四章完成适配的 π0.5 作为策略基座，在代码层实现经验学习闭环，而不是把论文中的 π0.6 名称当作一个可替换的权重目录。
2. **这里的“强化学习微调”不是在线 PPO。** 策略先采集经验，Value 再离线估计优势，最后通过优势条件化的监督目标更新策略。它利用奖励与失败经验改进策略，但不要求显式计算 flow matching 策略的精确对数概率。
3. **自动纠正只用于训练数据采集。** 最终评测禁止脚本专家、恢复动作和人工接管，否则测到的是“策略加纠正器”的系统成功率，而不是策略本身的能力。
4. **仿真纠正不等于真实机器人安全认证。** 真实设备还需要独立的碰撞检测、力矩限制、速度限制、急停和分级放权机制。

---

## 第一部分 从行为克隆到经验驱动改进

### 1.1 为什么只学习成功示教还不够

**直观理解**：行为克隆像“照着标准答案抄作业”。学生能学会标准解法，但一旦第一步写错，就不知道怎么改回来，因为标准答案里从来没有“写错之后怎么办”。

**原理与公式**：行为克隆把专家数据写成观察与动作对：

$$
\mathcal{D}_{\mathrm{demo}}=\{(o_t,a_t)\}
$$

并训练策略复现专家动作：

$$
\min_\theta
\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}_{\mathrm{demo}}}
\left[\mathcal{L}_{\mathrm{VLA}}(\pi_\theta(o_t),a_t)\right]
$$

符号说明：

| 符号 | 含义 | 本章取值/单位 |
|---|---|---|
| $o_t$ | 第 $t$ 帧观察 | 三路相机图像（head / left_wrist / right_wrist）+ 任务语言 + 16 维机器人状态 |
| $a_t$ | 第 $t$ 帧动作 | 16 维绝对关节与夹爪目标（rad），顺序为左臂 7、右臂 7、左夹爪 1、右夹爪 1 |
| $\pi_\theta$ | 参数为 $\theta$ 的策略 | π0.5（Gemma 2B VLM + Gemma 300M 动作专家） |
| $\mathcal{L}_{\mathrm{VLA}}$ | 监督损失 | 对 π0.5 而言，主要连续动作部分是 flow matching 损失 |

这种方法适合“教会机器人做什么”，但部署后常出现三类数据分布变化：

- 机械臂偏离示教轨迹后，看到了训练集里很少出现的恢复状态；
- 相机视角、物块位置和执行误差让同一任务产生不同难度；
- 一条轨迹虽然最终成功，但中间可能绕路、停滞或做出危险的大幅动作。

如果只保留成功轨迹，模型不知道哪些动作导致了失败；如果把自主采集的所有轨迹当作普通示教直接训练，又会把坏动作一并模仿。因此，本章不再只问“这个动作是否来自专家”，而是进一步估计：

> 在当前观察下执行这一步以后，任务成功的前景是变好还是变差？

这个问题由 Value 和 Advantage 两个量回答。

**图示**：

![图 15-2 行为克隆与经验驱动改进对比](assets/new_fig15_02_bc_vs_experience.svg)

<p align="center"><b>图 15-2</b> 左：行为克隆只学习专家轨迹，部署偏离后进入未见状态；右：经验驱动改进把好于预期的动作、变差的动作和自动纠正动作都纳入训练，并用条件标签加以区分</p>

**案例**：本章 SFT 模型在无纠正评测中，蓝色单物块成功率为 37/50 = 74%（见第八部分运行结果）。这 13 条失败轨迹在纯行为克隆路线中不会被使用；而在本章流程中，它们与成功轨迹一起进入 Value 训练，其中“抓取前对齐得好”的片段仍可能被标为 positive，“撞飞物块”的片段被标为 negative。

> 💡 **要点**：问题不是“数据不够多”，而是“数据缺少好坏的区分信号”。Value 与 Advantage 就是为每一帧补上这个信号。

### 1.2 Value 与 Advantage 的直观含义

理解 Value 前，必须先区分三个容易混淆的量：

1. **奖励 $r_t$**：人为规定的计分规则，例如普通步骤为 $-1$、成功终止为 $0$、失败终止受到额外惩罚；
2. **回报 target $y_t$**：一个 episode 结束后，根据最终成败和剩余步数为该轨迹的第 $t$ 帧计算出的监督标签；
3. **Value 预测 $\hat V(o_t,\ell)$**：模型只看当前观察与任务，对未来平均回报作出的估计。

前两个量在一条已经完成的轨迹上可以按公式确定，第三个量却不是全数据共用的固定数字。Value 函数 $V(o_t,\ell)$ 表示：在任务 $\ell$ 下，从当前观察 $o_t$ 继续按照参考策略执行，未来预期能获得多大回报。本章把回报设计成“越快成功越好，失败受到额外惩罚”，所以 Value 被归一化到 $[-1,0]$：

- 越接近 $0$，表示越接近成功；
- 越接近 $-1$，表示距离成功较远，或者当前状态更可能通向失败；
- 若机械臂抓稳物块并向盒子移动，Value 应总体上升；
- 若物块掉落、动作停滞或轨迹偏离，Value 通常下降。

可以把 Value 想成老师在学生做题过程中给出的“当前局面预期分”。老师只看到目前写到哪一步，并不知道学生接下来一定会怎样操作。奖励与最终成败负责告诉老师过去的练习结果，Value 训练则让老师学会看到新局面时也能估计后续前景。

Advantage 比较“执行当前动作片段后的实际局部结果”与“执行前 Value 的预期”：

$$
A_t^{(N)}=
\sum_{k=0}^{N-1}r_{t+k}
+V(o_{t+N},\ell)-V(o_t,\ell)
$$

本章固定 $N=50$ 且 $\gamma=1$。较高的 Advantage 表示这一动作片段让任务进展好于当前 Value 的预期，较低的 Advantage 表示动作没有带来足够进展，甚至使任务变差。后文的 positive/negative 是相对任务内分位数得到的类别名，并不要求 Advantage 严格大于或小于 $0$。

需要注意，Advantage 不是“成功帧为正、失败帧为负”的简单标签。一条失败轨迹中也可能包含正确的接近、抓取和搬运动作；一条成功轨迹中也可能包含碰撞、绕路和停滞。Value 提供当前状态的基准，Advantage 才负责判断局部动作相对这个基准做得更好还是更差。

> ⚠️ **注意**：Value 评价“状态”（局面好不好），Advantage 评价“动作片段”（这段操作有没有让局面比预期更好）。两者不能混用，第二部分会用具体数字说明。

### 1.3 为什么使用优势条件化，而不是直接做 PPO

**直观理解**：PPO 像“根据成绩直接调整每一个笔画的概率”，需要精确知道“写出这个笔画的概率是多少”；优势条件化则像“给每份作业贴上‘好作业/差作业’标签，让学生同时学两种风格，考试时只按‘好作业’风格写”。

π0.5 的连续动作由 flow matching 动作专家生成。传统 PPO 通常需要计算新旧策略对同一动作的概率比

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}
$$

但 flow matching 策略并没有像普通高斯策略那样方便、稳定的显式动作似然。若为了策略梯度再额外估计似然、噪声轨迹或重要性权重，系统会变得复杂，而且真实机器人数据量通常远小于仿真游戏中的采样量。

本章采用更适合大型 VLA 的方式：把 Advantage 离散成一个条件变量 $I_t$，再让策略学习

$$
\pi_\theta(a_t\mid o_t,\ell,I_t)
$$

其中 $I_t=1$ 表示 positive， $I_t=0$ 表示 negative。训练时成功示教、自主动作、失败轨迹和纠正动作都可以保留，只是在任务文本后附加不同的优势条件。推理时固定请求 positive 条件，相当于告诉策略：“请从训练数据中选择更有利于完成任务的动作模式。”

这种做法有三个工程优势：

1. 不需要给 flow matching 动作计算精确的 PPO 概率比；
2. 不丢弃失败数据，而是把失败经验作为反例条件保留下来；
3. 策略更新仍沿用 π0.5 已有的数据管线和监督式训练基础设施。

因此，本章的本质是**由 Value 提供学习信号、由 Advantage 区分动作质量、由条件化行为克隆完成策略提取**。

下面用一张对比表总结三种路线：

| 路线 | 需要动作似然 | 是否利用失败数据 | 训练基础设施 | 本章是否采用 |
|---|---|---|---|---|
| 行为克隆 / SFT | 否 | 否（只用成功示教） | 监督学习 | 作为可选初始策略 |
| PPO 等在线策略梯度 | 是（概率比） | 是 | 需要在线采样与 RL 训练框架 | 否 |
| 优势条件化（ACP） | 否 | 是（作为 negative 条件） | 复用 π0.5 flow matching 监督训练 | 是 |

**对应代码**：推理时“固定请求 positive 条件”只有一行，位于 `code/code_chapter15/policy_client.py`：

```python
def policy_observation(images, state, task: str) -> dict:
    """Build the native 16D G2 observation expected by ``pi05_adapter.Inputs``."""
    head, left, right = images
    return {
        "images": {
            "head": np.asarray(head, dtype=np.uint8),
            "left_wrist": np.asarray(left, dtype=np.uint8),
            "right_wrist": np.asarray(right, dtype=np.uint8),
        },
        "state": np.asarray(state, dtype=np.float32).reshape(ACTION_DIM),
        # Evo-RL inference requests the positive branch. No CFG is used.
        "prompt": tagged_task(task, positive=True),   # 部署时永远附加 "Advantage: positive"
    }
```

> 💡 **要点**：优势条件化把“强化学习”拆成两步：先离线算出每帧动作的好坏（Value + Advantage），再用普通监督学习把“好坏”作为输入条件学进策略。

---

## 第二部分 强化学习基础：MDP、回报、Value 与 Advantage

这一部分面向没有学过强化学习（Reinforcement Learning, RL）的读者，用本章的物块任务把 RL 的核心概念过一遍。后面所有公式都只是这些概念的具体化。

### 2.1 马尔可夫决策过程

**直观理解**：强化学习把“机器人做任务”抽象成一个循环：看一眼（观察），做一步（动作），环境变化并给出一个分数（奖励），再看一眼……直到任务结束。

**原理与公式**：马尔可夫决策过程（Markov Decision Process, MDP）由五元组描述：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)
$$

| 符号 | 含义 | 在本章中的对应 |
|---|---|---|
| $\mathcal{S}$ | 状态空间 | 物块、盒子、机械臂的真实物理状态；策略只能看到其观察 $o_t$ |
| $\mathcal{A}$ | 动作空间 | 16 维绝对关节/夹爪目标 |
| $P(s_{t+1}\mid s_t,a_t)$ | 状态转移 | Isaac Sim 物理仿真（120 Hz 物理，10 Hz 记录） |
| $r(s_t,a_t)$ | 奖励 | 每步 $-1$，成功终止 $0$，失败终止 $-C_{\mathrm{fail}}$ |
| $\gamma$ | 折扣因子 | 本章取 $\gamma=1$（不打折扣，episode 有限长） |

“马尔可夫”指下一状态只依赖当前状态和动作，与更早的历史无关。真实机器人中策略看到的是相机图像而非完整物理状态，严格说是部分可观测问题，本章沿用常见做法，把 $o_t$ 当作状态的近似。

一次完整执行称为一个 **episode**（回合），它产生一条轨迹：

$$
\tau=(o_0,a_0,r_0,o_1,a_1,r_1,\dots,o_{T-1},a_{T-1},r_{T-1})
$$

本章中 episode 在三种情况下结束：任务成功（`sim.task.success(color)` 为真）、达到帧数上限（`--max-frames 300`）、或自动纠正 IK 失败。

**图示**：

![图 15-3 物块入盒任务的 MDP](assets/new_fig15_03_mdp_loop.svg)

<p align="center"><b>图 15-3</b> 智能体（π0.5）与环境（Isaac Sim + G2）的交互循环：策略输出 16 维绝对动作，环境返回新观察和奖励</p>

**对应代码**：环境侧判定“成功”的几何条件来自 `code/code_chapter15/simulation.py`：

```python
    def inside_box(self, color):
        pos = self.block_position(color)
        box = np.asarray(self.cfg.box_position)
        hx, hz = np.asarray(self.cfg.box_inner_size) / 2
        return bool(
            abs(pos[0] - box[0]) <= hx + 0.02          # 水平方向在盒子内侧范围（留 2 cm 余量）
            and abs(pos[2] - box[2]) <= hz + 0.02
            and self.box_top_y - self.cfg.block_size / 2
            <= pos[1]
            <= self.cfg.table_top_y + self.cfg.block_size / 2   # 高度落在盒内
        )
    # ...
    def success(self, color):
        return color in self.completed and self.inside_box(color)   # 已松爪完成 且 仍在盒内
```

代码解读：`success()` 同时要求 `completed`（夹爪松开后被记为完成）和 `inside_box()`，这正是原文所说“目标物块必须位于盒子内部，并在夹爪松开后仍保持在盒内”，避免仅凭“末端到达盒子上方”就把尚未放稳的轨迹标成成功。

### 2.2 回报：把一串奖励加起来

**直观理解**：奖励是“每一步的得分”，回报是“从现在起到结束总共能得多少分”。

**原理与公式**：从第 $t$ 帧开始的回报（return）为

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

本章 $\gamma=1$，于是 $G_t$ 就是后续所有奖励之和。由于每走一步扣 1 分、失败再额外扣 $C_{\mathrm{fail}}$，回报自然编码了“越快成功越好、失败最差”。

**案例**：一条 4 帧的成功 episode（ $T=4$），奖励序列为 $(-1,-1,-1,0)$，于是

$$
G_0=-3,\quad G_1=-2,\quad G_2=-1,\quad G_3=0
$$

若同样 4 帧但失败，且 $C_{\mathrm{fail}}=4$，奖励为 $(-1,-1,-1,-4)$，则

$$
G_0=-7,\quad G_1=-6,\quad G_2=-5,\quad G_3=-4
$$

可以看到：成功轨迹的回报越往后越接近 0；失败轨迹整体被压低一个 $C_{\mathrm{fail}}$。第四部分会把它们除以 $T_{\max}+C_{\mathrm{fail}}=8$ 归一化到 $[-1,0]$，得到成功 $(-0.375,-0.25,-0.125,0)$ 与失败 $(-0.875,-0.75,-0.625,-0.5)$——这正是 `value_math.value_targets()` 在 $T_{\max}=4$ 时的真实输出。

> ⚠️ **注意**：回报 $G_t$ 是针对“某一条已经发生的轨迹”算出来的，是一个具体数字；Value 是对“所有可能的未来”求平均，是一个期望。

### 2.3 Value：对回报的期望

**原理与公式**：状态价值函数（state-value function）定义为在策略 $\pi$ 下从当前观察出发的期望回报：

$$
V^\pi(o_t,\ell)=\mathbb{E}_\pi\left[G_t\mid o_t,\ell\right]
$$

同样的“夹爪在物块上方”这一局面，在数据中可能 7 次成功、3 次失败， $V$ 就是这些结果的平均。它不知道这一次会怎样，只知道“平均来说”前景如何。

动作价值函数（action-value function）则多固定了一个动作：

$$
Q^\pi(o_t,a_t,\ell)=\mathbb{E}_\pi\left[G_t\mid o_t,a_t,\ell\right]
$$

本章不单独训练 $Q$ 网络，而是用“未来 $N$ 步真实奖励 + 第 $N$ 步的 Value”来近似它（称为 $N$ 步 bootstrap）：

$$
\hat Q_t^{(N)}=\sum_{k=0}^{N-1}r_{t+k}+\hat V(o_{t+N},\ell)
$$

**案例**：假设在“夹爪已到物块上方”状态，数据中有 10 条后续轨迹，其中 7 条成功、平均归一化回报 $-0.20$，3 条失败、平均归一化回报 $-0.70$，则该状态的 Value 应接近

$$
V\approx 0.7\times(-0.20)+0.3\times(-0.70)=-0.14-0.21=-0.35
$$

### 2.4 Advantage：比预期好多少

**原理与公式**：优势函数（advantage function）定义为

$$
A^\pi(o_t,a_t)=Q^\pi(o_t,a_t)-V^\pi(o_t)
$$

它回答“采取这个动作，比在这个状态下的平均表现好多少”。代入 $N$ 步近似：

$$
A_t^{(N)}=\sum_{k=0}^{N-1}r_{t+k}+\hat V(o_{t+N},\ell)-\hat V(o_t,\ell)
$$

**图示与案例**：为了直观理解，先把 50 步缩短成 5 步进行手算。假设机器人准备抓取红色物块，当前 Value 为 $\hat V(o_t)=-0.60$，未来 5 步累计时间代价为 $-0.10$。

- **较好动作**：夹爪先移动到物块正上方并完成对齐，5 步后 Value 提升到 $-0.25$：

$$
A_t^{(5)}=-0.10+(-0.25)-(-0.60)=+0.25
$$

- **较差动作**：夹爪从侧面横向扫动并撞飞物块，5 步后 Value 降到 $-0.75$：

$$
A_t^{(5)}=-0.10+(-0.75)-(-0.60)=-0.25
$$

![图 15-4 5 步 Advantage 手算](assets/new_fig15_04_advantage_example.svg)

<p align="center"><b>图 15-4</b> 同一起点 V=−0.60，较好动作把局面推到 −0.25（A=+0.25），较差动作把局面拉到 −0.75（A=−0.25）</p>

这个例子说明：Value 判断当前局面好不好，Advantage 判断刚才这一段动作有没有让局面变得比预期更好。实际代码仍使用 50 步，5 步只用于说明计算方向。

减去当前 Value 还有一个重要作用：让动作质量相对于状态难度进行比较。假设一个非常容易的状态原本为 $V=-0.05$，动作后局部结果为 $Q=-0.08$，虽然绝对分数仍高，但 $A=-0.03$，说明它低于这个简单状态本应达到的水平。另一个困难状态原本为 $V=-0.80$，恢复后达到 $Q=-0.55$，绝对分数仍不高，但 $A=+0.25$，说明这是一段有效恢复。没有 Value 基准时，困难状态中的好动作很容易被绝对回报掩盖。

| 场景 | $V(o_t)$ | $\hat Q_t$ | $A_t$ | 结论 |
|---|---:|---:|---:|---|
| 简单状态，动作平庸 | $-0.05$ | $-0.08$ | $-0.03$ | 绝对分高，但低于该状态应有水平 |
| 困难状态，有效恢复 | $-0.80$ | $-0.55$ | $+0.25$ | 绝对分低，但这是一段好动作 |

> 💡 **要点**：把 $V$ 作为“基线（baseline）”减掉，是策略梯度方法中降低方差的经典技巧；本章借用同样的思想，但把结果用来打标签而不是直接求梯度。

### 2.5 从 RL 概念到本章四个函数

把本部分的概念与代码对齐，后续各部分会逐一展开：

| RL 概念 | 本章中的具体形式 | 代码位置 |
|---|---|---|
| 奖励 $r_t$ | 每步 $-1$、成功 $0$、失败 $-C_{\mathrm{fail}}$；训练时用相邻 target 差构造稠密奖励 | `value_math.dense_rewards()` |
| 回报 $G_t$ / target $y_t$ | 归一化到 $[-1,0]$ 的 Monte Carlo 回报 | `value_math.value_targets()` |
| Value $V$ | 201-bin 分布式回归网络的期望输出 | `value_model.Pistar06Value` |
| Advantage $A$ | 50 步 bootstrap 估计 | `value_math.n_step_advantage()` |
| 策略改进 | 按任务内分位数二值化，条件化监督训练 | `value_math.acp_labels()`、`acp.py` |

---

## 第三部分 经验收集循环与自动纠正

### 3.1 一轮迭代包含哪些阶段

**直观理解**：一轮迭代就像“考试—批改—订正—再学习”：先让模型自己做（rollout），卡住时由助教局部提示（自动纠正），然后批改每一步（Value + Advantage），最后按批改结果再学一遍（条件化微调）。

本章的一轮改进按照下面的顺序进行。完整的一轮包括七步：①载入初始 π0.5 或上一轮策略；②进行自主 rollout；③在异常或停滞时执行局部自动纠正，并从纠正后的观察重新推理；④保存成功、失败、自主动作和纠正动作；⑤使用累计经验训练 Value；⑥计算 50 步 Advantage，并按任务生成正、负标签；⑦导出 LeRobot 数据、计算归一化统计并微调 π0.5。完成后可进入下一轮采集，所有轮次结束后再进行无纠正评测。

![图 15-5 经验驱动的 VLA 强化学习微调闭环](assets/vla_experience_loop.svg)

<p align="center"><b>图 15-5</b> 经验采集、价值评估、优势标注与策略更新构成逐轮累积的闭环</p>

七步与脚本的对应关系：

```text
 ①初始策略 ──► serve_policy.py --checkpoint <SFT 或上一轮>
 ②自主 rollout ─┐
 ③自动纠正     ├─► collect_rollouts.py（hil.py / rollout_core.py / auto_expert.py）
 ④保存经验     ┘        └─► data/raw/round_xxx/episode_*.npz
 ⑤训练 Value  ──► train_value.py        └─► checkpoints/value_round_xxx
 ⑥Advantage   ──► label_advantages.py   └─► data/labeled/round_xxx
 ⑦更新策略    ──► export_dataset.py → compute_norm_stats.py → train_policy.py
                                        └─► checkpoints/g2_pi05_acp_xxx  ──► 回到 ①
 最终评测     ──► evaluate.py（不加载自动专家）
```

初始策略有两条来源：一条是先用成功示教进行普通 SFT，再进入经验改进；另一条是从已经适配 G2 数据合同的 π0.5 checkpoint 直接开始。若使用通用基础权重，但没有匹配 G2 的相机键、16 维状态动作顺序和归一化统计，即使模型能加载，也不能认为它能够可靠控制当前机器人。

每一轮 Value 都读取当前轮及之前所有可用经验，而不是只训练最新 rollout。这样可以避免模型只记住最近一批状态，同时让示教、旧策略经验、新策略经验和纠正片段共同构成行为分布。下一轮策略则以前一轮 checkpoint 为初始化，从更新后的优势标签中继续学习。

### 3.2 采集主循环

**对应代码**：`code/code_chapter15/collect_rollouts.py` 中每个 episode 的主循环（删减了日志与续采逻辑）：

```python
                while len(recorder) < args.max_frames:
                    if sim.task.success(color):          # 每帧先检查是否成功
                        stop_reason = "success"
                        break

                    if trigger:                          # 上一帧触发了纠正
                        segment, correction_error = run_automatic_correction(
                            sim, robot, recorder, runtime, task, color, controller,
                        )
                        if correction_error is not None: # IK 失败：安全中止，episode 记为失败
                            stop_reason = "automatic_correction_ik_failed"
                            break
                        correction_segments.append(segment)
                        detector.reset(sim.task.goal_distance(color), robot.state())
                        trigger = False
                        continue

                    images = sim.cameras.capture()
                    state = robot.state()
                    try:
                        action = runtime.next_action(images, state, task)   # 从 50 步队列取动作
                    except Exception as exc:
                        # ...
                        trigger = True                   # 推理异常 → 触发纠正
                        continue
                    if unsafe_action(action, state):     # NaN/Inf/形状/关节跳变 → 触发纠正
                        runtime.reset()
                        trigger = True
                        continue
                    execute_recorded_action(
                        sim, robot, recorder, action,
                        state=HILState.POLICY, source="policy", policy_action=action,
                    )
                    trigger = detector.observe(          # 停滞检测 → 可能触发纠正
                        sim.task.goal_distance(color), robot.state()
                    )

                success = sim.task.success(color)
                path = recorder.save(                    # 成功与失败都保存
                    success=success,
                    episode_kind="corrected_rollout",
                    stop_reason="success" if success else stop_reason,
                    correction_segments=correction_segments,
                    use_for_sft=False,
                    use_for_value=True,
                )
```

代码解读：

- 三类触发源（推理异常、`unsafe_action`、`detector.observe`）最终都只做一件事：把 `trigger` 置为真，下一次循环统一进入 `run_automatic_correction()`；
- 无论 episode 最终成功、超时还是纠正失败，已经记录的轨迹都会由 `EpisodeRecorder.save()` 保存，保证 Value 同时看到成功和失败经验；
- rollout 数据 `use_for_sft=False`、`use_for_value=True`：它们不会混进 SFT，但一定进入 Value 训练；
- `collector_policy_id` 在创建 `EpisodeRecorder` 时设为 `acp_round_{round:03d}`，便于事后诊断数据来自哪一轮策略。

### 3.3 三态控制权管理

**直观理解**：驾校教练车有两套踏板。为了事后能分清“这一脚刹车是学员踩的还是教练踩的”，每一刻必须只有一个人在控制，并且交还控制权时要明确地“交接”一次。

自动纠正不能简单地把专家动作与策略动作混在一起，否则很难判断数据是谁产生的，也容易在专家结束后继续执行策略队列中的旧动作。本章为每一帧记录三种状态：

| 状态 | 含义 | 执行动作 | 数据要求 |
|---|---|---|---|
| `POLICY` | 策略正常控制 | π0.5 输出 | 保存实际策略动作 |
| `ACTIVE` | 自动专家正在纠正 | IK 与脚本轨迹 | `policy_action` 必须为全零 |
| `RELEASE` | 纠正结束后的交接帧 | 从纠正后的新观察重新推理 | 必须且只能持续一帧 |

状态转换严格遵循：

$$
\text{POLICY}\rightarrow\text{ACTIVE}\rightarrow
\text{RELEASE}\rightarrow\text{POLICY}
$$

`RELEASE` 是本章自动纠正设计中非常关键的一步。π0.5 一次会预测 50 步动作，并由客户端队列逐步执行。如果专家纠正完以后继续使用旧队列，旧动作仍然基于纠正前的图像与关节状态，可能立即把机器人带回错误轨迹。因此交接时必须清空动作队列，使用纠正后的三路图像与当前状态进行一次 fresh inference，记录一帧后再回到普通策略控制。

![图 15-6 自动纠正的三态控制权状态机](assets/control_handoff_state_machine.svg)

<p align="center"><b>图 15-6</b> 三态状态机保证动作来源唯一，并用单帧 RELEASE 隔离纠正前后的动作队列</p>

**对应代码**：状态机本身在 `code/code_chapter15/hil.py`，每个转换都有前置检查：

```python
class HILState(IntEnum):
    POLICY = 0
    ACTIVE = 1
    RELEASE = 2

class HILController:
    def __init__(self) -> None:
        self.state = HILState.POLICY

    def start(self) -> None:                     # POLICY → ACTIVE
        if self.state is not HILState.POLICY:
            raise RuntimeError("intervention can only start from POLICY")
        self.state = HILState.ACTIVE

    def finish(self) -> None:                    # ACTIVE → RELEASE
        if self.state is not HILState.ACTIVE:
            raise RuntimeError("only ACTIVE can finish")
        self.state = HILState.RELEASE

    def abort(self) -> None:                     # ACTIVE → POLICY（IK 预规划失败）
        """Return to POLICY when a preplanned correction cannot start."""
        if self.state is not HILState.ACTIVE:
            raise RuntimeError("only ACTIVE can abort")
        self.state = HILState.POLICY

    def after_frame(self) -> None:               # RELEASE → POLICY（只过一帧）
        if self.state is HILState.RELEASE:
            self.state = HILState.POLICY
```

单帧交接在 `code/code_chapter15/rollout_core.py`：

```python
def execute_release_frame(
    sim, robot, recorder, runtime, task: str, controller: HILController
) -> np.ndarray:
    """Clear the queue, infer from the corrected observation, and record exactly one S2 frame."""
    if controller.state is not HILState.RELEASE:
        raise RuntimeError("release frame requires RELEASE state")
    runtime.reset()                              # ① 丢弃纠正前的旧动作块
    images = sim.cameras.capture()               # ② 纠正后的新图像
    state = robot.state()                        #    与新关节状态
    action = runtime.next_action(images, state, task, force_fresh=False)  # ③ 队列为空 → fresh inference
    applied = robot.apply(action)
    recorder.add(
        images, state, applied,
        intervention_state=int(HILState.RELEASE),  # ④ 记录唯一一帧 RELEASE
        source="policy_release",
        policy_action=np.asarray(action, dtype=np.float32),
    )
    for _ in range(sim.cfg.record_every):        # 保持一个 10 Hz 周期（120/10 = 12 个物理步）
        sim.task.update((0.785 + float(applied[15])) / 0.785)
        sim.step(True)
    controller.after_frame()                     # ⑤ 立即回到 POLICY
    return applied
```

`collect_rollouts.run_automatic_correction()` 把这几步串起来，并在末尾断言 `RELEASE` 只持续一帧：

```python
    controller.start()
    expert = AutoExpert(sim, robot, recorder=recorder, intervention=True)
    try:
        segment = expert.correct(color)
    except IKError as exc:
        controller.abort()
        runtime.reset()
        return None, exc

    controller.finish()
    execute_release_frame(sim, robot, recorder, runtime, str(task), controller)
    if controller.state is not HILState.POLICY:
        raise AssertionError("RELEASE must last exactly one frame")
    return segment, None
```

**案例**：一条 episode 的 `intervention_state` 序列可能是

```text
帧号:   0 1 2 ... 79 80 81 ... 95 96 97 98 ...
状态:   0 0 0 ...  0  1  1 ...  1  2  0  0 ...
来源:   policy ...    auto_correction:pregrasp  policy_release  policy ...
```

其中 `1` 段的 `policy_action` 全为 0；恰好一个 `2`；`dataset.summarize()` 统计的 `release_frames` 应等于纠正片段数。

> ⚠️ **注意**：如果 `RELEASE` 之前不调用 `runtime.reset()`，队列里剩下的旧动作仍基于纠正前的观察，机器人会被“拉回”错误轨迹——这是“纠正结束后立即再次失败”最常见的原因。

### 3.4 自动纠正如何触发

原始的人机协同流程通常由人观察机器人、按键接管、遥操作纠正并在合适时机释放。本章把这些环节改写为仿真中可重复的自动机制，触发条件分为三类：

- **动作安全异常**：策略输出出现 NaN、Inf、形状错误，或机械臂单步关节跳变量过大；
- **推理异常**：模型服务超时、通信错误或没有返回可执行动作；
- **任务停滞**：一段时间内物块到盒子的平面距离没有明显下降，或右臂关节长期几乎不动。

任务进度与关节运动分别计时。这样可以识别“机械臂一直在动，但物块完全没有接近目标”的无效循环，也可以识别“策略卡死不动”的停滞。默认进度耐心为 80 个记录帧，运动耐心为 35 个记录帧；位置进步阈值和关节运动阈值分别独立设置。

**对应代码**：安全检查在 `code/code_chapter15/policy_client.py`：

```python
def unsafe_action(
    action: np.ndarray, state: np.ndarray, max_joint_jump: float = 0.65
) -> bool:
    action = np.asarray(action, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        return True                                        # 形状错误或 NaN/Inf
    return bool(np.max(np.abs(action[:14] - state[:14])) > max_joint_jump)  # 14 个手臂关节单步跳变 > 0.65 rad
```

双计时停滞检测在 `code/code_chapter15/hil.py`：

```python
@dataclass
class ProgressDetector:
    progress_patience: int = 80
    motion_patience: int = 35
    goal_epsilon: float = 0.008      # 物块到盒子平面距离需比历史最好再近 8 mm 才算进步
    joint_epsilon: float = 0.025     # 右臂 7 关节变化范数超过 0.025 rad 才算“在动”

    def observe(self, goal_distance: float, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32)
        goal_distance = float(goal_distance)
        progressed = goal_distance < self.best_goal_distance - self.goal_epsilon
        moved = (
            np.linalg.norm(state[7:14] - self.previous_state[7:14]) > self.joint_epsilon
        )
        if progressed:
            self.best_goal_distance = goal_distance
            self.frames_without_goal_progress = 0
        else:
            self.frames_without_goal_progress += 1
        if moved:
            self.frames_without_joint_motion = 0
        else:
            self.frames_without_joint_motion += 1
        self.previous_state = state.copy()
        return (
            self.frames_without_goal_progress >= self.progress_patience
            or self.frames_without_joint_motion >= self.motion_patience
        )
```

代码解读：两个计数器互不重置——关节在动不会清零“任务无进展”计数，这正是为了捕捉“机械臂一直乱动但物块没有接近盒子”。`goal_distance` 来自 `simulation.goal_distance()`，是物块与盒子在水平面（x、z 两轴）上的距离。

**案例**：

1. 若策略输出右臂第 2 关节目标 1.20 rad，而当前为 0.40 rad，跳变 0.80 rad > 0.65 rad，`unsafe_action` 返回真，立刻触发纠正。
2. 若机械臂每帧都在小幅摆动（关节变化 0.05 rad > 0.025），但物块到盒子距离 80 帧内都没比最好值近 8 mm，则在第 80 个记录帧触发“任务停滞”。按 10 Hz 记录频率，这相当于约 8 秒。
3. 若机械臂完全停住，第 35 个记录帧（约 3.5 秒）就会触发“运动停滞”。

### 3.5 自动专家：分段纠正与 IK 先规划后执行

自动触发并不表示专家立即播放完整抓取示教。系统先判断当前处于哪一阶段，再选择最短的恢复片段：

- `pregrasp`：末端离预抓取位置较远，只移动到安全预抓取位；
- `grasp_and_lift`：已经接近物块，完成下降、闭爪和抬升；
- `place_and_release`：物块已经被抓起，或已接近盒内，完成移动、放置和开爪。

这种分段设计保留了策略在纠正前后的自主行为，不会每次触发都让专家代替策略完成整个任务。对于 Value 来说，它也能看到“错误状态—恢复动作—恢复后的策略动作”这一完整因果链。

自动专家根据物块位置、夹爪闭合度、物块高度、末端是否接近预抓取点和物块是否进入盒子来选择片段。右臂目标点通过数值 IK 求解，并依次尝试当前关节、home 姿态、当前与 home 的中点以及关节范围中心等多个初值。

更重要的是，纠正片段会在真正进入 `ACTIVE` 动作记录前完成所需路径点规划。如果所有初值都不能得到满足误差阈值的 IK 解，本次纠正被安全中止，episode 作为失败经验保存，而不是伪造一段无效专家动作或错误的 `RELEASE` 帧。采集任务也不会因为单个 episode 的 IK 失败而丢失前面已经获得的数据。

**对应代码**：`code/code_chapter15/auto_expert.py`：

```python
@dataclass(frozen=True)
class CorrectionState:
    block_lifted: bool
    gripper_closed: bool
    inside_box: bool
    near_pregrasp: bool


def choose_segment(s):
    if s.inside_box or (s.block_lifted and s.gripper_closed):
        return "place_and_release"
    return "grasp_and_lift" if s.near_pregrasp else "pregrasp"

class AutoExpert:
    # ...
    @staticmethod
    def _ik_seeds(seed):
        """Use a few fixed initial guesses to avoid numerical IK local minima."""
        current = np.clip(np.asarray(seed, float), ARM_LOWER_7, ARM_UPPER_7)
        home = HOME_ARMS_14[7:14]
        center = (ARM_LOWER_7 + ARM_UPPER_7) / 2
        candidates = (current, home, (current + home) / 2, center)   # 最多 4 个初值
        # ... 去重
        return unique

    def solve(self, point, seed):
        answers = [self.ik.solve(point, guess) for guess in self._ik_seeds(seed)]
        best = min(answers, key=lambda answer: answer.error)
        if not best.success and best.error > IK_TOLERANCE:          # IK_TOLERANCE = 0.012
            raise IKError(point, best.error, len(answers))
        return best.joints

    def correct(self, color):
        segment = choose_segment(self.state(color))
        self.source = f"auto_correction:{segment}"
        above, grasp, above_box, place = self.targets(color)
        seed = self.robot.state()[7:14]
        if segment == "pregrasp":
            qa = self.solve(above, seed)            # 先求解，再移动
            self.move_right(seed, -0.785, 0.25)
            self.move_right(qa, -0.785, 0.75)
        elif segment == "grasp_and_lift":
            qa = self.solve(above, seed)
            qg = self.solve(grasp, qa)              # 两个路径点都求出来才开始动
            self.move_right(qa, -0.785, 0.4)
            self.move_right(qg, -0.785, 0.6)
            self.move_right(qg, 0, 0.5)             # 闭爪
            self.move_right(qa, 0, 0.8)             # 抬升
        else:
            qa = self.solve(above_box, seed)
            qp = self.solve(place, qa)
            self.move_right(qa, 0, 0.9)
            self.move_right(qp, 0, 0.5)
            self.move_right(qp, -0.785, 0.4)        # 开爪
            self.move_right(qa, -0.785, 0.5)
        return segment
```

`state(color)` 的四个判据分别是：物块高于桌面一定距离（`block_lifted`）、右夹爪闭合比例 ≥ 0.45（`gripper_closed`）、`inside_box()`、末端到预抓取点距离 ≤ 0.11 m（`near_pregrasp`）。

**案例**（按 `choose_segment` 的判断顺序推演）：

| 场景 | lifted | closed | inside | near_pregrasp | 选中片段 |
|---|---|---|---|---|---|
| 策略在桌面上方乱晃，离物块远 | 否 | 否 | 否 | 否 | `pregrasp` |
| 末端已在物块上方但迟迟不下降 | 否 | 否 | 否 | 是 | `grasp_and_lift` |
| 已抓起物块但在盒子外停滞 | 是 | 是 | 否 | — | `place_and_release` |
| 物块已在盒内但夹爪未松开 | — | — | 是 | — | `place_and_release` |

> 💡 **要点**：`AutoExpert._step()` 在物理 120 Hz 下运行，但只在 `tick % record_every == 0` 时记录，因此纠正片段与策略帧一样以 10 Hz 写入数据；`move_right` 只改右臂 7 关节与右夹爪（索引 7–13 和 15），其余维度保持当前状态。

### 3.6 本章对经验采集流程的改进

本章不是把人工接管简单替换为一个完整脚本，而是加入了几项针对教学复现和批量采集的改进：

1. **自动触发而非等待人工按键。** 危险动作、推理错误、任务无进展和关节停滞都能进入统一恢复流程。
2. **局部纠正而非整段接管。** 专家根据任务状态选择最短恢复片段，尽快把控制权还给策略。
3. **交接时强制重新推理。** 清空旧动作块，只允许一帧 `RELEASE`，消除纠正前动作队列的污染。
4. **IK 先规划后执行。** 纠正失败不会产生半段专家轨迹，也不会破坏状态机语义。
5. **失败经验完整保留。** 自动纠正后仍可能失败，所有成功与失败 episode 都进入 Value 数据，而不是只保存漂亮结果。
6. **训练与评测彻底隔离。** 纠正器仅为采集提供恢复和探索能力，最终结果必须由无干预评测给出。

这些改进让整个训练过程可以在 Isaac Sim 中重复运行，也让每一帧的来源、动作和控制权都有明确含义。

### 3.7 数据合同：每一帧保存什么

每个 NPZ episode 的逐帧字段包括：

`head_image`、`left_image`、`right_image`、`state[16]`、`action[16]`、`policy_action[16]`、`intervention_state`、`is_intervention`、`episode_index`、`frame_index`、`task_index`、`collector_policy_id` 和 `source`。

episode 级元数据还包括任务文本、目标颜色、成功标记、停止原因、纠正片段列表，以及是否进入 SFT 和 Value 训练。

**对应代码**：`code/code_chapter15/dataset.py` 的写入与读取校验：

```python
    def add(self, images, state, action, *, intervention_state: int, source: str, policy_action=None):
        # ...
        active = int(intervention_state) == 1
        if policy_action is None:
            policy_action = np.zeros(ACTION_DIM, np.float32) if active else action  # ACTIVE 时策略动作为全零
        values = {
            # ... 三路图像、state、action
            "policy_action": np.asarray(policy_action, np.float32).reshape(ACTION_DIM),
            "intervention_state": np.int8(intervention_state),
            "is_intervention": np.int8(active),          # 只有 ACTIVE 才算干预
            # ... collector_policy_id / episode_index / frame_index / task_index / source
        }

def load_episode(path: Path) -> dict[str, np.ndarray]:
    # ...
    states = ep["intervention_state"].astype(int)
    intervention = ep["is_intervention"].astype(bool)
    if not np.array_equal(intervention, states == 1):
        raise ValueError(f"{path}: intervention flags disagree")
    if np.any(ep["policy_action"][states == 1] != 0):
        raise ValueError(f"{path}: ACTIVE policy_action must be zero")
    return ep
```

必须保持以下约束：

1. `ACTIVE` 帧由自动专家产生，`policy_action` 必须全零；
2. `RELEASE` 必须恰好一帧，并使用纠正后的观察重新推理；
3. `state` 与 `action` 固定为 16 维，动作是全绝对目标；
4. `norm_stats.json` 必须来自原生 16 维数据，而不是补零后的 32 维；
5. 优势阈值按 `task_index` 分别计算；
6. 自动纠正帧强制为 positive；
7. 条件 dropout 只删除文本后缀，不删除样本、不翻转标签；
8. 最终评测不允许调用自动专家。

> ⚠️ **注意**：“是否成功”（episode 级 `success`）、“是否纠正”（帧级 `is_intervention`）、“是否为高优势”（帧级 `acp_indicator`）是三个不同概念，不能压成一个标签。

### 3.8 累计数据为何重要

第 $k$ 轮训练的数据集合可以写成：

$$
\mathcal D_k=
\mathcal D_{\mathrm{demo}}
\cup\mathcal D_{\mathrm{rollout}}^{(1)}
\cup\cdots\cup
\mathcal D_{\mathrm{rollout}}^{(k)}
$$

其中包含：

- 成功脚本示教；
- 当前策略和旧策略的自主动作；
- 自动纠正动作；
- 纠正后成功的 episode；
- 纠正后仍失败或因 IK、超时停止的 episode。

Value 在累计数据上重新训练，优势标签也根据更新后的 Value 重新计算。这样模型不仅学习“专家怎么做”，还会逐轮看到自己常见的失败模式以及如何恢复。与此同时，数据必须保留 `collector_policy_id`、`source`、episode 成功标记和控制状态，以便诊断某一轮是否被过多纠正帧或某一种失败模式主导。

**对应代码**：累计逻辑只有一行，位于 `code/code_chapter15/config.py`：

```python
def raw_dirs(round_id: int) -> list[Path]:
    return [demos_dir(), *(rollout_dir(i) for i in range(1, round_id + 1))]
```

**案例**：`raw_dirs(2)` 返回 `[data/raw/demonstrations, data/raw/round_001, data/raw/round_002]`。若示教每色 20 条、每轮每色 20 条，则第 2 轮 Value 会看到 $3\times20\times3=180$ 条 episode。episode 编号由 `round * 1_000_000 + color_index * episodes_per_color + episode_in_color` 生成，保证不同轮次不冲突。

---

## 第四部分 Value 学习：从成功标签到分布式价值估计

### 4.1 从成功标签构造逐帧 Value target

**直观理解**：每条 episode 只告诉我们一件事——“最后成功了没有、一共用了多少步”。从这一个事实出发，就可以倒推出“每一帧离终点还有多远”，于是得到逐帧的监督标签。

每个 episode 只需要一个最终成功或失败标签，就能构造逐帧监督目标。对应的稀疏奖励规则可以写成：

$$
r_t=
\begin{cases}
0, & t=T\text{ 且 episode 成功}\\
-C_{\mathrm{fail}}, & t=T\text{ 且 episode 失败}\\
-1, & \text{其他时刻}
\end{cases}
$$

这条规则表达了两个偏好：失败要受到明显惩罚；在都成功的情况下，使用更少步骤完成更好。设某个任务在累计数据中的最大 episode 长度为 $T_{\max}$，当前 episode 长度为 $T$，当前帧为 $t$（从 0 开始计），剩余步数为

$$
R_t=T-t-1
$$

失败惩罚设为

$$
C_{\mathrm{fail}}=T_{\max}
$$

把后续奖励累加起来，当前帧的未归一化回报为

$$
g_t=
\begin{cases}
-R_t, & \text{episode 成功}\\
-R_t-C_{\mathrm{fail}}, & \text{episode 失败}
\end{cases}
$$

最终 target 为

$$
y_t=\mathrm{clip}
\left(
\frac{g_t}{T_{\max}+C_{\mathrm{fail}}},-1,0
\right)
$$

这里固定的是**奖励规则和已完成轨迹的监督 target**，不是 Value 网络对所有状态输出同一个值。同一条轨迹中， $R_t$ 会随时间减少，因此每一帧的 target 也会变化。

**对应代码**：`code/code_chapter15/value_math.py`：

```python
def value_targets(
    episodes: np.ndarray,
    frames: np.ndarray,
    info: dict[int, EpisodeInfo],
    task_max: dict[int, int],
    c_fail_coef: float = 1.0,
) -> np.ndarray:
    # ...
    out = np.empty(len(episodes), np.float32)
    for i, (ep_id, frame) in enumerate(zip(episodes, frames, strict=True)):
        ep = info[int(ep_id)]
        maximum = int(task_max[ep.task_index])        # T_max：按任务（颜色）分别取
        remaining = ep.length - int(frame) - 1        # R_t = T - t - 1
        c_fail = maximum * c_fail_coef                # C_fail = T_max（config.VALUE_C_FAIL = 1.0）
        ret = -float(remaining) - (0.0 if ep.success else c_fail)   # g_t
        out[i] = np.clip(ret / (maximum + c_fail), -1.0, 0.0)       # y_t
    return out
```

代码解读：`c_fail_coef` 是系数，默认 1.0，所以 $C_{\mathrm{fail}}=1.0\times T_{\max}$；`task_max` 由 `dataset.task_max_lengths()` 从累计数据中逐任务统计。

**案例**：例如取

$$
T_{\max}=100,\qquad C_{\mathrm{fail}}=100
$$

某条 episode 一共有 60 帧，则第 0 帧剩余 59 步，第 30 帧剩余 29 步。成功与失败轨迹的 target 如下：

| 当前位置 | 剩余步数 | 成功轨迹 $y_t$ | 失败轨迹 $y_t$ |
|---|---:|---:|---:|
| 第 0 帧 | 59 | $-59/200=-0.295$ | $-(59+100)/200=-0.795$ |
| 第 30 帧 | 29 | $-29/200=-0.145$ | $-(29+100)/200=-0.645$ |
| 最后一帧 | 0 | $0$ | $-100/200=-0.5$ |

这组数字可以直接用代码验证（`value_targets` 对上述 6 帧输出 `[-0.295, -0.145, 0, -0.795, -0.645, -0.5]`）。

当 $C_{\mathrm{fail}}=T_{\max}$ 时，成功 target 大体位于 $(-0.5,0]$，失败 target 位于 $[-1,-0.5]$。严格来说，失败 episode 的最后一帧恰好可能等于 $-0.5$，因此不能把“失败恒小于 $-0.5$”理解成严格不等式。

这一构造让 Value target 同时包含三层信息：

- 最终成功还是失败；
- 当前距离 episode 结束还有多少步；
- 同样成功时，哪条轨迹完成得更快。

按每个任务自己的最大长度归一化，可以降低不同任务时长差异对 Value 的影响。但 $-0.5$ 只是在当前奖励设计下形成的成功/失败区间边界，后面的 positive/negative 并不是用 $V>-0.5$ 直接划分，而是由 50 步 Advantage 的任务内排序得到。

![图 15-7 成功与失败轨迹的逐帧 Value target](assets/value_target_curves.svg)

<p align="center"><b>图 15-7</b> 逐帧 target 同时编码最终成败、剩余步数与完成速度</p>

> ⚠️ **注意**： $T_{\max}$ 来自“累计数据中该任务的最长 episode”，所以新一轮数据加入后，同一条旧轨迹的 target 也可能变化。这也是每轮都要重新训练 Value、重新标注的原因之一。

### 4.2 Value 到底训练什么

一条 episode 结束后，程序已经知道它成功还是失败，也能直接计算这一条轨迹每一帧的 $y_t$。但真正部署时，机器人站在当前时刻并不知道未来：它不知道接下来会不会抓空、物块会不会掉落，也不知道当前姿态最终是否容易成功。因此还需要训练 Value 网络，把离线计算出的轨迹 target 变成可以从当前观察预测的函数：

$$
\hat V_\phi(o_t,\ell)
\approx
\mathbb E[G_t\mid o_t,\ell,\pi_{\mathrm{ref}}]
$$

Value 的输入只有当前三路图像、当前机器人状态和任务文本，不读取 episode 最终成败、未来动作、剩余长度或 episode 编号。它要学习的是：在当前状态下继续按照参考策略执行，平均来看能够获得多大回报。因而，训练数据中每一帧的 target 虽然已经确定，训练完成后的 Value 却仍是随观察变化的函数，而不是一个全局固定值。

这里必须把 `value_target` 和 `value_prediction` 分开理解。`value_targets()` 根据一条已经完成的轨迹构造逐帧 Monte Carlo 监督目标 $y_t$，它相当于训练阶段由完整结果给出的“参考答案”；`Pistar06Value` 产生的则是 $\hat V_t$，相当于模型只看当前信息后对未来结果作出的“现场估计”。同一帧的 target 在该数据集生成后是固定的，但不同观察、不同任务和不同训练轮次的 Value 预测并不是一个固定常数。随着累计经验增加，网络参数会重新训练，同一类状态的预测也可能被修正。

在红色物块入盒任务中，一个合理的 Value 变化可能是：

| 当前观察 | 可能的 Value 预测 |
|---|---:|
| 夹爪离物块很远且方向错误 | $-0.72$ |
| 夹爪已经移动到物块上方 | $-0.38$ |
| 物块被稳定抓起 | $-0.20$ |
| 物块移动到盒子上方 | $-0.08$ |
| 物块已经放入盒内并释放 | 接近 $0$ |

无论是在大规模离线机器人数据上预训练 Value，还是在目标任务的新 rollout 上重新训练或微调 Value，它的核心作用都相同：把稀疏的 episode 成败信号转成可以作用于中间状态的进度估计。前者提供更通用的任务进度先验，后者让 Value 适应当前策略、当前机器人和新出现的失败模式。本章代码采用后一种工程路径：每一轮读取截至当前的累计经验，独立训练 Value，再固定该轮 Value 为所有帧生成预测与 Advantage 标签；下一轮获得新经验后重新训练，而不是永久使用同一个评价器。

### 4.3 为什么使用分布式回归：201 个 bin 与 two-hot

**直观理解**：普通回归只让网络“报一个分数”；分布式回归让网络“给出每个分数段的概率”，就像天气预报给的是“降水概率”而不只是“下/不下”。

本章 Value 不直接用一个标量做 MSE 回归，而是把 $[-1,0]$ 均匀划分为 201 个 bin，bin 中心为

$$
c_b=-1+\frac{b-1}{200},\qquad b=1,\dots,201
$$

相邻中心间距为 $0.005$。真实 target 通常落在相邻两个 bin 之间，因此使用 two-hot 权重把监督分配给左右两个 bin：若 $c_j\le y<c_{j+1}$，则

$$
w_{j+1}=\frac{y-c_j}{c_{j+1}-c_j},\qquad w_j=1-w_{j+1}
$$

其余 bin 权重为 0，再以交叉熵训练：

$$
\mathcal L_{\mathrm{value}}=-\sum_{b=1}^{201}w_b\log p_\phi(b\mid o_t,\ell)
$$

推理时对 softmax 概率和 bin 中心求期望，得到连续 Value：

$$
\hat V(o_t,\ell)=
\sum_{b=1}^{201}p_\phi(b\mid o_t,\ell)c_b
$$

分布式表示能够保留一定的不确定性。例如相似的预抓取画面在数据中可能有时成功、有时失败，输出分布可以同时在多个 Value 区域保留概率，而不仅是强迫一个标量回归头立即给出唯一答案。最终用于 Advantage 的仍是这个分布的期望值。

![图 15-8 two-hot 编码与期望读出](assets/new_fig15_05_two_hot.svg)

<p align="center"><b>图 15-8</b> (a) target −0.296 被拆成相邻两个 bin 的 0.2/0.8 权重；(b) 不确定状态的预测分布在 −0.7 与 −0.1 各有一半概率，期望为 −0.4</p>

**对应代码**：`code/code_chapter15/value_math.py` 与 `value_model.py`：

```python
def bin_centers(count: int = 201, low: float = -1.0, high: float = 0.0) -> np.ndarray:
    return np.linspace(low, high, count, dtype=np.float32)


def two_hot(values: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
    centers = bin_centers() if centers is None else np.asarray(centers, np.float32)
    values = np.clip(np.asarray(values, np.float32).reshape(-1), centers[0], centers[-1])
    scaled = (values - centers[0]) / (centers[1] - centers[0])   # 连续的 bin 下标
    lo = np.floor(scaled).astype(np.int64)                       # 左邻 bin j
    hi = np.minimum(lo + 1, len(centers) - 1)                    # 右邻 bin j+1
    hi_w = np.clip(scaled - lo, 0, 1)                            # w_{j+1}
    lo_w = 1 - hi_w                                              # w_j
    result = np.zeros((len(values), len(centers)), np.float32)
    rows = np.arange(len(values))
    np.add.at(result, (rows, lo), lo_w)
    np.add.at(result, (rows, hi), hi_w)
    return result
```

```python
def two_hot_cross_entropy(logits, targets):
    return -(targets.to(logits.dtype) * F.log_softmax(logits, -1)).sum(-1).mean()


def expected_from_logits(logits, low=-1.0, high=0.0):
    return (
        logits.softmax(-1)
        * torch.linspace(low, high, logits.shape[-1], device=logits.device)
    ).sum(-1)
```

**案例**：取 $y=-0.296$。`scaled = (-0.296+1)/0.005 = 140.8`，所以 `lo=140`（ $c=-0.300$）、`hi=141`（ $c=-0.295$），权重为 $0.2$ 与 $0.8$。验证： $0.2\times(-0.300)+0.8\times(-0.295)=-0.06-0.236=-0.296$，two-hot 的期望恰好还原 target。

### 4.4 Value 网络结构

当前 Value 模型使用三路图像和语言—状态提示：

- 视觉编码器为 SigLIP SO400M（`google/siglip-so400m-patch14-384`）；
- 语言编码器为本地 Gemma 3 270M（`checkpoints/gemma-3-270m`）；
- 三路图像特征先分别编码，再进行相机维度平均；
- 视觉与语言特征投影到 512 维后拼接；
- 输出 201 维 Value logits。

16 维机器人状态先根据训练集的 1% 和 99% 分位数映射到 $[-1,1]$，再量化为 256 个离散区间，并补到 32 个状态 token。这样 Value 既能利用图像中的物块和夹爪关系，也能读取关节与夹爪数值。

![图 15-9 Pistar06Value 网络结构](assets/new_fig15_06_value_model.svg)

<p align="center"><b>图 15-9</b> 视觉分支（SigLIP）与语言—状态分支（Gemma 3 270M）分别投影到 512 维，拼接后经 LayerNorm 与 MLP 输出 201 个 logits</p>

**对应代码**：前向传播，`code/code_chapter15/value_model.py`：

```python
    def forward(self, input_ids, attention_mask, images):
        # images: B,N,C,H,W uint8/float
        b, n = images.shape[:2]
        x = images.reshape(b * n, *images.shape[2:]).float()      # 三路相机展平到批次维
        x = x / 255 if x.max() > 1 else x
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x, self.image_size, mode="bilinear", align_corners=False)
        x = (x - self.image_mean.to(x)) / (self.image_std.to(x))
        x = x.to(self.model_dtype)
        with torch.no_grad() if self.cfg.freeze_vision else nullcontext():
            vf = self._image_features(x)                           # SigLIP 图像特征
        with torch.no_grad() if self.cfg.freeze_language else nullcontext():
            out = self.language(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = out.last_hidden_state
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            lf = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)  # 有效 token 平均
        image = self.image_projector(vf.float()).view(b, n, -1).mean(1)   # 512 维，相机维平均
        language = self.language_projector(lf.float())                    # 512 维
        return self.head(self.norm(torch.cat([image, language], -1)))     # → 201 logits
```

状态离散化与 prompt 构造，`code/code_chapter15/train_value.py`：

```python
        state = ep["state"][j]
        normalized = np.clip(
            2 * (state - self.q01) / np.maximum(self.q99 - self.q01, 1e-6) - 1, -1, 1
        )                                                     # 1%/99% 分位数映射到 [-1, 1]
        q = np.digitize(normalized, np.linspace(-1, 1, 257, dtype=np.float32)[:-1]) - 1  # 256 档：0..255
        q = np.pad(q, (0, 32 - len(q)))                       # 16 维补到 32 个 token
        prompt = f"Task: {scalar(ep['task'])}, State: {' '.join(map(str, q.tolist()))}\nValue: "
```

**案例**：某关节归一化后为 $0.5$，`np.digitize` 得到档位 192；归一化值 $-1.0$、 $0.0$、 $1.0$ 分别对应 0、128、255。于是 prompt 形如 `Task: Pick up the red block and place it into the empty box., State: 128 192 ... 0 0\nValue: `（后 16 个为补零）。

> ⚠️ **注意**：`state_q01` / `state_q99` 会写入 Value checkpoint 的 `value_config.json`；`label_advantages.py` 推理时必须读回同一套分位数，否则离散 token 的含义会错位。

### 4.5 Value 训练循环

**对应代码**：`code/code_chapter15/train_value.py` 的主循环（删减）：

```python
    paths = episode_paths(raw_dirs(a.round))          # 示教 + 第 1..round 轮全部 episode
    data = Frames(paths)
    # ...
    optim = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad),
        lr=preset.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=preset.weight_decay,
    )
    scheduler = cosine_with_warmup(optim, preset, a.steps)   # 500 步 warmup，余弦退火到 1e-6
    for step in range(1, a.steps + 1):
        # ... 取 batch
        logits = model(tok["input_ids"].to(device), tok["attention_mask"].to(device), images.to(device))
        soft = torch.from_numpy(two_hot(target.numpy())).to(device)
        loss = two_hot_cross_entropy(logits, soft) / a.grad_accumulation
        loss.backward()
        if step % a.grad_accumulation == 0:              # 梯度累积
            torch.nn.utils.clip_grad_norm_(model.parameters(), preset.grad_clip)   # 裁剪 10.0
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
        if step == 1 or step % 200 == 0:
            print(f"step={step} loss={...:.5f} value={...:.4f}", flush=True)
    model.save(output, {"round": a.round, "steps": a.steps,
                        "task_max_lengths": task_max_lengths(paths),
                        "state_q01": data.q01.tolist(), "state_q99": data.q99.tolist(),
                        "seed": preset.seed})
```

`ValueTrainPreset` 的默认值：8,000 steps、batch 64、学习率 5e-5、weight decay 1e-5、梯度裁剪 10、warmup 500、最终学习率 1e-6、融合维度 512、dropout 0.1。

**案例**：显存不足时使用 `--batch-size 8 --grad-accumulation 8`，等效 batch 为 $8\times8=64$，与默认一致；但注意 `--steps 8000` 指的是前向次数，参数更新次数变为 $8000/8=1000$ 次。

> 💡 **要点**：Value loss 是交叉熵，最小值并不是 0——当 target 落在两个 bin 之间时，two-hot 分布本身有熵。因此判断 Value 质量不能只看 loss 是否接近 0，要看第五部分介绍的 Advantage 分布检查。

---

## 第五部分 Advantage 与优势条件化策略更新

### 5.1 稠密奖励与 50 步 Advantage

有了逐帧 target，可以通过相邻 target 的差构造稠密奖励。对于同一 episode 中连续的两帧：

$$
r_t=y_t-y_{t+1}
$$

最后一帧没有下一帧可用时，直接使用该帧 target 作为终止奖励。随后先把未来 $N$ 步的奖励和尾部 Value 合在一起，得到动作片段后的局部结果估计：

$$
\hat Q_t^{(N)}=
\sum_{k=0}^{N-1}r_{t+k}
+\mathbf 1_{t+N\ \mathrm{存在}}\hat V(o_{t+N},\ell)
$$

再减去动作执行前的预期 Value：

$$
A_t^{(N)}=\hat Q_t^{(N)}-\hat V(o_t,\ell)
$$

本章固定 $N=50$，所以实际计算为：

$$
A_t^{(50)}=
\sum_{k=0}^{49}r_{t+k}
+\mathbf{1}_{t+50\ \mathrm{存在}}\hat V(o_{t+50},\ell)
-\hat V(o_t,\ell)
$$

实现中只有在帧号严格连续、仍属于同一个 episode 且第 50 步确实存在时才进行 bootstrap，绝不会跨 episode 或跨缺失帧。若轨迹在 50 步内结束，尾部 Value 按 0 处理。

**为什么 $N=50$？** 50 正好等于 π0.5 的动作 horizon（`PI05_HORIZON = PI05_ACTION_STEPS = 50`），也就是策略一次推理输出并执行的动作块长度。用 50 步 Advantage，相当于评价“一整个动作块”的效果，与策略的决策粒度一致。按 10 Hz 记录，50 帧约 5 秒。

**对应代码**：`code/code_chapter15/value_math.py`：

```python
def dense_rewards(targets, episodes, frames):
    targets, episodes, frames = map(np.asarray, (targets, episodes, frames))
    out = np.empty(len(targets), np.float32)
    for i in range(len(targets)):
        contiguous = (
            i + 1 < len(targets)
            and episodes[i + 1] == episodes[i]
            and frames[i + 1] == frames[i] + 1
        )
        out[i] = targets[i] - targets[i + 1] if contiguous else targets[i]   # r_t = y_t - y_{t+1}；末帧 r = y_T
    return out


def n_step_advantage(rewards, values, episodes, frames, n_step: int = 50):
    # ...
    out = np.zeros(len(rewards), np.float32)
    for i in range(len(rewards)):
        total = 0.0
        steps = 0
        j = i
        while (                                         # 累加最多 n_step 个同 episode、帧号连续的奖励
            steps < n_step
            and j < len(rewards)
            and episodes[j] == episodes[i]
            and frames[j] == frames[i] + steps
        ):
            total += float(rewards[j])
            steps += 1
            j += 1
        bootstrap = (                                   # 只有第 n_step 帧确实存在才 bootstrap
            float(values[j])
            if steps == n_step
            and j < len(values)
            and episodes[j] == episodes[i]
            and frames[j] == frames[i] + n_step
            else 0.0
        )
        out[i] = total + bootstrap - float(values[i])   # A = Σr + V(t+N) - V(t)
    return out
```

**案例（可直接用代码验证）**：取一条 4 帧成功 episode， $T_{\max}=4$，则 $y=(-0.375,-0.25,-0.125,0)$，`dense_rewards` 得到 $r=(-0.125,-0.125,-0.125,0)$。设 Value 预测为 $\hat V=(-0.40,-0.30,-0.10,-0.05)$，取 $N=2$：

| $t$ | $r_t+r_{t+1}$ | bootstrap $\hat V_{t+2}$ | $-\hat V_t$ | $A_t^{(2)}$ |
|---:|---:|---:|---:|---:|
| 0 | $-0.25$ | $-0.10$ | $+0.40$ | $+0.05$ |
| 1 | $-0.125$ | $-0.05$ | $+0.30$ | $\approx 0$ |
| 2 | $-0.125$（只剩 2 帧） | 0（第 4 帧不存在） | $+0.10$ | $-0.025$ |
| 3 | $0$（只剩 1 帧） | 0 | $+0.05$ | $+0.05$ |

注意 $t=2$：虽然累加了两帧奖励，但 $t+2=4$ 超出 episode，因此不 bootstrap。

### 5.2 必须理解的数学边界：Value 过拟合会让 Advantage 塌缩

由于

$$
\sum_{k=0}^{N-1}(y_{t+k}-y_{t+k+1})=y_t-y_{t+N}
$$

如果 Value 把每一条训练轨迹逐帧死记住，达到 $\hat V(o_t)=y_t$，那么

$$
A_t^{(N)}
=y_t-y_{t+N}+y_{t+N}-y_t
=0
$$

所有 Advantage 都会塌缩到 0 附近。因此 Value 的目标不是成为“按 episode 编号查询 target 的表”，而是学习当前观察下的条件期望。同一个或相似的预抓取状态，在数据中可能因为后续动作不同而成功、失败或被纠正；Value 应学习这些结果的平均基准，具体动作片段的实际结果才能相对基准形成正负 Advantage。

**案例**：在 5.1 的 4 帧例子中，把 $\hat V$ 换成 $y$ 本身再算，`n_step_advantage` 对全部 4 帧都输出 $0$——这就是塌缩。

这也意味着 Value 过拟合是实际风险。数据太少、网络容量过大或训练过久时，即使 Value loss 很低，Advantage 也可能只剩数值噪声。训练后除了检查 loss，还应检查：Value 是否随真实任务进度总体上升，失败和掉落附近是否下降，自动纠正后是否回升，以及 Advantage 的均值、标准差和直方图是否已经集中到接近 0 的极窄范围。

> ⚠️ **注意**：Value 越“准”（对训练集）不代表越“好用”。它必须在相似状态之间泛化、给出平均意义上的基准，Advantage 才有区分度。

### 5.3 从连续 Advantage 到正负条件

不同颜色任务分别计算自己的优势阈值。默认取每个 `task_index` 的 70% 分位点，使大约前 30% 的帧标为 positive：

$$
\epsilon_\ell=
Q_{0.70}\left(\{A_t\mid \text{task}=\ell\}\right)
$$

$$
I_t=\mathbf{1}\left[A_t\geq\epsilon_\ell\right]
$$

这里不能把 $-0.5$ 当作正负条件的分界线。 $-0.5$ 来自 Value target 的成功/失败惩罚设计，而 positive/negative 评价的是**动作相对于当前状态预期的改善程度**。一个 Value 很高的简单状态也可能执行出负 Advantage 动作；一个 Value 很低的困难状态也可能通过恢复动作得到高 Advantage。

实际标注也不会简单使用 $A>0$ 判 positive、 $A<0$ 判 negative，而是对同一任务的 Advantage 排序。这样即使某个任务整体较难、Advantage 普遍偏低，也能选出其中相对更好的动作；如果所有 Advantage 都非常接近 0，则应先排查 Value 是否过拟合，而不能因为分位数仍能机械地选出 top 30% 就认为标签一定有效。

按任务分别取阈值很重要，因为红、绿、蓝物块的初始位置、IK 难度和轨迹长度可能不同。如果把所有任务混在一起取一个全局阈值，较容易任务的普通动作可能压过较难任务中的优质动作。

所有 `ACTIVE` 纠正帧会强制标为 positive。这里的假设是自动专家提供的动作是当前状态下的可靠恢复动作，应该被策略作为可取行为学习。与此同时，纠正前导致停滞的自主动作仍按 Value 和 Advantage 正常得到 positive 或 negative 标签，并不会因为同一 episode 最终成功而全部变成正样本。

**对应代码**：`code/code_chapter15/value_math.py`：

```python
def acp_labels(task_indices, advantages, interventions,
               positive_ratio: float = 0.3, force_intervention_positive: bool = True):
    # ...
    thresholds = {
        int(t): float(np.quantile(advantages[task_indices == t], 1 - positive_ratio))  # 每个任务的 70% 分位点
        for t in np.unique(task_indices)
    }
    labels = np.array(
        [int(a >= thresholds[int(t)]) for t, a in zip(task_indices, advantages, strict=True)],
        np.int64,
    )
    if force_intervention_positive:
        labels[interventions.astype(np.float32) > 0.5] = 1     # ACTIVE 帧强制 positive
    return labels, thresholds
```

**案例（可直接用代码验证）**：某任务 10 帧的 Advantage 为

$$
(0.12,\,-0.05,\,0.30,\,0.02,\,-0.20,\,0.08,\,0.15,\,-0.01,\,0.05,\,0.25)
$$

第 5 帧（ $A=-0.20$）是 `ACTIVE` 纠正帧。NumPy 线性插值的 70% 分位点为 $\epsilon=0.129$，于是 $A\ge0.129$ 的 3 帧（0.30、0.15、0.25）为 positive；再加上强制 positive 的纠正帧，最终标签为

```text
[0 0 1 0 1 0 1 0 0 1]   thresholds = {0: 0.129}
```

实际 positive 比例为 40%，高于设定的 30%——这正说明纠正帧越多，positive 中“专家动作”的占比越大，需要在 `label_advantages.py` 打印的 `positive_ratio` 中留意。

### 5.4 批量标注脚本如何串起来

**对应代码**：`code/code_chapter15/label_advantages.py`（删减）：

```python
    model, meta = Pistar06Value.load(ckpt, device)
    q01 = np.asarray(meta["state_q01"]); q99 = np.asarray(meta["state_q99"])   # 复用训练时的分位数
    paths = episode_paths(raw_dirs(a.round))
    episodes = [load_episode(x) for x in paths]
    # ... 统计每个任务最大长度 maxima
    with torch.inference_mode():
        for path, ep in zip(paths, episodes, strict=True):
            # ... 分 batch 预测 preds = expected_from_logits(model(...))
            info = {eid: EpisodeInfo(eid, tid, n, bool(scalar(ep["success"])))}
            targets = value_targets(np.full(n, eid), np.arange(n), info, maxima)      # ① y_t
            rewards = dense_rewards(targets, np.full(n, eid), np.arange(n))           # ② r_t
            advantage = n_step_advantage(rewards, np.asarray(preds),                  # ③ A_t
                                         np.full(n, eid), np.arange(n), a.n_step)
            staged.append((path, ep, targets, np.asarray(preds, np.float32), advantage))
            all_task.extend([tid] * n); all_adv.extend(advantage); all_inter.extend(ep["is_intervention"])
    labels, thresholds = acp_labels(np.asarray(all_task), np.asarray(all_adv),        # ④ 全量统一取阈值
                                    np.asarray(all_inter), a.positive_ratio, True)
    for path, ep, targets, preds, adv in staged:
        payload = {**ep, "value_target": targets, "value_prediction": preds,
                   "advantage": adv, "acp_indicator": labels[cursor : cursor + n]}
        np.savez_compressed(out / path.name, **payload)                              # 写入 data/labeled/round_xxx
```

把公式对应到实际数组后，一帧数据会依次经历下面四步：

1. `value_targets()` 根据 episode 长度和成败生成 $y_t$。例如成功轨迹越接近终点，target 越接近 0；失败轨迹还会承担 $C_{fail}$ 惩罚。
2. `dense_rewards()` 计算 $r_t=y_t-y_{t+1}$。这样原本只在整条轨迹末端可知的结果被改写成逐步的时间代价与终局代价，同时保持累计回报一致。
3. `n_step_advantage()` 计算 $\sum_{k=0}^{49}r_{t+k}+\hat V_{t+50}-\hat V_t$。前半部分描述接下来 50 步实际取得的局部结果，最后减去当前 Value 基线，回答“这段动作是否比原先预期更好”。不足 50 步或已经到达 episode 末端时不再从别的轨迹 bootstrap。
4. `acp_labels()` 不用固定的 0 或 $-0.5$ 判断正负，而是在每个任务内部取 Advantage 前 30% 为 positive，其余为 negative；自动纠正产生的 `ACTIVE` 帧再被强制设为 positive。

因此，前文“五步示例”中的“当前估计 $-0.60$、执行后局部结果折算为 $-0.35$、Advantage 为 $+0.25$”，在代码中分别对应 `values[i]`、`total + bootstrap` 和输出数组 `advantage[i]`。代码默认只是把五步换成 50 步，判断逻辑完全相同。

所有 episode 的 Advantage 会先暂存在内存中，之后才由 `acp_labels()` 按 `task_index` 统一计算分位数阈值。这样同一任务的阈值来自整轮累计数据，而不是每个小 batch 或每条 episode 各算一次。最后脚本把 `value_target`、`value_prediction`、`advantage` 和 `acp_indicator` 写入新的 `data/labeled/round_xxx` 副本，不会原地修改原始 rollout。这一“原始数据只读、派生标签另存”的约束便于重新训练 Value 或改变 positive 比例后重复标注。

### 5.5 优势条件如何进入 π0.5

**直观理解**：同一道题的作文，有的贴“范文”标签，有的贴“反面教材”标签，学生两种都读、都模仿，但考试时老师只说“请按范文写”。

策略训练前，原始任务文本会附加精确的条件文本：

> 优势条件仅使用两个固定文本：`Advantage: positive` 与 `Advantage: negative`。

这段文本在 tokenization 之前写入 prompt。训练时以 0.3 的概率移除优势标签，仅保留原始任务文本。这个 dropout 不会删除样本，也不会把正标签翻成负标签；它让同一个模型同时学会有条件和无条件的动作分布，并减少模型对固定后缀的机械依赖。

**对应代码**：`code/code_chapter15/acp.py`：

```python
_TAG = re.compile(r"\s*Advantage:\s*(positive|negative)\s*$", re.I)

def clean_task(task: str) -> str:
    return _TAG.sub("", str(task)).strip()                    # 删除已有后缀，防止重复叠加

def tagged_task(task: str, positive: bool) -> str:
    return f"{clean_task(task)}\n{ACP_POSITIVE if positive else ACP_NEGATIVE}"
```

训练数据管线中的实际变换在 `code/code_chapter15/pi05_adapter.py`，它位于 `Inputs` 之后、`TokenizePrompt` 之前：

```python
    @dataclasses.dataclass(frozen=True)
    class ACPTransform(transforms.DataTransformFn):
        enabled: bool = True
        dropout: float = 0.3
        seed: int = 1000
        # ...
        def __call__(self, data):
            if not self.enabled:
                return data                                       # SFT 阶段不加条件
            raw = np.asarray(data["indicator"]).reshape(-1)
            if raw.size != 1 or int(raw[0]) not in (0, 1):
                raise ValueError("indicator must be scalar 0/1")
            # ... 惰性初始化 rng
            task = clean_task(data.get("prompt", ""))
            return {
                **data,
                "prompt": task
                if rng.random() < self.dropout                    # 30%：只保留原任务文本
                else tagged_task(task, bool(raw[0])),             # 70%：附加 positive / negative
            }
```

**案例**：红色任务一帧 `acp_indicator = 0`，训练时其 prompt 有 70% 概率为

```text
Pick up the red block and place it into the empty box.
Advantage: negative
```

有 30% 概率为 `Pick up the red block and place it into the empty box.`。部署时 `policy_observation()` 始终发送 positive 版本。

策略训练仍使用全部动作数据与 π0.5 的 flow matching 目标。区别只是观察条件中增加了 $I_t$：

$$
\min_\theta
\mathbb{E}_{(o_t,a_t,I_t)\sim\mathcal D}
\left[
\mathcal L_{\mathrm{flow}}
\big(\pi_\theta(o_t,\ell,I_t),a_t\big)
\right]
$$

因此，第七步的“强化学习微调”在参数优化形式上与 SFT 很接近：两者使用相同的 π0.5、flow matching 损失、AdamW 和数据适配基础设施。强化学习信号主要发生在训练之前——成功、失败和完成速度先训练 Value，Value 再产生 Advantage 和二值条件，最后由监督式 flow matching 把条件行为写入策略。

两者的差异集中在数据语义：

| 对比项 | 成功示教 SFT | 优势条件策略更新 |
|---|---|---|
| 主要数据 | 经过筛选的成功示教 | 累计示教、成功/失败 rollout、自主动作与纠正动作 |
| 是否使用 Value | 否 | Value 已在训练前生成 Advantage 标签 |
| prompt | 原始任务文本 | 原始任务文本加 positive/negative，或因 dropout 暂时无条件 |
| 动作损失 | flow matching | flow matching |
| 常用初始化 | π0.5 基础权重 | SFT checkpoint 或上一轮策略 |
| 推理方式 | 普通任务文本 | 固定请求 positive 条件 |

negative 样本不是通过一个“反向损失”直接从模型中减掉。模型仍然在 negative 条件下拟合这些动作，只是把较好和较差行为放进不同条件分支。推理时始终使用 positive 条件，相当于要求模型从训练数据中选择高质量行为分支。本章服务端使用 10 个 flow sampling steps，不启用 classifier-free guidance（CFG），策略选择主要由训练时的任务内优势标签完成。

![图 15-10 从 Advantage 计算到优势条件化策略更新](assets/advantage_conditioned_update.svg)

<p align="center"><b>图 15-10</b> Value 提供状态基准，Advantage 评价动作片段，条件文本将质量信息写入策略</p>

> 💡 **要点**：为什么“学 negative”反而有用？如果把差动作直接丢掉，模型仍可能在相似状态下自发生成它们；把它们显式归入 negative 分支，相当于告诉模型“这种动作属于另一种风格”，positive 分支因此更纯净。

### 5.6 策略训练配置与数据合同

**对应代码**：`code/code_chapter15/pi05_adapter.py` 中的模型与训练配置：

```python
def model_config(lora=False):
    from openpi.models import pi0_config
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=PI05_PAD_DIM,             # 32：模型内部动作维度
        action_horizon=PI05_HORIZON,         # 50
        max_token_len=PI05_TOKEN_LENGTH,     # 200
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        image_augmentation=False,
    )

def train_config(name, repo_id, initial, stage, batch_size=32, steps=30000, lora=False, seed=1000, fsdp_devices=1):
    # ...
        lr_schedule=optimizer.CosineDecaySchedule(1000, 2.5e-5, 30000, 2.5e-6),   # warmup / 峰值 / 衰减步数 / 最终
        optimizer=optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01, clip_gradient_norm=1.0),
    # ...
        policy_metadata={
            "base_model": "pi0.5", "contract": "g2_pi05_v1", "action_transform": "all_absolute",
            "state_dim": 16, "action_dim": 16, "action_horizon": 50, "flow_steps": 10,
        },
```

本章采用 G2 专用的数据合同：

- 状态与动作都是 16 维；
- 顺序为左臂 7、右臂 7、左夹爪 1、右夹爪 1；
- 所有动作均为绝对目标；
- 归一化统计在原生 16 维上计算；
- 归一化之后才补零到模型的 32 维动作空间；
- 输出只截取前 16 维，不做额外 delta 恢复；
- 数据记录为 10 Hz，物理仿真为 120 Hz；
- 动作 horizon 为 50，服务端采用 10 个 flow sampling steps。

这与第十四章中“机械臂关节转 delta、夹爪保持 absolute”的适配方式不同。若把两个章节的动作变换混用，会造成重复差分、错误反归一化或大幅关节跳变。它也与 π0.6/π0.6\* 的模型输入、离散动作分支和高层子任务上下文不同，不能互换 tokenizer、模型配置或 checkpoint。

当前代码的关键默认参数如下：

| 类别 | 默认设置 |
|---|---|
| 仿真与数据 | 物理 120 Hz，数据 10 Hz，物块位置扰动 0.01 |
| π0.5 输入 | 三相机，统一缩放到 224×224，最大 token 长度 200 |
| π0.5 动作 | 原生 16D 全绝对动作，内部 32D，horizon 50，执行 50 步，10 个 flow steps |
| 策略训练 | 30,000 steps，batch 32，AdamW，峰值学习率 2.5e-5，warmup 1,000，最终学习率 2.5e-6，梯度裁剪 1.0 |
| Value 训练 | 201 bins，范围 $[-1,0]$，8,000 steps，batch 64，学习率 5e-5，梯度裁剪 10 |
| 优势标注 | 50-step，按任务 top 30% 为 positive，纠正帧强制 positive，条件 dropout 0.3 |

---

## 第六部分 本章代码、π0.6 / π\*0.6 与新增机制的具体差异

### 6.1 π0.6、π\*0.6 与 RECAP 是什么关系

需要先区分两个名称：**标准 π0.6** 是 π0.5 的模型架构与数据升级版本；**π0.6\***（论文中写作 π\*0.6）则是在该模型上加入二值 Advantage 条件、分布式 Value 学习和多轮经验迭代的策略版本。参考资料中的 π0.6 论文（`2511.14759v2.pdf`）把这套“采集经验与纠正—训练 Value—优势条件化提取策略”的方法称为 RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）。本章要复现的是后者所展示的经验学习方法，但不能直接把 π0.6/π0.6\* 当作现成基座使用。

把 RECAP 的思路按步骤展开，就是本章第三至第五部分的内容：

```text
RECAP 方法步骤                              本章对应
────────────────────────────────────────    ──────────────────────────────────────
1. 用当前策略在任务上采集自主经验            collect_rollouts.py（Isaac Sim）
2. 在错误发生时引入纠正（论文中为人类遥操作）  auto_expert.py 自动分段纠正
3. 以成功/失败与完成时间训练分布式 Value      value_math.value_targets + train_value.py
4. 用 Value 估计每段动作的优势并二值化        n_step_advantage + acp_labels
5. 把二值优势作为条件，监督训练策略           ACPTransform + flow matching
6. 推理时请求“高优势”条件；循环到第 1 步       policy_observation(positive=True)
```

### 6.2 为什么本章不直接使用 π0.6

当前 OpenPI 的模型定义、训练配置和公开 checkpoint 表只直接支持 π0、π0-FAST 与 π0.5，没有可以直接接入本章 G2 链路的 π0.6/π0.6\* 模型配置、训练实现和 checkpoint。更重要的是，π0.6 不只是“比 π0.5 多训练了几步”：它更换了 VLM 主干和动作专家，增加 FAST 离散动作与中间子任务预测，输入 token、参数树和训练目标都发生了变化。因此，即使得到一个名称为 π0.6 的权重目录，也不能通过修改 `--checkpoint` 路径让当前 `pi05_adapter.py` 正确加载。

本章选择 π0.5 有三个工程原因：第一，π0.5 的基础权重和 OpenPI 训练、推理代码可获得；第二，第十四章已经完成 G2 三相机、16 维状态动作和 WebSocket 服务适配；第三，使用可运行的基座后，可以把教学重点放在经验采集、Value、Advantage、条件策略训练和自动纠正，而不是把大量篇幅消耗在复现一个尚无公开工程入口的新模型上。

模型层面的差异如下：

| 对比项 | 标准 π0.6 / π0.6* | 本章自行实现的代码 |
|---|---|---|
| VLA 基座 | Gemma 3 4B VLM | π0.5，Gemma 2B VLM |
| 动作专家 | 约 860M 参数 | Gemma 300M 动作专家 |
| 连续动作 | flow matching | flow matching |
| 离散动作 | 同时预测 FAST 离散动作 token | 不训练 FAST 离散动作分支 |
| 高层推理 | 先预测中间子任务，再生成低层动作 | 直接使用给定任务文本生成动作 |
| 预训练数据 | 更大规模、多机器人、多模态混合 | 复用公开 π0.5 checkpoint，再用 G2 任务数据后训练 |
| 优势条件 | 标准 π0.6 不含；π0.6* 将二值条件放入动作生成上下文 | 在 tokenization 前把条件写成任务 prompt 后缀 |
| 动作平台 | 面向多种真实机器人 | G2 Omnipicker Isaac Sim 仿真 |
| 动作合同 | 随机器人平台定义 | 固定 16D 全绝对动作，模型内部补到 32D |

这意味着本章复现的是**经验驱动的训练闭环和优势条件策略提取方法**，不是 π0.6 全部预训练、架构与权重的同构复刻。

### 6.3 经验学习流程与策略更新的具体差异

两套方法都遵循“采集经验—训练 Value—估计 Advantage—条件化更新策略—再次采集”的主循环，但具体实现并不相同。下面的对比按照本章代码实际行为展开：

| 流程环节 | π0.6* 的论文实现 | 本章自行实现的代码 |
|---|---|---|
| 初始策略 | 经过大规模多机器人预训练的 π0.6 | OpenPI π0.5，经 G2 可选 SFT 后作为初始策略 |
| 在线数据 | 自主 rollout，并可由人类操作者在线纠正 | Isaac Sim 自主 rollout，由停滞与安全规则自动触发纠正 |
| 纠正动作 | 人类遥操作干预，主要处理明显错误与探索困难 | 分段脚本专家，根据当前状态选择预抓取、抓取抬升或放置释放 |
| 控制权记录 | 区分自主动作与干预动作 | 每帧保存 `POLICY / ACTIVE / RELEASE`，并强制单帧交接 |
| 奖励来源 | episode 成功/失败与完成时间 | episode 成功/失败与完成时间，归一化到 $[-1,0]$ |
| Value target | 剩余完成步数；失败叠加较大惩罚 | `value_targets()` 按任务最大长度计算同类 target |
| Value 模型 | 与 VLA 设计接近、使用较小 Gemma 3 多模态主干，并混入额外多模态数据 | 独立 PyTorch 模型，SigLIP SO400M + Gemma 3 270M + 201-bin 融合头，只读取本章累计机器人数据 |
| Advantage | 使用 Value 的多步估计产生二值最优性条件 | 固定 50-step；严格检查 episode 与 frame 连续性后 bootstrap |
| 正样本阈值 | 按任务设置阈值，不同任务可使用不同正样本比例 | 每个 `task_index` 取 70% 分位点，默认 top 30% 为 positive |
| 纠正标签 | 干预动作强制作为高优势样本 | 所有 `ACTIVE` 自动纠正帧强制 positive |
| 条件注入 | 在高层子任务之后、动作生成之前加入二值条件 | 在 tokenization 前追加 `Advantage: positive/negative` |
| 条件 dropout | 随机移除条件，以同时学习有条件与无条件策略 | `ACPTransform` 以 0.3 概率只移除后缀，不删除样本、不翻转标签 |
| 策略目标 | 优势条件化的 VLA 监督训练，可结合条件/无条件分支进行引导 | 保持 π0.5 flow matching 损失；推理只请求 positive 分支，CFG 关闭 |
| 数据聚合 | 新旧轮次经验持续加入训练集 | `raw_dirs(round_id)` 累计示教与截至当前轮的全部 rollout |
| 最终评测 | 无人工干预地评价更新后策略 | `evaluate.py` 完全不加载自动专家，异常与不安全动作直接判失败 |

从训练目标看，本章没有实现 PPO、REINFORCE 或显式 KL 正则化，也没有计算 flow matching 策略的精确动作对数概率。它先把连续 Advantage 压缩为可解释的二值条件，再继续使用 π0.5 原有的监督式 flow matching 目标。换句话说，Value 负责“评价哪些动作更值得模仿”，策略训练仍负责“在给定条件下生成这些动作”。这种拆分降低了训练复杂度，但其能力上限也受到 π0.5 基座、Value 精度和二值标签信息量的限制。

**补充：为什么条件 dropout 能支持“引导”？** 当模型同时学会有条件分布 $\pi_\theta(a\mid o,\ell,I)$ 和无条件分布 $\pi_\theta(a\mid o,\ell)$ 时，理论上可以像扩散模型的 classifier-free guidance 那样，把两者的差异放大，进一步偏向 positive 分支。本章 `serve_policy.py` 明确关闭了 CFG（启动时打印 `CFG=off`），只直接请求 positive 分支；dropout 在本章的作用主要是让模型不过度依赖后缀文本。

### 6.4 Value 的解耦设计

本章的 Value 与策略完全解耦。Value 不是从 π0.5 参数树中增加一个 head，而是由 `value_model.py` 独立构建：三路图像由 SigLIP 编码，任务和离散状态由 Gemma 3 270M 编码，视觉与语言投影后输出 201 个 Value bin。这样做便于在单独的 PyTorch 环境中训练和检查，但不会获得 π0.6\* 中 Value 与通用 VLA 预训练数据、表征结构更接近所带来的潜在优势。

| 方面 | 解耦设计的好处 | 代价 |
|---|---|---|
| 环境 | Value 用 PyTorch、策略用 JAX/OpenPI，互不干扰 | 需要维护两套依赖 |
| 调试 | 可单独检查 `value_prediction` 曲线 | 无法共享策略的视觉表征 |
| 迭代 | 每轮可独立重训 Value | 只能看到本章累计数据，泛化先验较弱 |

### 6.5 本章新增的自动纠正闭环

论文流程中的在线纠正通常依赖人类操作者发现错误并遥操作恢复。本章将其改造成适合 Isaac Sim 批量采集的自动系统，这部分是当前代码相对于原始流程最主要的工程扩展：

- 双计时停滞检测，同时监控任务进展与关节运动；
- 对 NaN、Inf、动作维度和单步关节跳变进行自动安全检查；
- 根据“预抓取、抓取抬升、放置释放”选择最短局部纠正片段；
- 使用多个 IK 初值，并在进入 `ACTIVE` 前完成所需路径点规划；
- 用 `POLICY / ACTIVE / RELEASE` 明确记录每帧控制权；
- `RELEASE` 只持续一帧，清空旧动作队列并强制 fresh inference；
- IK 失败时安全中止纠正，保存此前经验，不伪造交接帧；
- 使用仿真几何条件自动判断物块是否真正放入盒内；
- 支持中断续采，并为每一轮写入独立 `collector_policy_id`；
- 最终执行单色和连续三色无干预评测。

自动纠正的目的不是让脚本专家替模型完成任务，而是把策略带回可恢复区域，并生成“错误状态—局部恢复—重新自主执行”的训练片段。随着策略变好，纠正次数和纠正帧占比都应逐轮下降；如果成功率提高但纠正率没有下降，就不能证明策略本身已经学会恢复。

> 💡 **要点**：本章复现的是 RECAP 的“方法”，不是 π\*0.6 的“模型”。只要数据语义（谁执行、是否成功、优势高低）正确，同一套闭环可以迁移到其他可微调的 VLA 基座。

---

## 第七部分 代码结构与实现解析

### 7.1 文件组织

```text
code/code_chapter15/
├── config.py               # 全工程唯一配置：维度、horizon、Value bins、阈值、路径、训练预设
├── simulation.py           # Isaac Sim 场景、三色物块与盒子、success()/goal_distance()
├── robot.py                # G2 16D 状态读取与绝对动作应用
├── kinematics.py           # 右臂数值 IK（RightArmIK）
├── auto_expert.py          # 自动示教 demonstrate() 与分段纠正 correct()
├── hil.py                  # HILState 三态、HILController、ProgressDetector
├── rollout_core.py         # 统一的单帧记录与单帧 RELEASE 交接
├── policy_client.py        # WebSocket 客户端、50 步动作队列、unsafe_action()
├── collect_demos.py        # 采集自动成功示教（可选 SFT 数据）
├── collect_rollouts.py     # 采集一轮带自动纠正的策略经验
├── dataset.py              # NPZ episode 记录、加载与一致性校验
├── value_math.py           # target / two-hot / 稠密奖励 / N 步 Advantage / ACP 标签
├── value_model.py          # Pistar06Value 分布式 Value 网络
├── train_value.py          # 在累计数据上训练 Value
├── label_advantages.py     # Value 推理 + Advantage 计算 + 写出 labeled 副本
├── acp.py                  # 优势条件文本：clean_task / tagged_task / ACPPrompt
├── export_dataset.py       # NPZ → LeRobot 数据集
├── compute_norm_stats.py   # 原生 16D 归一化统计
├── pi05_adapter.py         # G2 与 π0.5 的唯一适配层（Inputs/Outputs/ACPTransform/配置）
├── openpi_runtime.py       # OpenPI 路径准备、checkpoint 元数据与 latest() 工具
├── train_policy.py         # SFT / ACP 两阶段策略训练入口
├── serve_policy.py         # 加载 checkpoint 并启动 WebSocket 策略服务
├── evaluate.py             # 无纠正评测：单色 + 连续三色
├── run_pipeline.py         # 打印/执行完整流程命令
├── setup_env.sh / run.sh / download_checkpoint.sh   # 环境检查、统一运行入口、下载 π0.5 base
├── pi05_assets/            # 各阶段 norm_stats.json
├── reports/                # evaluation_sft_50.json / evaluation_rl_round1_50.json
└── third_party/openpi      # OpenPI 源码快照
```

原始数据默认写入 `data/raw`，优势标注写入 `data/labeled`，LeRobot 数据写入 `data/lerobot`，策略与 Value 权重写入 `checkpoints`，最终报告写入 `reports`。所有路径都可以通过 `CHAPTER15_OUTPUT_ROOT` 切换到其他磁盘。

`code/code_chapter15` 按照“仿真执行—经验记录—价值学习—策略更新—独立评测”拆分：

| 模块 | 作用 |
|---|---|
| `config.py`、`simulation.py`、`robot.py` | 统一任务参数，构建 Isaac Sim 场景，提供 G2 的 16D 状态与绝对动作接口 |
| `kinematics.py`、`auto_expert.py` | 右臂数值 IK、自动示教和分段纠正 |
| `hil.py`、`rollout_core.py` | 停滞检测、三态控制权、统一动作记录和 `RELEASE` 交接 |
| `collect_demos.py`、`collect_rollouts.py` | 采集可选成功示教和带自动纠正的策略经验 |
| `dataset.py`、`export_dataset.py` | 保存可检查的 NPZ episode，并导出 LeRobot 数据集 |
| `value_math.py`、`value_model.py`、`train_value.py` | 构造 Value target、训练分布式 Value |
| `label_advantages.py`、`acp.py` | 计算逐帧 Advantage，生成优势条件和训练 dropout |
| `compute_norm_stats.py`、`pi05_adapter.py` | 计算原生 16D 统计量，完成 G2 与 π0.5 的唯一适配 |
| `train_policy.py`、`serve_policy.py`、`policy_client.py` | 训练、服务和调用 π0.5 策略 |
| `evaluate.py`、`run_pipeline.py` | 无纠正最终评测与流水线组织 |

### 7.2 模块调用关系

![图 15-11 模块调用关系与数据流](assets/new_fig15_07_code_dataflow.svg)

<p align="center"><b>图 15-11</b> 从策略服务、经验采集到 Value、标注、导出、训练与评测的模块调用关系；新 checkpoint 回到策略服务开始下一轮</p>

### 7.3 关键流程逐段解读

前文讲到的经验采集、自动纠正、Value 学习、Advantage 标注和条件策略更新，在工程上不是由一个大脚本一次完成，而是通过若干具有明确输入输出合同的模块串联。阅读代码时，建议始终沿着“配置约束 → 在线采集 → 原始 episode → Value → 标签 → LeRobot 数据 → π0.5 更新 → 无干预评测”这条数据流向下看，而不要只从某个训练入口开始。

**1. `config.py`：把论文超参数变成全工程唯一配置。**

`config.py` 是本章的数据合同和训练合同来源。`STATE_DIM` 与 `ACTION_DIM` 固定 G2 原生状态、动作为 16 维，`PI05_PAD_DIM` 指定模型内部使用 32 维，`PI05_HORIZON` 与 `PI05_ACTION_STEPS` 把动作块长度和默认执行长度都设为 50，`PI05_FLOW_STEPS` 则规定推理时进行 10 次 flow 积分。Value 一侧的 201 个离散 bin、 $[-1,0]$ 范围、50 步 Advantage、任务内前 30% positive 和 0.3 条件 dropout 也都在这里统一定义。

```python
TASK_TEMPLATE = "Pick up the {color} block and place it into the empty box."
ACP_POSITIVE = "Advantage: positive"
ACP_NEGATIVE = "Advantage: negative"

STATE_DIM = ACTION_DIM = 16
PI05_PAD_DIM = 32
PI05_IMAGE_SIZE = 224
PI05_TOKEN_LENGTH = 200
PI05_HORIZON = PI05_ACTION_STEPS = 50
PI05_FLOW_STEPS = 10

VALUE_BINS = 201
VALUE_RANGE = (-1.0, 0.0)
VALUE_C_FAIL = 1.0
ADVANTAGE_N_STEP = 50
ACP_POSITIVE_RATIO = 0.30
ACP_DROPOUT = 0.30
SEED = 1000
```

这样设计的意义是避免不同脚本各自复制常量。例如，如果标注脚本按 50 步计算 Advantage，而策略数据适配层只导出 10 步动作块，训练语义就会悄悄错位；如果归一化脚本认为动作是 16 维，而模型服务直接按 32 维统计量反归一化，机器人会收到错误关节目标。`PolicyTrainPreset`、`ValueTrainPreset` 和路径辅助函数还统一了训练步数、学习率、数据轮次与 checkpoint 目录，因此修改实验配置时应先检查这里，而不是分别修改各入口文件。

**2. `collect_rollouts.py`、`hil.py` 与 `auto_expert.py`：实现策略执行和自动纠正。**

`collect_rollouts.py` 是一轮在线经验采集的总入口。主循环每帧先检查任务是否成功，再根据 `trigger` 决定继续调用策略还是进入 `run_automatic_correction()`。策略推理异常、动作安全检查失败，以及 `ProgressDetector.observe()` 判断任务或关节停滞，都会把 `trigger` 置为真。无论 episode 最终成功、超时还是纠正失败，已经记录的轨迹都会由 `EpisodeRecorder.save()` 保存，以保证 Value 同时看到成功和失败经验。

`hil.py` 把控制权显式写成 `POLICY=0`、`ACTIVE=1`、`RELEASE=2` 三种状态。`HILController.start()` 只允许从策略态进入纠正态，`finish()` 只允许从纠正态进入交接态，`after_frame()` 再把交接态恢复为策略态。状态机不是为了界面显示，而是为了让每一帧都能回答“这一动作究竟由谁产生”。`ProgressDetector` 同时维护两个互不重置的计时器：物块到目标的距离长期没有改善时触发任务停滞；右臂关节长期几乎不变化时触发运动停滞。这样可以区分“机械臂一直乱动但任务没进展”和“机械臂完全卡住”两种失败。

`policy_client.py` 中的 `RemotePolicy` 维护一个 50 步动作队列。队列为空时才请求新的动作块，正常情况下逐帧弹出动作；`reset()` 会丢弃尚未执行的旧动作。`unsafe_action()` 对动作维度、NaN、Inf 和最大手臂关节跳变进行检查，当前阈值为 0.65 rad。推理 observation 通过 `policy_observation()` 组织三路图像、16 维状态和任务文本，并固定请求 positive 分支。

```python
    def reset(self) -> None:
        """Discard stale chunk actions; RELEASE always calls this before inference."""
        self._queue.clear()

    def next_action(self, images, state, task: str, *, force_fresh: bool = False) -> np.ndarray:
        if force_fresh:
            self.reset()
        if not self._queue:
            return self.infer_fresh(images, state, task)   # 请求 50 步动作块并入队
        return self._queue.popleft().copy()               # 逐帧弹出
```

真正的自动恢复由 `auto_expert.py` 完成。`CorrectionState` 根据物块是否抬起、夹爪是否闭合、物块是否已在盒内和末端是否靠近预抓取点概括当前阶段，`choose_segment()` 据此只选择 `pregrasp`、`grasp_and_lift` 或 `place_and_release` 中最短的必要片段。`AutoExpert._ik_seeds()` 为同一目标依次准备当前关节、home 姿态、二者中点和关节范围中心等初值，`solve()` 选择误差最小的结果；全部初值都失败才抛出 `IKError`。`correct()` 会先求解所需路径点，再开始记录 `ACTIVE` 帧，因而不会出现“纠正动作执行一半后才发现后续 IK 不可达”的伪数据。

**3. `rollout_core.py`：保证策略与纠正器安全交接。**

`execute_recorded_action()` 统一完成四件事：采集观察、应用绝对动作、写入一帧数据、再把目标保持一个 10 Hz 数据周期。策略帧和纠正帧都经过同一个记录入口，因此图像、状态、实际应用动作和控制状态在时间上保持一致。

纠正结束后，`execute_release_frame()` 先调用 `runtime.reset()` 清空旧动作块，再从纠正后的新图像与新关节状态请求一次 fresh inference，并把该动作记录为唯一一帧 `RELEASE`。随后 `HILController.after_frame()` 立即返回 `POLICY`。这一实现对应前文的“单帧交接”原则：`RELEASE` 不是一段模糊的过渡时间，而是可测试、可验证的一帧；它既防止旧动作队列把机器人拉回纠正前的错误轨迹，也保留了策略恢复控制权的第一个真实动作。

**4. `dataset.py`、`export_dataset.py` 与 `compute_norm_stats.py`：把控制语义保存成训练数据。**

`EpisodeRecorder.add()` 负责逐帧写入三路图像、16 维观察状态、实际执行动作、策略原始动作、控制状态、数据来源和策略版本。`ACTIVE` 期间策略没有产生可用动作，因此 `policy_action` 必须为全零；普通策略帧和 `RELEASE` 帧则保存真实策略输出。`save()` 再加入成功标记、停止原因、纠正片段、episode 类型和训练用途。`load_episode()` 不只是读取 NPZ，还会检查帧数、状态与动作形状、干预标记一致性以及 `ACTIVE policy_action` 是否为零，使损坏的数据在进入训练前尽早失败。

`export_dataset.py` 将 NPZ 转换为 OpenPI 使用的 LeRobot 数据集。它保留三路相机、16 维状态和动作，同时把 `acp_indicator`、`is_intervention` 与 `policy_action` 放入辅助信息字段。SFT 阶段只选择允许用于 SFT 的成功示教；优势条件训练阶段则要求每帧已经具有合法的 0/1 `acp_indicator`。因此“是否成功”“是否纠正”“是否为高优势”是三个不同概念，不会被压成一个标签。

```python
        if stage == "sft":
            eligible = bool(scalar(episode.get("use_for_sft", False))) and bool(scalar(episode["success"]))
            if not eligible:
                continue
            indicators = np.ones(len(episode["state"]), dtype=np.int64)  # schema only; SFT adapter ignores it
        else:
            if "acp_indicator" not in episode:
                raise ValueError(f"{path} has no acp_indicator; run label_advantages.py first")
            indicators = np.asarray(episode["acp_indicator"], dtype=np.int64)
            # ... 检查形状与取值必须为 0/1
```

`compute_norm_stats.py` 在原生 16 维状态和动作上计算统计量。补到 32 维发生在归一化之后的模型变换中，而不是发生在统计阶段。这个顺序很关键：若先补 16 个恒为零的维度再统计，额外维度会产生退化分布，并使 checkpoint 的统计合同与实际机器人接口混在一起。

```python
    if state.shape[1:] != (16,) or action.shape[1:] != (16,):
        raise ValueError("normalization must be computed in native 16D")
```

**5. `value_math.py`、`value_model.py` 与 `train_value.py`：把 episode 结果变成逐帧 Value。**

`value_math.value_targets()` 是第 4.1 节公式的直接实现。它按任务读取最大 episode 长度，计算当前帧的剩余步数，并为失败 episode 叠加 `C_fail`，最后裁剪到 $[-1,0]$。`bin_centers()` 产生 201 个均匀 bin，`two_hot()` 把连续 target 分配给相邻两个 bin，`expected_value()` 或模型侧的 `expected_from_logits()` 再把概率分布还原为连续期望值。这几项函数共同对应“分布式 Value 回归”，而不是普通标量 MSE。

`value_model.Pistar06Value` 实现独立的多模态 Value 网络。三路图像先展平到批次维，经过视觉编码器后再按相机维平均；任务文本与离散状态 token 经过语言模型，并使用 attention mask 做有效 token 平均。视觉和语言特征分别投影到 512 维，拼接后经 LayerNorm 与 MLP 输出 201 个 logits。`two_hot_cross_entropy()` 对 two-hot 目标计算交叉熵，`save()` 和 `load()` 同时保存模型配置与本轮状态分位数，避免推理阶段使用另一套离散化尺度。

从输入合同看，Value 网络只接收当前三路图像、当前机器人状态和任务文本；它不会把 episode 的最终成功标签、未来动作、剩余轨迹长度或 episode 编号作为推理输入。成功与失败结果只用于离线生成监督 target。最终，`expected_from_logits()` 对 201 个 bin 的概率求期望，得到后续 Advantage 计算使用的 $\hat V(o_t,\ell)$。这正对应第 4.2 节所说的条件期望：模型需要从当前观察中学习“现在大概有多接近完成”，而不是在推理时偷看轨迹结局。

`train_value.py` 中的 `Frames` 会读取示教和截至当前轮的所有原始 episode，而不是只训练最新一轮。它先用全部状态计算 1% 与 99% 分位数，将每个 16 维状态缩放到 $[-1,1]$、量化成 256 档并补到 32 个离散 token，再和任务文本组成 Value prompt。训练循环使用 two-hot 交叉熵、AdamW、带 warmup 的余弦学习率和梯度裁剪。最终 checkpoint 保存 `state_q01`、`state_q99`、任务最大长度和轮次信息，供后续全量推理复用。

**6. `label_advantages.py` 与 `value_math.py`：从 Value 预测得到动作优劣标签。**

`label_advantages.py` 首先加载训练好的 Value，对每个 episode 的所有帧批量预测 $\hat V_t$；随后重新调用 `value_targets()` 得到监督 target，用 `dense_rewards()` 计算相邻 target 差，再通过 `n_step_advantage()` 计算 50 步 Advantage。`n_step_advantage()` 会检查 episode 编号和 frame 编号连续性，只有完整存在第 50 步时才使用 bootstrap Value，因而不会跨 episode 借用下一条轨迹的预测。所有 episode 的 Advantage 汇总后才统一由 `acp_labels()` 计算任务内阈值（详见 5.4 节）。

训练后不能只看 Value loss，还应把 `value_target` 与 `value_prediction` 放在同一条时间轴上检查：正常推进时预测是否总体上升，掉落、碰撞或停滞附近是否下降，自动纠正后是否恢复。还应统计 Advantage 的均值、标准差和直方图，并抽取 positive、negative 帧查看动作是否符合直觉。如果大多数 Advantage 都挤在 0 附近，top 30% 仍然会机械地选出一批 positive，但它们可能只是数值噪声；这通常提示 Value 过拟合逐帧 target、泛化不足或输入信息不够，而不是策略真的已经没有可区分的好坏动作。

**7. `acp.py`、`pi05_adapter.py` 与策略训练服务：把标签变成可执行策略。**

`acp.py` 只定义三件事：`clean_task()` 删除已有优势后缀，`tagged_task()` 添加精确的 positive 或 negative 文本，`ACPPrompt` 以给定概率只删除条件。它不会删除样本，也不会把标签随机翻转。因此 0.3 dropout 学到的是“有条件分支与无条件分支共享一个模型”，而不是制造带噪声的监督标签。

`pi05_adapter.py` 是 G2 数据与 π0.5 之间唯一允许存在的适配层。`Inputs` 把 LeRobot 字段重组为三相机、16 维状态、动作和 prompt；`ACPTransform` 必须在 tokenization 之前读取 0/1 indicator，先清理旧标签，再决定附加条件还是执行 dropout；`G2Data.create()` 随后依次完成图像缩放、归一化后的状态动作补零和 prompt tokenization。`Outputs` 只截取模型 32 维输出中的前 16 维。`model_config()` 明确选择 π0.5、Gemma 2B、Gemma 300M 动作专家和 50 步 horizon，`train_config()` 则绑定 AdamW、余弦退火、LoRA/全参数模式和 checkpoint 元数据。

```python
                data_transforms=transforms.Group(
                    inputs=[Inputs(), ACPTransform(self.acp, self.dropout, self.seed)],   # 先加条件
                    outputs=[Outputs()],                                                  # 输出截取前 16 维
                ),
                model_transforms=transforms.Group(
                    inputs=[
                        transforms.InjectDefaultPrompt(None),
                        transforms.ResizeImages(PI05_IMAGE_SIZE, PI05_IMAGE_SIZE),        # 224×224
                        transforms.PadStatesAndActions(model.action_dim),                 # 归一化后补到 32 维
                        transforms.TokenizePrompt(                                         # 最后才 tokenize
                            tokenizer_lib.PaligemmaTokenizer(model.max_token_len),
                            discrete_state_input=True,
                        ),
                    ]
                ),
```

`train_policy.py` 通过 `--stage sft` 与 `--stage acp` 区分初始行为克隆和优势条件更新，并由 `resolve_initial()` 确认本轮从哪个 checkpoint 开始。训练结束写出的 `chapter15_policy.json` 记录基础模型、数据合同、动作变换、初始权重和 LoRA/FSDP 信息。`serve_policy.py` 在启动时检查合同必须为 `g2_pi05_v1`、动作必须为 `all_absolute`、归一化统计必须为原生 16 维；随后固定 `num_steps=10`，不启用 CFG。`policy_client.policy_observation()` 在部署时始终附加 positive 条件，这对应“从同一个条件策略中选择高优势行为分支”。

```python
    info = metadata(checkpoint)
    if (
        info.get("contract") != "g2_pi05_v1"
        or info.get("action_transform") != "all_absolute"
    ):
        raise ValueError(f"incompatible checkpoint metadata: {info}")
    stats = normalize.load(norm_directory(checkpoint))
    if any(len(getattr(stats[key], "mean")) != 16 for key in ("state", "actions")):
        raise ValueError("checkpoint norm stats must be native 16D")
    cfg = runtime_config(bool(info.get("lora", False)), info)
    policy = policy_config.create_trained_policy(
        cfg, checkpoint, norm_stats=stats, sample_kwargs={"num_steps": 10},
    )
```

最后，`evaluate.py` 不导入也不调用 `AutoExpert`。不安全动作、服务异常和超时都会直接计为失败，报告中明确写入 `no_intervention: true`，并分别统计单色任务和随机顺序连续三色任务。`run_pipeline.py` 只负责按正确依赖顺序打印或执行上述阶段，同时区分 Isaac Sim Python 与 OpenPI 训练环境；它不把策略服务的生命周期隐藏在训练脚本里，因此每一轮究竟使用哪个 checkpoint 采集数据仍然可追踪。

```python
        if sim.task.success(color):
            return True, "success", frame
        images, state = sim.cameras.capture(), robot.state()
        try:
            action = runtime.next_action(images, state, task)
        except Exception as exc:
            # ...
            return False, f"policy_error:{type(exc).__name__}", frame   # 推理异常直接判失败
        if unsafe_action(action, state):
            return False, "unsafe_action", frame                        # 不触发纠正，直接判失败
```

连续三色任务中，`rng.shuffle(colors)` 随机打乱顺序，任一颜色失败即终止该 episode，只有 `completed == 3` 才算成功。

通过以上对应关系可以看出，本章所谓“强化学习微调”并不是某个单独损失函数，而是一组必须共同成立的工程语义：采集脚本决定经验来自哪里，状态机决定动作由谁执行，Value 决定如何评价进度，标注脚本决定哪些动作进入 positive 分支，适配层决定条件如何进入 π0.5，评测脚本则负责证明性能提升不是自动纠正器代做出来的。

---

## 第八部分 代码运行

### 8.1 环境部署

> 💡 **路径约定**：下文命令中 `/path/to/project-root` 表示本仓库在你电脑上的根目录，`/path/to/isaac-sim` 表示 Isaac Sim 的安装目录，均为占位符，请替换为自己机器上的实际路径。

**后续所有的代码都可进行终端恢复**

恢复时将--overwrite参数替换为--resume：
- 使用 --resume；
- 不要同时使用 --overwrite。

本章为了方便训练采用了--headless，如果想要看类似于14章的效果，去掉这个参数即可，本章不对可视化部分做过多说明。

进入本章代码目录并建立独立环境：

将openpi项目git到third_party文件夹中，目录为third_party\openpi

openpi地址：https://github.com/Physical-Intelligence/openpi

```bash
cd /path/to/project-root/code/code_chapter15
./setup_env.sh
```

**以下两个为运行参考，不用运行**

1、非 Isaac Sim 脚本统一通过本章环境运行：

```bash
./run.sh some_script.py [arguments...]
```

2、采集和评测需要 Isaac Sim 自带 Python：

```bash
/path/to/isaac-sim/python.sh some_isaac_script.py [arguments...]
```

下载或准备 π0.5 基础权重：

```bash
./download_checkpoint.sh
```

默认目录为：

```text
checkpoints/pi05_base/
├── params/
└── assets/
```

下载或准备 gemma-3-270m 基础权重：
```bash
cd /path/to/project-root/code/code_chapter15

modelscope download \
  --model google/gemma-3-270m \
  --local_dir /path/to/project-root/code/code_chapter15/checkpoints/gemma-3-270m
```

没有modelscope的使用:
```bash
pip install modelscope
```

常用环境变量如下：

```bash
export CHAPTER15_OPENPI_ROOT=/path/to/project-root/code/code_chapter15/third_party/openpi
export CHAPTER15_ISAAC_PYTHON=/path/to/isaac-sim/python.sh
export CHAPTER15_PI05_BASE=/path/to/project-root/code/code_chapter15/checkpoints/pi05_base
export CHAPTER15_OUTPUT_ROOT=/path/to/project-root/code/code_chapter15
```

服务端会校验 `chapter15_policy.json` 和唯一的 `norm_stats.json`。未经本章适配的通用权重即使包含 `params`，也可能没有 G2 的 16 维统计量和全绝对动作合同，不能直接用于 rollout。

> 💡 **要点**：`run.sh` 会把本章目录、`openpi/src` 和 `openpi-client` 加入 `PYTHONPATH`，再用 `third_party/openpi/.venv/bin/python` 执行脚本；`train_value.py` 要求本地 Gemma 目录中存在 `config.json`、`model.safetensors`、`tokenizer_config.json`，缺任何一个都会直接报错。

### 8.2 代码执行

下面按“路线 A：先 SFT 再 RL”的顺序给出全部命令，最后给出“路线 B：完全跳过 SFT”和流程检查命令。

#### 8.2.1 可选：先进行成功示教 SFT

采集每种颜色 20 条自动示教（脚本专家在 Isaac Sim 中执行完整抓放，结果写入 `data/raw/demonstrations`）：

```bash
cd /path/to/project-root/code/code_chapter15

/path/to/isaac-sim/python.sh collect_demos.py \
  --episodes-per-color 20 \
  --position-noise 0.01 \
  --seed 1000 \
  --headless \
  --overwrite
```

只有最终成功的示教会进入 SFT，但成功和失败示教都可保留给 Value 调试。随后导出 LeRobot 数据并计算统计量（第一条把成功示教转成 LeRobot 格式，第二条在原生 16 维上计算 `norm_stats.json`）：

```bash
./run.sh export_dataset.py --stage sft --overwrite
./run.sh compute_norm_stats.py --stage sft
```

推荐使用单卡 LoRA 微调（从 π0.5 base 初始化，训练 8000 步 SFT）：

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 ./run.sh train_policy.py \
  --stage sft \
  --initial checkpoints/pi05_base \
  --steps 8000 \
  --batch-size 32 \
  --lora \
  --fsdp-devices 1 \
  --overwrite
```

训练完成后应直接查找实际最大的数字 step，不要假定最后目录一定为 `7999`：

```bash
ls -1 checkpoints/g2_pi05_sft_000/sft_round_000
```

如果跳过 SFT，应直接使用已经完成 G2 数据适配的 π0.5 checkpoint 启动 rollout。该 checkpoint 必须包含本章的 `chapter15_policy.json`、唯一的 16D `norm_stats.json`，并声明 `g2_pi05_v1` 与 `all_absolute` 合同；通用 π0.5 base 缺少这些内容时不能直接可靠执行。

#### 8.2.2 采集一轮带自动纠正的经验

终端 A 启动当前策略服务（加载 checkpoint，校验合同后在 8000 端口提供 WebSocket 推理）：

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 \
./run.sh serve_policy.py \
  --checkpoint /path/to/current/checkpoint
```
保持终端 A 不关闭。

终端 B 启动第一轮经验采集（每色 20 条、每条最多 300 帧，进度耐心 80、运动耐心 35，结果写入 `data/raw/round_001`）：

```bash
cd /path/to/project-root/code/code_chapter15

/path/to/isaac-sim/python.sh collect_rollouts.py \
  --round 1 \
  --episodes-per-color 20 \
  --max-frames 300 \
  --position-noise 0.01 \
  --progress-patience 80 \
  --motion-patience 35 \
  --host 127.0.0.1 \
  --port 8000 \
  --seed 1000 \
  --headless \
  --overwrite
```

采集时应关注每条 episode 打印的 `success`、`frames`、`corrections` 和 `stop_reason`。纠正片段可能使最终 episode 长度略超过开始下一次操作前检查的帧预算，这是为了保证一段 `ACTIVE` 纠正和唯一一帧 `RELEASE` 原子完成，而不是在专家动作中间强制截断。

若采集中断，应使用相同的 episode 数和随机种子，并把 `--overwrite` 改为 `--resume`。续采会校验已有文件并只补齐缺失 episode。

#### 8.2.3 训练 Value 并生成 Advantage 标签

训练第一轮 Value（读取示教与第 1 轮 rollout，输出到 `checkpoints/value_round_001`）：

```bash
./run.sh train_value.py --round 1 --overwrite
```

显存不足时可以减小 batch 并使用梯度累积：

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 \
./run.sh train_value.py \
  --round 1 \
  --language-model checkpoints/gemma-3-270m \
  --steps 8000 \
  --batch-size 8 \
  --grad-accumulation 8 \
  --dtype bfloat16 \
  --overwrite
```

然后进行 Value 推理、50 步 Advantage 计算和任务内分位数标注：

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 ./run.sh label_advantages.py \
  --round 1 \
  --value-checkpoint checkpoints/value_round_001 \
  --batch-size 16 \
  --n-step 50 \
  --positive-ratio 0.30 \
  --overwrite
```

标注结果写入 `data/labeled/round_001`，原始 episode 不会被原地修改。每帧新增：

`value_target`、`value_prediction`、`advantage` 和 `acp_indicator`。

建议在进入策略训练前抽查：成功轨迹末端的 Value 是否接近 0、失败轨迹是否整体偏低、掉落或停滞附近是否出现 Value 下跌、纠正动作是否被强制标为 positive，以及每个任务的实际 positive 比例是否接近设定值。

需要特别注意，Value 训练 loss 很低并不等于标签一定可靠。如果 `value_prediction` 近似逐帧记住 `value_target`，由 $r_t=y_t-y_{t+1}$ 计算出的 Advantage 可能几乎全部抵消到 0。此时即使分位数代码仍能划出 30% positive，也不应直接进入策略微调；应先检查 Advantage 标准差和直方图、可视化若干正负样本，并从数据多样性、Value 容量、正则化和验证集泛化等方面排查塌缩。

#### 8.2.4 导出数据并更新 π0.5

导出优势条件数据并计算当前轮统计量（读取 `data/labeled/round_001`，每帧带 `acp_indicator`）：

```bash
./run.sh export_dataset.py --stage acp --round 1 --overwrite
./run.sh compute_norm_stats.py --stage acp --round 1
```

以当前策略为初始化进行优势条件微调（从 SFT 的 7999 步 checkpoint 出发，训练 30000 步）：

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
./run.sh train_policy.py \
  --stage acp \
  --round 1 \
  --initial checkpoints/g2_pi05_sft_000/sft_round_000/7999 \
  --steps 30000 \
  --batch-size 32 \
  --lora \
  --fsdp-devices 1 \
  --overwrite
```

下一轮把 `--round` 改为 2，并使用第一轮最终 checkpoint 启动服务与初始化训练。Value 会自动读取示教和截至当前轮的累计原始数据。

全参数 π0.5 训练需要远高于普通单卡的显存。当前代码在常见 32 GB GPU 上推荐 LoRA；减小 batch size 只能减少激活显存，不能消除完整模型与 AdamW 状态的常驻开销。

#### 8.2.5 最终无纠正评测

##### （1）评估 ACP/RL Round 1 模型

终端 A：启动 ACP

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 \
./run.sh serve_policy.py \
  --checkpoint checkpoints/g2_pi05_acp_001/acp_round_001/29999 \
  --host 0.0.0.0 \
  --port 8000
```

终端 B：ACP 各 50 次测试

```bash
cd /path/to/project-root/code/code_chapter15

/path/to/isaac-sim/python.sh evaluate.py \
  --episodes-per-color 50 \
  --sequential-episodes 50 \
  --max-frames 300 \
  --position-noise 0.01 \
  --host 127.0.0.1 \
  --port 8000 \
  --connect-timeout 60 \
  --inference-timeout 300 \
  --progress-every 50 \
  --seed 11000 \
  --headless \
  --output reports/evaluation_rl_round1_50.json
```

结果文件：
reports/evaluation_rl_round1_50.json
每个模型的测试规模为：
red：50
green：50
blue：50
单颜色合计：150
三物块顺序 episode：50

##### （2）评估 SFT 模型

终端 A：启动 SFT

```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 \
./run.sh serve_policy.py \
  --checkpoint checkpoints/g2_pi05_sft_000/sft_round_000/7999 \
  --host 0.0.0.0 \
  --port 8000
```

终端 B：SFT 各 50 次测试

```bash
cd /path/to/project-root/code/code_chapter15

/path/to/isaac-sim/python.sh evaluate.py \
  --episodes-per-color 50 \
  --sequential-episodes 50 \
  --max-frames 300 \
  --position-noise 0.01 \
  --host 127.0.0.1 \
  --port 8000 \
  --connect-timeout 60 \
  --inference-timeout 300 \
  --progress-every 50 \
  --seed 11000 \
  --headless \
  --output reports/evaluation_sft_50.json
```

结果文件：
reports/evaluation_sft_50.json

> 💡 **要点**：两次评测使用完全相同的 `--seed 11000`、`--position-noise 0.01` 和 `--max-frames 300`，因此物块随机位置序列一致，SFT 与 RL 的对比是公平的。

#### 8.2.6 路线 B：完全跳过 SFT

如果不使用 SFT，则跳过：
- collect_demos.py
- SFT 数据导出；
- SFT norm stats；
- SFT policy training。

但 checkpoints/pi05_base 必须已经包含：
- 本任务适用的 16D G2 normalization statistics；
- chapter15_policy.json；
- 正确的 all_absolute action contract。


1） 启动 base 策略采集 Round 1

终端 A：
```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 ./run.sh serve_policy.py \
  --checkpoint checkpoints/pi05_base \
  --host 0.0.0.0 \
  --port 8000
```

终端 B：
```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=0 /path/to/isaac-sim/python.sh collect_rollouts.py \
  --round 1 \
  --episodes-per-color 10 \
  --max-frames 300 \
  --position-noise 0.01 \
  --progress-patience 80 \
  --motion-patience 35 \
  --host 127.0.0.1 \
  --port 8000 \
  --headless \
  --overwrite
```

采集结束后停止服务器，然后依次执行（训练 Value → 标注 Advantage → 导出 ACP 数据 → 计算统计量 → 从 π0.5 base 直接做优势条件训练）：
```bash
cd /path/to/project-root/code/code_chapter15

CUDA_VISIBLE_DEVICES=1 ./run.sh train_value.py \
  --round 1 \
  --steps 8000 \
  --batch-size 64 \
  --dtype float32 \
  --overwrite
```

```bash
CUDA_VISIBLE_DEVICES=1 ./run.sh label_advantages.py \
  --round 1 \
  --value-checkpoint checkpoints/value_round_001 \
  --batch-size 16 \
  --n-step 50 \
  --positive-ratio 0.30 \
  --overwrite
```

```bash
./run.sh export_dataset.py \
  --stage acp \
  --round 1 \
  --overwrite
```

```bash
./run.sh compute_norm_stats.py \
  --stage acp \
  --round 1
```

```bash
CUDA_VISIBLE_DEVICES=1 ./run.sh train_policy.py \
  --stage acp \
  --round 1 \
  --initial checkpoints/pi05_base \
  --steps 30000 \
  --batch-size 32 \
  --lora \
  --fsdp-devices 1 \
  --overwrite
```

> ⚠️ **注意**：路线 B 中 `train_policy.py` 检测到 ACP 直接从 π0.5 base 出发时，会打印 `warning: ACP starts directly from pi0.5 base because SFT was skipped`，这是预期提示。

#### 8.2.7 流程检查

只打印计划、不执行（`run_pipeline.py` 默认只打印命令，便于核对两条路线的依赖顺序）：

```bash
./run.sh run_pipeline.py --with-sft --round 1
./run.sh run_pipeline.py --no-sft --round 1 --init /path/to/g2_init
```

不启动 Isaac Sim 的测试与静态检查（单元测试、编译检查和 ruff 代码检查）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run.sh -m pytest -q tests
./run.sh -m compileall -q .
./run.sh -m ruff check . --exclude third_party
```

### 8.3 运行结果

SFT的结果：

```bash
  "complete": true,
  "no_intervention": true,
  "completed_single_episodes": 150,
  "completed_sequential_episodes": 50,
  "single_color_success": {
    "red": 0.98,
    "green": 0.96,
    "blue": 0.74
  },
  "sequential_three_color_success": 0.54,
```

强化学习微调后的结果：

```bash
  "complete": true,
  "no_intervention": true,
  "completed_single_episodes": 150,
  "completed_sequential_episodes": 50,
  "single_color_success": {
    "red": 0.98,
    "green": 1.0,
    "blue": 0.76
  },
  "sequential_three_color_success": 0.62,
```

| 指标 | SFT | RL Round 1 | 绝对提升 |
|---|---:|---:|---:|
| 红色单物块 | 49/50 = 98% | **49/50 = 98%** | **0 个百分点** |
| 绿色单物块 | 48/50 = 96% | **50/50 = 100%** | **+4 个百分点** |
| 蓝色单物块 | 37/50 = 74% | **38/50 = 76%** | **+2 个百分点** |
| 单物块总计 | 134/150 = 89.3% | **137/150 = 91.3%** | **+2 个百分点** |
| 连续完成三个物块 | 27/50 = 54% | **31/50 = 62%** | **+8 个百分点** |

可以看出，rl确实效果有所提升，但是蓝色物块的整体成功率依然不高，训练集中并没有包含连续完成三个物块的结果，但是pi0.5和改进后的rl结果都能够泛化到连续完成三个物块的长程任务，其实pi0.5对于这种简单任务，性能已经足够好了，所以rl的提升不大，可能对于更复杂的任务更大的数据集，rl的提升效果会更好，此处仅做教程学习使用，为了方便大家理解，采用统一的教程案例。

但是rl的进步一定程度上依赖于sft的结果，大家可以尝试一下不采用sft，纯rl的微调，看看效果如何，同时，因为采集的时候，考虑到没有实物的人工示教和训练时间问题，为了方便大家学习，采用了前面章节的自动示教，同时每个物块仅自动采集了20条轨迹数据，没有人工示教那样的高准确性，同时可能采集轨迹不够，因此提升有限，但是大家应该能够从中学习到了具体的算法流程，如果想要进一步的改进，可以自行尝试，谢谢大家~

**如何读懂这些输出**：

- `complete: true` 表示全部 150 条单色与 50 条连续任务都跑完；中途中断时 `evaluate.py` 也会把已完成部分写入报告，但该字段为 `false`。
- `no_intervention: true` 由 `build_report()` 固定写入，表明这是纯策略评测，没有自动专家参与。
- `single_color_success` 中的小数是成功率，乘以 50 即成功条数，例如 `0.74 × 50 = 37`。
- `sequential_three_color_success` 是“随机顺序连续完成三个物块”的比例；它近似受三个单色成功率共同制约。作为粗略参考，若三次互相独立，SFT 的单色成功率乘积为 $0.98\times0.96\times0.74\approx0.70$，实测 0.54 更低，说明连续任务中前一个物块的执行会影响后续状态（例如机械臂起点不再是 home 姿态）。
- 报告 JSON 中的 `episodes` 数组还记录了每条 episode 的 `reason`（`success` / `timeout` / `unsafe_action` / `policy_error:*`）与 `frames`，可以据此分析蓝色物块的主要失败原因。

---

## 本章小结

### 知识回顾

本章建立了一条从部署经验到策略改进的完整链路：π0.5 在仿真中自主执行，自动系统发现危险与停滞后进行局部纠正，成功、失败和恢复过程全部进入累计数据；分布式 Value 把 episode 级成功信号转成逐帧进度估计，50 步 Advantage 再把动作划分为更优和较差两类；最后通过优势条件化 prompt 和 flow matching 监督训练更新策略，并在无纠正条件下独立评测。

与直接部署标准 π0.6 或 π0.6\* 相比，本章保留 π0.5 的 Gemma 2B 与 300M 动作专家，不包含 Gemma 3 4B 主干、860M 动作专家、FAST 离散动作和中间子任务预测；Value 也采用独立的 SigLIP 与 Gemma 3 270M 融合实现。但经验收集、Value 学习、Advantage 标注、条件化策略提取和迭代数据聚合构成了完整且可运行的教学闭环。

本章最重要的工程经验不是“把失败数据重新训练一遍”，而是保证失败为什么发生、纠正由谁执行、控制权何时交还、Value 如何定义、标签如何按任务产生，以及最终评测是否真正无干预。只有这些数据语义和控制边界都清楚，经验驱动的 VLA 微调才可能带来可解释、可复现的性能提升。

| 环节 | 核心公式/规则 | 关键参数 | 代码 |
|---|---|---|---|
| Value target | $y_t=\mathrm{clip}\big(g_t/(T_{\max}+C_{\mathrm{fail}}),-1,0\big)$ | $C_{\mathrm{fail}}=T_{\max}$ | `value_math.value_targets` |
| 分布式 Value | two-hot 交叉熵； $\hat V=\sum_b p_b c_b$ | 201 bins， $[-1,0]$ | `two_hot`、`Pistar06Value` |
| 稠密奖励 | $r_t=y_t-y_{t+1}$，末帧 $r=y_T$ | — | `dense_rewards` |
| Advantage | $A_t=\sum_{k<N}r_{t+k}+\hat V_{t+N}-\hat V_t$ | $N=50$，不跨 episode | `n_step_advantage` |
| 二值条件 | $I_t=\mathbf 1[A_t\ge Q_{0.7}^{(\ell)}]$，ACTIVE 强制 1 | top 30% | `acp_labels` |
| 条件注入 | prompt 追加 `Advantage: positive/negative` | dropout 0.3 | `acp.py`、`ACPTransform` |
| 策略训练 | flow matching 监督损失 | 30k steps，lr 2.5e-5，LoRA | `train_policy.py` |
| 控制权 | POLICY→ACTIVE→RELEASE(1 帧)→POLICY | 进度耐心 80，运动耐心 35，跳变 0.65 rad | `hil.py`、`rollout_core.py` |
| 评测 | 不加载自动专家 | 每色 50 + 连续 50 | `evaluate.py` |

### 课后练习

1. **手算题（Value target）**：某任务 $T_{\max}=80$，一条失败 episode 共 40 帧。求第 0 帧、第 20 帧和最后一帧的 $y_t$。（提示： $C_{\mathrm{fail}}=80$，分母为 160；答案依次为 $-119/160\approx-0.744$、 $-99/160\approx-0.619$、 $-80/160=-0.5$。）
2. **思考题（Advantage 塌缩）**：用 5.2 节的推导说明，为什么“Value 在训练集上的 loss 越低”不一定意味着“Advantage 标签越有用”。如果要检测这种问题，你会划分怎样的验证集？
3. **思考题（阈值设计）**：如果把所有颜色混在一起取一个全局 70% 分位阈值，而蓝色任务的 Advantage 普遍偏低，蓝色任务的 positive 比例会怎样变化？这对策略学习蓝色物块有什么影响？
4. **动手题（value_math 单元实验）**：用 `./run.sh` 进入本章环境，写一个小脚本调用 `value_math.value_targets / dense_rewards / n_step_advantage`，复现 5.1 节 4 帧例子的结果；再把 `values` 换成 `targets` 本身，验证所有 Advantage 都变为 0。
5. **动手题（positive 比例）**：把 `label_advantages.py` 的 `--positive-ratio` 从 0.30 改为 0.50 重新标注、导出并训练一个短程 ACP 策略（例如 `--steps 2000`），观察脚本打印的实际 `positive_ratio` 与每任务 `thresholds` 如何变化，并在相同 `--seed 11000` 下与 0.30 的结果比较。
6. **动手题（停滞阈值）**：把 `collect_rollouts.py` 的 `--progress-patience` 改为 40、`--motion-patience` 改为 20 采集一小轮（如 `--episodes-per-color 3`），统计每条 episode 打印的 `corrections` 数量与 `dataset.summarize()` 中的 `intervention_frames`，讨论阈值过敏对“positive 几乎全来自纠正帧”的影响。

### 参考资料

- π0 论文：`https://arxiv.org/abs/2410.24164`
- π0.5 论文：`https://arxiv.org/abs/2504.16054`
- π0.6 论文：`2511.14759v2.pdf`
- Evo-RL：`https://github.com/MINT-SJTU/Evo-RL`
- OpenPI 源码快照：`code/code_chapter15/third_party/openpi`
- OpenPI 项目地址：`https://github.com/Physical-Intelligence/openpi`
