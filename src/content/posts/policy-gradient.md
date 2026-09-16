---
title: "策略梯度"
pubDate: 2025-02-17
description: "从策略梯度的基本公式出发，整理模型、训练过程、优缺点以及 LLM 中的应用。"
author: "Westwoods"
tags: [强化学习, LLM]
draft: false
---

# 介绍
策略梯度（PG）是典型的基于策略的强化学习方法。以 Q学习和 DQN 为代表的基于价值函数的方法是通过求出一个可以估计综合总回报的函数，并且每次都贪心地选择总回报最高的行动来实现回报最大化；基于策略的强化学习则是试图直接以总回报最大化为目标训练一个决策模型来作为策略。相比之下，基于策略的方法要更加直接，也更加容易理解。
# 策略梯度
强化学习的策略用 $\pi(a|s)$ 表示，要将其作为模型来训练，就需要求出**梯度**，因此这个方法叫做策略梯度。

参数为 $\theta$ 的策略 $\pi(a|s;\theta)$ 看似是输入状态，输出行动；实际上我们应该将其看作输入整场游戏，输出一条轨迹（$\tau=s_{0},a_{0},s_{1},a_{1},\dots,s_{T},a_{T}$）。我们的目标是调整参数 $\theta$，实现回报最大化，即最大化回报函数 $J(\theta)$：
$$
\max _{\theta}J(\theta)=\mathbb{E}_{\tau \sim\pi}[R(\tau)]=\sum_{\tau}P(\tau;\theta)R(\tau)
$$
其中 $\tau\sim\pi$ 是因为轨迹 $\tau$ 完全是由 $\pi$ 决定并且产生的，即按照上面说的，可以看作 $\pi$ 的输出。因此我们可以说其服从于 $\pi$；$R(\tau)$ 是轨迹 $\tau$ 在游戏中获得的回报。接下来我们需要一步步解开这个公式。由于 $\tau$ 是一个随机变量序列且具有马尔可夫决策过程的性质，因此有
$$
P(\tau;\theta)=P(s_{0})\prod_{t=1}^T\pi(a_{t}|s_{t};\theta)P(s_{t+1}|s_{t},a_{t})
$$
其中 $P(s_{t+1}|s_t,a_{t})$ 是状态转移概率，这是由环境决定的，和决策无关。因此对其使用**对数求导**（导数与微分），可得：
$$
\nabla_{\theta}P(\tau;\theta)=P(\tau;\theta) \nabla_{\theta}\log P(\tau;\theta)=P(\tau;\theta)\sum_{ t =1}^{ T }\nabla_{\theta}\log\pi(a_{t}|s_{t};\theta)
$$
于是用于更新策略模型参数的梯度（策略梯度，回报函数的导数）：
$$
\begin{array}{l}
\nabla_{\theta}J(\theta) & = & \displaystyle \sum_{\tau}\nabla_{\theta}P(\tau;\theta)R(\tau) \\
 & = & \displaystyle   \sum_{\tau}R(\tau)P(\tau;\theta)\sum_{ t =1}^{ T }\nabla_{\theta}\log\pi(a_{t}|s_{t};\theta) \\
 & = & \displaystyle \mathbb{E}_{\tau\sim\pi}\left[ R(\tau)\sum_{ t =1}^{ T }\nabla_{\theta}\log\pi(a_{t}|s_{t};\theta) \right]
\end{array}
$$
一般的机器学习模型都是支持求导的，求 $\nabla_{\theta}\log\pi(a_{t}|s_{t};\theta)$ 不会有什么问题。因此我们可以用反向的梯度下降法求出能使 $J(\theta)$ 最大化的 $\theta$ 值，这样就实现了直接优化策略函数实现回报最大化。上述对抽象的回报函数求导的方法，有时候也叫做**REINFORCE trick**。
# 模型与训练
策略梯度可以适应离散行动或者连续行动。
- 如果行动是离散的，则可以让策略模型输出 softmax 概率；
- 如果行动是连续的，则可以让策略模型输出正态分布 （输出参数 $\mu,\sigma$），再从分布中采样；
# 优缺点
1. 优点：
	1. 行动可以是连续的；
	2. 决策可以是随机的；
2. 缺点：
	1. 需要统计轨迹 $\tau$ 出现的概率，实际上轨迹太长的话可能重复一万次实验都无法有效统计到其概率；
	2. 由于存在**过冲**和**下冲**，难以收敛。

# LLM 的策略梯度
以 LLM 的 RL 为例：
1. $\pi_{\theta}(y_{t}|x,y_{<t})$：表示参数为 $\theta$ 的 LLM 基于 prompt token 序列 x 和已经 decode 的前缀 $y_{<t}$ 预测出新 token $y_{t}$；在这里 PG 的轨迹 $\tau$ 的 state 相当于当前的 prefix，采样得到的 token $y_{t}$ 是采取的 action； 
2. 我们给轨迹 $\tau$ 打个分 $G(\tau)$ 作为奖励；实现上 $G(\tau)$ 可以是一个打分模型，也可以直接就是用户提供的反馈（比如 ChatGPT 每个回答下的赞/踩）

LLM 产生 $\tau$ 这个序列的概率是：
$$
P_{\theta}(\tau|x)=\prod_{t}^{|\tau|}\pi_{\theta}(y_{t}|x,y_{<t})
$$
那最后总奖励是：
$$
J(\theta)=\sum_{\tau}G(\tau)P_{\theta}(\tau|x)
$$
这个 $\tau$ 其空间极大，也是不能穷举，所以我们其实会继续用**蒙特卡洛估计**，采样 N 个 $\tau$ 代替整体：
$$
J(\theta)=\frac{1}{N}\sum_{\tau}G(\tau)
$$
这个东西因为经过了采样，理论上无法求导。所以蒙特卡洛不能在这里做。我们回到理论总奖励，对它求梯度并且使用 log trick：
$$
\nabla_{\theta}J(\theta)=\sum_{\tau}G(\tau)P_{\theta}(\tau|x)\sum_{t}^{|\tau|}\nabla_{\theta}\log \pi_{\theta}(y_{t}|x,y_{<t})
$$
然后进行蒙特卡洛估计：
$$
\hat{\nabla_{\theta}J(\theta)}=\frac{1}{N}\sum_{\tau}G(\tau)\sum_{t}^{|\tau|}\nabla_{\theta}\log \pi_{\theta}(y_{t}|x,y_{<t})
$$
损失梯度至此得到：
$$
\nabla_{\theta}L=-\hat{\nabla_{\theta}J(\theta)}
$$
