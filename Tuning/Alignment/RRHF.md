# RRHF

## 论文
来自阿里达摩院和清华推出的对齐方法，[《RRHF: Rank Responses to Align Language Models with Human Feedback without tears》](https://arxiv.org/pdf/2304.05302.pdf)

## 背景
RLHF的研究工作主要使用PPO算法对语言模型进行优化。

然而，PPO算法包含许多超参数，并且在算法迭代过程中需要多个独立模型相互配合，因此错误的实现细节可能会导致训练结果不佳。

## 思路
无需强化学习，即可实现语言模型的对齐任务。论文推出了一种基于排序的人类偏好对齐的方法。

RRHF利用不同语言模型（ChatGPT、GPT-4、当前的训练模型或人类专家）生成回复，对回复进行评分，并通过排名损失来使回复与人类偏好对齐。

与PPO不同，RRHF的训练过程可以利用人类专家或GPT-4的输出作为对比。

训练好的RRHF模型可以同时用作生成语言模型和奖励模型。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-process1.png width=60% />
</div>


## 计算逻辑

对于输入query，首先通过不同的方式获得k个回复，再用奖励模型对这k个回复分别打分。

为了使模型与分数{r_i}_k对齐，让模型π对每一个y_i使用下式计算分数p_i，p_i 是模型π下y_i的对数条件概率。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-func1.png width=40% />
</div>

使用ranking loss来优化这样的目标：
- 目的是使模型π对高质量输出给更大概率，对低质量输出给小概率。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-func2.png width=40% />
</div>

给模型另外一个目标是去直接学习得分最高的回复

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-func3.png width=40% />
</div>

总损失

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-func4.png width=40% />
</div>


## 效果

Loss稳步下降的同时奖励得分稳步上升。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-eff1.png width=60% />
</div>

RRHF与PPO的效果对比
- RRHF算法可以在较低的训练难度下拟合奖励模型的偏好，达到PPO算法的效果，并且避免了PPO算法中的复杂性和不稳定性问题。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rrhf-eff2.png width=50% />
</div>

