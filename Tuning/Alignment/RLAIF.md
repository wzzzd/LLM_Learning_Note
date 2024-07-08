# RLAIF
## 论文

[《Scaling Reinforcement Learning from Human Feedback with AI Feedback》](https://arxiv.org/abs/2309.00267)

## 思路
- 主要思路是使用LLM替换人类标记偏好，即RM的标签数据使用LLM进行标注，以及使用LLM标注的偏好数据训练好的RM，来训练RL模型。

- 在RLAIF中，首先，使用LLM来评估给定的文本和2个候选回复；然后，这些由LLM生成的偏好数据被用来训练一个奖励模型，这个奖励模型用于强化学习，以便进一步优化LLM。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-process1.png width=60% />
</div>

## 结构

- 用于评估偏好回复的Prompt格式为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-process2.png width=60% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-process3.png width=60% />
</div>

- 在解码阶段添加了以下这些处理
    - 避免位置偏差（Addressing Position Bias）
        - 候选回复喂给LLM的顺序可能会偏向它喜欢的候选顺序，尤其是在 LLM 参数较小的情况下。
        - 为了减轻位置偏差的影响，作者进行了双重推理和平均处理。
    - 思维链推理（Chain-of-thought Reasoning）
        - 论文将标准提示符（即“Preferred Summary=”）的结尾替换为“ Consider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better. Rationale:”，让LLM在多个方面进行偏好的选择和分析
    - 自洽性（Self-Consistency）
        - 采样多个推理结果，并按照投票的方式，选出投票最多的最终推理结果。
## 结果

在人类评估上，与SFT策略相比，有71%统计概率认为RLAIF的结果更好，有73%统计概率认为RLHF的结果更好。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-eff1.png width=40% />
</div>

SFT/RLHF/RLAIF三种结果对比

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-eff2.png width=60% />
</div>

少样本（few-shot-learning）并没有带来效果上的提升，甚至效果更差

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-eff3.png width=40% />
</div>

Self-Consistency with CoT对性能的影响如下，用T=1采样会导致与人类偏好的一致性较低。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-eff4.png width=40% />
</div>

对用于评估的LLM的参数大小进行了探索，发现与人类偏好的一致性随着LLM大小的增加而增加。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlaif-eff5.png width=40% />
</div>


## Reference
- https://cloud.tencent.com/developer/article/2352114
- https://blog.csdn.net/qq_27590277/article/details/132769778


