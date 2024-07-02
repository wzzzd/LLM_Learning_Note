# DPO

## 论文
来自论文[《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》](https://arxiv.org/abs/2305.18290)

## 思路
 在RLHF过程中，需要显式训练一个Reward Model，再根据Reward Model的奖励机制，来优化Policy Model。

 DPO是一种直接使用偏好进行策略优化的方法。

 将对奖励函数的损失转化为对策略的损失，跳过显式的奖励建模步骤直接进行策略优化。

 实质上，策略网络既代表语言模型，又代表奖励。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-dpo-process1.png width=70% />
</div>


## 结构
### 损失函数

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-dpo-func1.png width=60% />
</div>

其中，
- σ 表示sigmoid函数
- β 表示超参数，一般0.1-0.5
- y_w 表示偏好数据中好的response
- y_l 表示偏好数据中差的response
- π_θ 表示policy model生成的response概率累积
- π_ref 表示reference model生成的response概率累积

上式括号中可以简化为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-dpo-func2.png width=40% />
</div>

优化目标可以理解为，括号左边的差值，比右边的差值，margin越大越好。即，好的response相对ref的效果，要比差的response相对ref的效果要好才行。


### 模型
reference model和policy model来自同一个SFT模型进行初始化。

训练过程reference model不会更新，只会更新policy model。

## 流程

1.对于每个prompt，采样回答<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-dpo-func3.png width=15% />，基于人类偏好标注并构建离线数据集<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-dpo-func4.png width=15% />。

2.对于给定的π_ref和数据集D，优化语言模型π_θ以最小化L_DPO和期望β。

## Reference

- https://zhuanlan.zhihu.com/p/642569664
- https://zhuanlan.zhihu.com/p/634705904# 

