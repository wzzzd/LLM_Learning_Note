# ORPO

## 论文
来自2024年3月的论文[《ORPO: Monolithic Preference Optimization without Reference Model》](https://arxiv.org/pdf/2403.07691)

## 背景
从PPO、到DPO、再到ORPO，是一个偏好优化算法的演进过程。

三种算法目的都是在解决，怎么在监督微调SFT后，如何优化LLM，使其生成符合人类偏好的内容（如有用性、无害性、诚实性）
- PPO：引入奖励模型
- DPO：不需要奖励模型，直接让LLM同时学习偏好
- ORPO：直接添加一个对不受欢迎生成内容的惩罚项

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-process1.png width=60% />
</div>


## 思路
在sft的损失函数基础上，添加一个对不受欢迎的生成风格的惩罚项，就足以实现偏好对齐了。 （类似回归模型里添加L1、L2正则惩罚项）

惩罚项，是一个基于赔率比（Odds Ratio）的惩罚，用来区分受欢迎（对应w）和不受欢迎（对应l）的响应
- OR用来表示相对于生成l，模型更偏爱生成w的程度。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-func1.png width=30% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-func2.png width=30% />
</div>


## 损失函数
SFT的损失

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-func3.png width=30% />
</div>

Odds损失
- 期望通过最小化 L_OR来增加y_w和y_l之间的赔率比。
- 反过来讲，目的是希望最大化期望回答和非期望回答之间的赔率比。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-func4.png width=30% />
</div>

总损失
- 目标函数一方面可以通过SFT损失来优化模型在特定领域的适应性，同时还可以通过损失来惩罚不受欢迎的生成风格，从而实现偏好对齐。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-func5.png width=30% />
</div>


## 实验

λ 对实验结果的影响
- 通过调整λ的值，可以动态控制模型对偏好的敏感程度。
- 不同 λ 值对选定响应（chosen responses）和拒绝响应（rejected responses）的对数概率的影响。
    - 较大的 λ 值会导致模型更强烈地歧视拒绝响应。
    - 在下游任务中，较大的 λ 值可能会导致模型在需要确定性答案的类别中表现不佳，而在需要开放式回答的类别中表现更好

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-orpo-eff1.png width=50% />
</div>


## Reference
    * https://zhuanlan.zhihu.com/p/688583797

