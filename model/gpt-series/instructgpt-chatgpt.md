# InstructGPT/ChatGPT

## 介绍
由于OpenAI没有公开ChatGPT的具体细节，而ChatGPT与InstructGPT的原理类似，故本章基于ChatGPT的官方博客，结合InstructGPT的论文进行理解。

而InstuctGPT发表于2022年，论文《[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)》。

总体来说，InstructGPT采用了GPT-3的网络结构，通过指示学习（instruction learning）构建训练样本来训练一个反应预测内容效果的奖励模型（RM），最后通过这个奖励模型的打分来指导强化学习模型的训练。


## 背景技术
- GPT系列：GPT-1、GPT-2、GPT3、InstructGPT
- 提示学习（Prompt Learning）
- 指示学习（Instruct Learning）
- 强化学习
    - RLHF (Reinforcement Learning From Human Feedback)
    - PPO算法 (Proximal Policy Optimization)


## 流程&模型

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/chatgpt.png width=100% />
</div>

### 1.SFT阶段(Supervised FineTune)

**(1) 主旨**

基于GPT3结构以及参数，初始化一个模型。根据采集的SFT（Supervised FineTune）数据集对此模型进行有监督的微调。

**(2) 流程**

事实上这个SFT数据集，是一个关于指令(Instruction)的数据集，由**提示-答案**组成。部分来自OpenAI的API用户输入，部分来自OpenAI雇佣的40名标注工（labeler），标注的三个原则
- 简单任务：labeler给出任意一个任务，且确保任务多样性
- Few-shot任务：给出一个指示，以及该指示的多个查询-答复对
- 用户相关：从API获取用例，labeler编写指示

换句话说，这一步是对GPT3进行指令微调。

**(3) 模型**

与GPT3结构一致，这次不详细介绍。可跳转到[GPT-3: 《Language Models are Few-Shot Learners》](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/gpt-3.md)进行阅读。


### 2.RM阶段(Reword Model)

**(1) 主旨**

收集人工标注的对比数据，训练奖励模型（Reword Model，RM）。

**(2) 流程步骤：**
- 从SFT数据集中，抽取出部分prompt样本。
- 将样本其输入到SFT模型中，生成大量的答案。
- 标注人员，会对SFT生成的多个答案，两两对比生成质量，给出一个具体的分数，并进行整体排序，由此得到标注数据集2。具体而言，就是对于每个给出的Prompt，SFT模型生成$k$个输出结果，将$k$个结果，整理成$C^2_k$个组合结果，提供给标注人员进行标注。
- 基于标注数据，训练一个RM模型。

**(3) 模型**

同样使用预训练+SFT的方式，初始化了一个6B大小(billion)的RM模型。与SFT模型除了模型参数量不同，RM模型还去掉了最后的embedding编码层。

训练阶段，模型输入是来自RM的标注数据集；推断阶段，模型输入来自PPO模型的生成结果文本。输出是奖励值。

具体而言，就是对于输入的Prompt，SFT模型生成$k$个输出结果，将$k$个结果，整理成$C^2_k$个组合结果，RM模型输出奖励值，同时与标注人员的标注值，将$k$个结果放到一个batch里计算损失。损失函数的目标是**最大化标注人更喜欢的生成结果和不喜欢的生成结果之间的差值**。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/chatgpt-func1.png width=55% />
</div>


### 3.PP0阶段(Proximal Policy Optimization)

**(1) 主旨**

使用 RM 作为强化学习的优化目标，利用 PPO算法 微调 SFT模型。

**(2) 流程步骤：**
- 使用SFT阶段得到的SFT模型，初始化一个新的模型，暂且叫PPO模型。
- 从SFT阶段的数据集中，抽取一些新的prompt，以及从OpenAI的API中抽取部分用户的query prompt。
- 使用PPO模型生成结果文本集合。
- 将结果文本集合输入到RM奖励模型，得到奖励值
- 根据奖励值，使用PPO策略指导PPO模型训练和参数更新

**(3) 模型**

与GPT3结构参数量一致。这里使用的是SFT步骤得到的模型，来进行初始化175B的PPO模型。

训练阶段，模型输入prompt，输出生成文本。

学习目标分为两个：
- 1.PPO策略学习目标
    - 策略梯度的奖励期望$$r_θ(x,y)$$
    - KL惩罚项，来自PPO的策略原理，目的是避免策略优化后的模型与SFT模型的差异。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/chatgpt-func2.png width=27% />
</div>

- 2.通用语言模型学习目标
    - 在paper中，该模型叫PPO-ptx。为了防止模型在学习期间遗忘了最原始的LM能力，缓解了灾难性遗忘的问题。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/chatgpt-func3.png width=25% />
</div>

整体学习目标为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/chatgpt-func4.png width=65% />
</div>


## 优势
- 相比GPT-3，ChatGPT生成效果更真实
- ChatGPT在无害性上，比GPT-3效果要好
- ChatGPT有code能力

## 缺点
- 降低了在通用NLP任务上的效果
- 偶尔输出常识性错误
- 对指示非常敏感
- 有害的指示，可能输出有害的答复


## Reference

- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

- [OpenAI-Introducing ChatGPT](https://openai.com/blog/chatgpt/)

- [李沐-InstructGPT](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=32744&ctype=0)

- [符尧-追溯ChatGPT各项能力的起源](https://mp.weixin.qq.com/s/9NNS6AIVRLQmxKe2YxdPcw)




