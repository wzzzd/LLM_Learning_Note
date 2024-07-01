# RLHF


## 思路

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-process1.png width=50% />
</div>

以ChatGPT/InstructGPT为例，进行说明。

采用了GPT-3的网络结构，通过指示学习构建训练样本来训练一个SFT模型。

通过SFT生成的样本pair，以及人为标注的偏好标签，训练一个反应预测内容效果的奖励模型（RM）。

最后通过这个奖励模型RM的打分来指导强化学习模型PPO的训练。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-process2.png width=60% />
</div>


## 流程

### Step1: 训练SFT模型
#### 逻辑
根据采集的SFT（Supervised FineTune）数据集对GPT-3进行有监督的微调。
#### 数据集
由 **提示-答复** 对组成的样本。

部分数据来自，OpenAI的API中用户输入。

部分数据来自，雇佣的40名标注工，进行文本标注（生成文本）。
- 简单任务：labeler给出任意一个任务，且确保任务多样性
- Few-shot任务：给出一个指示，以及该指示的多个查询-答复对
- 用户相关：从API获取用例，labeler编写指示

#### 训练方式

使用GPT-3初始化 SFT模型（175亿）。

使用的是instruction learning的训练方式，即构造一批指示数据集，并进行自回归的监督训练。

学习任务是：next token prediction

学习目标是:  极大似然估计

### Step2: 训练RM模型
#### 逻辑
- 收集人工标注的对比数据，训练奖励模型（Reword Model，RM）

#### 数据集
预计规模为50k左右。

从SFT数据集中抽取出部分提示。将提示输入step1的SFT模型，生成一批候选文本。

labeler根据生成质量，对生成的候选文本排序。

对于每个Prompt，LM生成K个输出后，将K个输出整理成 $C_2^K$ 个结果，提供给labeler标注。

得到标注数据 pair，如（text1，text2，偏好分数）。

#### 训练方式
初始化构造一个6B的LM模型。

模型输入为（prompt，LM生成的文本），输出是奖励值。

损失函数方面，
- 对于单轮数据中，标注集中的K个LM的输出结果pair，放到一个batch里，计算奖励模型的损失函数。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-func1.png width=30% />
</div>

- 损失函数的目标是最大化labeler更喜欢的生成文本和不喜欢的生成文本之间的差值。
- 值得注意的是，对于一个样本pair，RM会运行两次，得到两个分数，再对这两个分数计算loss。

### Step3: 训练PPO模型
#### 逻辑
使用 RM 作为强化学习的优化目标，利用 PPO算法 微调 SFT模型

#### 数据集
来自OpenAI中API的用户数据，即 提示instruction

#### 训练方式
1.使用Step1训练好的SFT模型来初始化PPO模型
2.抽取prompt数据集中的提示，PPO模型输出生成文本
3.生成文本，输入RM奖励模型，得到奖励值
4.使用PPO策略指导PPO模型继续训练

#### 损失函数
添加一个KL惩罚项，确保PPO和SFT输出差距不会太大。原因是来避免PPO策略优化后的模型与SFT模型出现太大的差异。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-func2.png width=35% />
</div>


添加一个通用语言模型学习目标，在paper中，叫PPO-ptx。原因是只进行PPO训练会导致模型在通用NLP任务中，性能大幅下降。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-func3.png width=35% />
</div>


#### 整体目标

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/align-rlhf-func4.png width=30% />
</div>


## 强化学习在LLM中的概念
- 策略(policy)：是一个接受提示并返回一系列文本 (或文本的概率分布) 的 LM
- 行为(action space)：LM 的词表对应的所有词元
- 观察(observation space)：是可能的输入词元序列，也比较大 (词汇量 ^ 输入标记的数量)
- 奖励函数(reward model)：是偏好模型和策略转变约束 (Policy shift constraint) 的结合

## 优点
- 相比GPT-3，ChatGPT生成效果更真实
- ChatGPT在无害性上，比GPT-3效果要好
- ChatGPT有code能力

## 缺点
- 降低了在通用NLP任务上的效果
- 偶尔输出常识性错误
- 对指示非常敏感
- 有害的指示，可能输出有害的答复



##  相关论文
- 《Deep Reinforcement Learning from Human Preferences》
    - RLHF applied on preferences between Atari trajectories.
- 《Training language models to follow instructions with human feedback》
- 《Fine-Tuning Language Models from Human Preferences》
    - An early paper that studies the impact of reward learning on four specific tasks.
- 《Learning to summarize with human feedback》
    - RLHF applied to the task of summarizing text. Also, Recursively Summarizing Books with Human Feedback (OpenAI Alignment Team 2021), follow on work summarizing books.
- 《WebGPT: Browser-assisted question-answering with human feedback》
    - Using RLHF to train an agent to navigate the web.
- 《InstructGPT: Training language models to follow instructions with human feedback》
    - RLHF applied to a general language model [Blog post on InstructGPT].
- 《ChatGPT: Optimizing Language Models for Dialogue》
    - Training a LM with RLHF for suitable use as an all-purpose chat bot.
- 《Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback》
    - A detailed documentation of training a LM assistant with RLHF.
- 《Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned》
    - A detailed documentation of efforts to “discover, measure, and attempt to reduce [language models] potentially harmful outputs.”
- 《Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning》
    - Using RL to enhance the conversational skill of an open-ended dialogue agent.

