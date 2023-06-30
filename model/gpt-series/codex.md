# Codex

## 论文信息
OpenAI于2021年提出了Codex代码生成模型，来自论文 [《Evaluating Large Language Models Trained on Code》](https://arxiv.org/pdf/2107.03374.pdf)

## 介绍

基于GPT模型结构，Codex拥有12B的参数量。

使用GPT家族进行模型参数的初始化，使用github上的代码进行fine-tuning。

注意：codex已经被openai官方弃用，目前官方已经将其能力整合到chatGPT中。

## 输入输出定义

输入：代码的注释

输出：根据注释生成代码

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-input-output.png width=50% />
</div>

## 模型结构

模型结构采用GPT家族系列之一，参数范围为12M/25M/42M/85M/300M/679M/2.5B/12B。

由于GPT是由transformer decoder构成的，故在inference阶段，需要选择一种解码方法。本论文采用了nucleus sampling的采样方法，使得每一个时间step生成的token不会出现特别不靠谱的词，同时增加了结果的多样性。

常规解码方法有
- **贪婪算法**：在每一个step，获取当前概率值最大的token
- **beam search**：在每一个step，保留k个当前解码的序列结果
- 本论文方法-**nucleus sampling**
    - 在每一个step，保留概率值加起来达到0.95的所有token
    - 从这些token中，随机采样一个token作为当前token生成结果

关于停止生成的问题，一般生成模型会学习一个特殊的停止生成符号，而本论文则自定义了一些的停止生成符号：‘\nclass’, ‘\ndef’, ‘\n#’, ‘\nif’,  ‘\nprint’。（也就是编程里面一些常规的新逻辑开始的标志符）

## 数据集
论文中提及了3个数据集，分别为：预训练数据集、监督微调数据集和评估数据集。
- **预训练数据集**
    - 目的：基于自回归任务，继续训练模型
    - 说明：从github上爬取的python代码，共179GB的python文件，过滤后得到159GB的训练数据，训练得到模型为Codex
- **监督微调数据集**
    - 目的：用于微调预训练后的模型，为了与评估数据集做数据分布对齐（alignment）
    - 说明：额外采集了部分跟评估数据集相近的数据（包含1w的编程比赛数据，4w单元测试服务数据），训练得到的模型为Codex-s
- **评估数据集**
    - 目的：本论文构建的用于评估代码生成任务的新数据集，HumanEval（Hand-Written Evaluation Set）
    - 说明：包含了164个编程问题，包含语言理解、算法、简单数学、编程面试问题等。
    - 链接：https://github.com/openai/human-eval

## 评估方法
对于序列生成任务，一般通用方法是：BLEU。

由于BLEU是基于模糊匹配来计算生成准确率，而在代码生成上要求的是整体的准确性。故此方法不适用于评估本场景。

本论文提出了一种新的评估方法：pass@k
- **逻辑**：
    - 1.生成k个结果，只要有一个结果通过unit test（单元测试），则认为生成正确。
- **流程**：
    - 1.对于一个问题，先生成n个结果，再从中抽取k个结果（n≥k）
    - 2.计算n个结果中，通过unit test的结果数为c
    - 3.计算评估指标

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-func.png width=50% />
</div>


- **解析**：
    - 对于单个问题，所有生成结果中，至少存在1个正确结果的概率，并求在所有问题下的均值
    - 即1 - k个结果中无一个结果是正确的概率，则得到至少存在1个正确结果的概率。

## 效果

**与GPT3的效果对比**：明显看出codex(s)效果比GPT3好许多。
- GPT3：最好结果中，top1 acc=0
- Codex：最好结果中，top1 acc=28.8%+
- Codex-S：最好结果中，top1 acc=37.7%
- Codex-S + mean log-probability排序（100个生成结果）：最好结果中，top1 acc=44.5%
- Codex-S + 通过unit test（100个生成结果）：最好结果中，77.5%

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-test1.png width=50% />
</div>


**与其他GPT模型对比**
- 在其他开源数据集 APPS Dataset上进行效果对比 
- 可以看出，同等模型参数量级下，codex效果更好 
- 达到同样效果下，codex参数量更小 

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-test2.png width=50% />
</div>

**Codex与Codex-s的对比**
- 在top1 acc和top100 acc下，Codex-s（加入监督数据微调的模型）效果相对较好。
- 对生成结果中，使用排序规则(unit test至少通过1条)，相对其他排序方法的指标更优。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-test3.png width=50% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/codex-test4.png width=50% />
</div>


## 局限
- **样本有效性**
    - 需要用很多代码的训练数据，才能生成比较简答的代码 
- **prompt问题**
    - 当prompt，或docstring（代码注释）越长，生成效果会下降 
- **数学问题**
    - 对于数学类的理解和生成效果不太好 


## Reference
https://platform.openai.com/docs/guides/code

https://www.bilibili.com/video/BV1iY41137Zi/?spm_id_from=333.337.search-card.all.click


