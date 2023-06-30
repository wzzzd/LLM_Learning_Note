# GPT-1: 《Improving Language Understanding by Generative Pre-Training》

## 论文信息
OpenAI于2018年提出了GPT-1，论文

- [《Improving Language Understanding by Generative Pre-Training》](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)


## 范式
属于pre-training+fine-tuning范式。

第一阶段，使用transform-decoder结构，基于自监督的自回归方式，训练一个单向的语言模型。

第二阶段，添加专属于下游任务的网络层，通过Fine-tuning的方式，拟合下游数据，微调模型。


## 模型结构

### 整体结构
以transform-decoder作为block，构造的网络：

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt1-structure.png width=45% />


### 基本结构
(1) 分词
- 采用BPE (bytepair encoding) 作为分词方法，vocab size 达4w。

(2) 位置编码
- position embedding，size为512

(3) layers
- 12层transformer layers
- 选取了transformer decoder作为基本层结构
    - 原始transformer decoder包含两个multi-head attention结构
    - 此处只保留了masked multi-head attention，去掉了multi-head attention
- multi-head attention 是768维，12head
- 激活函数，用的是GELU (Gaussian Error Linear Unit)


## 训练数据
1. 使用了BooksCorpus dataset进行预训练
2. 1B Word Benchmark 跟ELMo用的一样数据


## 学习目标

### 预训练阶段
学习任务是根据当前文本 $w_{1:i}$ ，预测下一个单词 $w_{i+1}$ 。
优化目标为最大化似然函数
$$L_{1}(X)=\sum\limits_{i} logP(x_i|x_{i-k},...,x_{i-1};\Theta) $$
其中，$x$是token，$k$是上下文窗口大小，$\Theta$是模型参数。


### 微调阶段
学习任务是根据下游数据进行拟合，目标是最大化条件概率（交叉熵），根据不同的任务，选取不同的输出进行loss计算：
- token classification任务：选取每个token的输出，经过线性转换后，计算loss
- sentence classification任务：选取首个token的输出，经过线性转换后，计算loss

总体而言，微调阶段的损失函数形式，如下
$$P(y|x_1,...,x_m)=softmax(h_lW_y)$$
$$L_{2}(X)=\sum\limits_{x,y} logP(y|x_1,...,x_m;\Theta)$$
下游任务在微调时，为了防止模型将预训练任务中学习到的知识遗忘，往往可以加上预训练任务，构造联合学习目标
$$L_{3}(X)=L_{1}(X)+\lambda L_{2}(X)$$
其中，$\lambda$是超参，取值区间为[0,1]。


## 下游任务使用方式

(1) 分类任务
- 句子前后加起始和结束符，(\<s>,\<e>)

(2) 句子蕴含
- 句子前后加起始和结束符，(\<s>,\<e>)
- 两个句子之间加分隔符 $

(3) 文本相似性
- 句子前后加起始和结束符，(\<s>,\<e>)
- 分别颠倒两个句子的输入顺序，并在句子之间加分隔符（让模型知道，该任务与句子顺序无关）

(4) QA & 阅读理解
- 句子前后加起始和结束符，(\<s>,\<e>)
- 对于context、question、answer(多个)，拆分answer成多个，分别构造多条训练样本，其中每条训练样本只带有一个answer [z;q;$;ak]

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt1-use.png width=100% />



## 优点
- 易于并行训练（teacher-forcing方式），捕获长距离特征能力加强
- 使用了两阶段模式fine-tuning下游任务
    - 提高了下游任务的表现
    - 加速了下游任务的收敛速度，一般3个epoch就能达到最好的效果
    - 适合用于文本生成类任务，因为GPT是根据上文信息预测下一个单词，与文本生成类任务的过程匹配。
## 缺点
- 语言模型是单向的，不能捕获token下文的特征信息。
- 对于某些类型的任务需要对输入数据结构作调整。

