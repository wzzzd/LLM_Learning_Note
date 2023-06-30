# GPT-2: 《Language Models are Unsupervised Multitask Learners》

## 论文信息
OpenAI于2019年提出了GPT-2，论文

- [《Language Models are Unsupervised Multitask Learners》](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 整体思路
整体网络结构与GPT-1类似，都是decoder-only transform作为编码层block。
预训练目标也和GPT-1相同，都是自监督自回归的训练任务。


## 论文思想
### (1) 语言模型，也是在给序列进行条件概率建模
我们知道，语言模型的建模目标是计算句子所有词的联合概率，可拆解成多个条件概率的乘积
$$p(x)= \prod\limits_{i=1}^np(s_n|s_1,...,s_{n-1})$$

这种建模形式，同样也是在给序列进行条件建模
$$p(s_{n-k},...,s_n|s_1,...,s_{n-k-1})$$

对于任何的有监督训练任务，都是在学习和估计$p(output|input)$，通常会用特定的网络结构进行任务建模。

如果将上述思想进行通用化，能得到这样一个通用的学习范式：基于同一个输入，进行不同任务的建模，即估计$p(output|input, task)$。

基于这种学习范式，可以将所有任务，用有监督的方式，训练单独一个模型，即对单个模型进行多任务学习。如
- 翻译模型的训练样本，可以表示为：(translate to
french, english text, french text).
- 阅读理解的训练样本，可以表示为：(answer the question, document, question, answer).


### (2) 语言模型，是一种无监督多任务学习
相对于有监督的多任务学习，语言模型只是不需要显式地定义哪些字段是要预测的输出；

实际上有监督的输出，只是语言模型序列中的一个子集。举个栗子：
- 当训练样本中出现这么一些句子，在语言模型完成学习后，也就自然学习了不同的监督任务。
    - 翻译任务（英->法）：“I hate the word ‘perfume,”’ Burr says. ‘It’s somewhat better in French: ‘parfum.’
    - 问答任务：姚明现在是中国篮球协会主席。

## 模型结构

### 整体结构
整体结构基本上与GPT-1一致

不过，论文提供了4种不同尺寸（M->milion，指参数量），最大的尺寸模型，叫做GPT2。

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt2-param.png width=45% />



### 基本结构
整体结构基本上与GPT-1一致，以下阐述不同之处：

(1) input representation
- 如果采用word-level做embedding，需要解决OOV问题
- 如果采用char-level做embedding，模型效果没有word-level好
- 选择了一种折中的办法，将罕见词拆分为子词，类似BPE，参考《Neural Machine Translation of Rare Words with Subword Units》

(2) Layer Norm
- Layer Norm移动到了每个sub-block前（类似于pre-activation residual network），在每个self-attention之后额外添加了一个Layer Normalization

(3) 残差层
- 残差层的参数初始化，用进行缩放，N是残差层的个数。根据网络深度进行调节

(4) 其他细节
- 扩大了字典尺寸（从40000提高到50257 token）
- 扩大了输入长度（从512提高到1024）
- batch size为512



## 训练数据
使用了WebText，包含40GB高质量的网络文本。

## 学习目标

可参考[GPT-1](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/gpt-1.md)

### 预训练阶段
预训练目标也和GPT-1相同，都是自监督自回归的训练任务。

### 微调阶段
微调目标也和GPT-1相同，都是极大化条件概率（交叉熵）。

## 参考
- https://zhuanlan.zhihu.com/p/57251615
- https://terrifyzhao.github.io/2019/02/18/GPT2.0%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB.html
