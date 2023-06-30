# GPT-3: 《Language Models are Few-Shot Learners》

## 论文信息
OpenAI于2020年提出了GPT-3，论文

- [《Language Models are Few-Shot Learners》](https://arxiv.org/abs/2005.14165)

## 整体思路
整体网络结构与GPT-2类似，都是decoder-only transform作为编码层block。

预训练目标也和GPT-1/2相同，都是自监督自回归的训练任务。

主打一个思想：**用更少的领域数据，且不经过精调步骤去解决问题。**

GPT3试图解决Bert类AE模型的两个缺点：
- **1.对领域内数据的过分依赖。**
虽然有了预训练+精调的两段式框架，但还是少不了一定量的领域标注数据，否则很难取得不错的效果，而标注数据的成本又是很高的。
- **2.对领域数据分布的过拟合。**
在精调阶段，因为领域数据有限，模型只能拟合训练数据分布。如果数据较少的话就可能造成过拟合，致使模型的泛化能力下降，更加无法应用到其他领域。

## 模型尺寸

模型尺寸最大达到了1750亿参数量

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt3-size.png width=80% />


## 数据

预训练使用了45TB的数据，其中不同数据集占比如下

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt3-data.png width=80% />


### 提出概念

整体上，根据论文阐述可以看出GPT3与GPT2除了在训练数据量和模型尺寸方面有所差别，其他方面几乎是一致的。但OpenAI的作者们，提出了关于GPT3新的思想

**(1) Zero-Shot**

没有示例演示，仅向模型提供描述任务的自然语言指令（instruction），让模型输出根据指令输入生成的结果。在生成期间，模型没有权重更新，即这是一个推断过程。

**(2) One-Shot**

给出一个学习示例，即包含输入输出信息，作为上下文信息。在加入一个新的输入，一并输入到GPT3。同样是一个推断过程，不需要更新模型参数。

这个过程和人类处理任务最相似，要处理一件事情，如果先有一个示例给你学习，解决问题时会更加知道往哪个方向着手。

**(3) Few-Shot**

指的是在推理时对模型进行一些任务相关的示例演示，同样不需要更新模型参数。对于一个典型的数据集，一个示例具有上下文和所需的补全（例如英语句子和对应的法语句子），并通过给出K个示例上下文和补全的例子进行了Few-Shot。通常将K设置在10到100的范围内。

FS的主要优点是，大大减少了对特定任务数据的需求，并减少了过拟合的可能性。主要缺点也比较显而易见，这种没有经过参数更新的方式，得到结果要比最新的微调模型差很多，而且，仍然需要少量的任务特定数据。

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt3-use.png width=100% />


## 参考
- https://mp.weixin.qq.com/s/GvCp1KrZY1YDByFHHrYPqA
- https://blog.csdn.net/weixin_41089007/article/details/106501248
