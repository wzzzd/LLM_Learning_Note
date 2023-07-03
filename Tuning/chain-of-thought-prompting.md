# Chain-of-Thought(CoT)

## 1.介绍

在过去几年的探索中，业界发现了一个现象，在增大模型参数量和训练数据的同时，在多数任务上，模型的表现会越来越好。因而，现有的大模型LLM，最大参数量已经超过了千亿。

然而，增大模型参数规模，对于一些具有挑战的任务（例如算术、常识推理和符号推理）的效果，并没有太大提升。对于算术类推理任务，我们期望模型生成自然语言逻辑依据来指导并生成最终答案，但是获得逻辑依据是比较复杂昂贵的（标注成本层面）。

自从发现了大模型ICL（In-Context Learning）的能力后，这个问题有个新的解决思路：对某个Task，能否为大模型提供一些上下文in-context example作为Prompt，以此来提升模型的推理能力？实验表名，在复杂推理任务上加入ICL带来的增益不明显。因此，变衍生出了CoT的技术。

Chain-of-Thought(CoT)是一种改进的Prompt技术，目的在于提升大模型LLMs在复杂推理任务上的表现，如算术推理（arithmetic reasoning）、常识推理（commonsense reasoning）、符号推理（symbolic reasoning）。


## 2.思路

ICL的思路是在新测试样本中加入示例（demonstration）来重构prompt。

与ICL（In-Context Learning）有所不同，CoT对每个demonstration，会使用中间推理过程（intermediate reasoning steps）来重新构造demonstration，使模型在对新样本预测时，先生成中间推理的思维链，再生成结果，目的是提升LLM在新样本中的表现。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-prompt.png width=50% />
</div>


## 3.CoT方法

一般来说CoT会分为两种：基于人工示例标注的Few-shot CoT和无人工示例标注的Zero-shot CoT。下面将逐一介绍。


### 3.1 Few-shot CoT

假设基于ICL的测试样本输入表示为$<input, demonstrations>$，那么加入Few-shot CoT的测试样本输入，可表示为$<input, CoT>$。

#### 3.1.1 CoT Prompt设计

我们知道了加入CoT的示例后，能提升LLM的表现。那么我们应该如何构造或使用CoT？

##### 投票式CoT

**《Self-Consistency Improves Chain of Thought Reasoning in Language Models》**

论文基于一个思想：一个复杂的推理任务，其可以有多种推理路径（即解题思路），最终都能够得到正确的答案。故Self-Consistency在解码过程中，抛弃了greedy decoding的策略，而是使用采样的方式，选择生成不同的推理路径，每个路径对应一个最终答案。

具体做法为：
- 对于单一的测试数据，通过多次的解码采样，会生成多条推理路径和答案。
- 基于投票的策略，选择最一致的答案。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-self-consistency.png width=50% />
</div>

实验表明，对于同一问题生成更多的推理链以供投票往往能取得更好的效果。当推理链数量足够多时，这种方法效果能够胜过使用greedy decoding的CoT方法。


**《On the advance of making language models better reasoners》**

论文在Self-Consistency的基础上，进一步做了优化。

- 1.Diverse Prompts
    - 对于每个测试问题，构造了$M_1$种不同的prompt(即由不同demonstration构造的prompt)
    - 对于每种不同的prompt，让LLM生成$M_2$条推理路径。
    - 则对于同一个测试问题，共生成了$M_1*M_2$条结果
- 2.Verifier
    - 训练了一个Verifier，用于判断当前推理路径得出的答案正确与否。
    - 关于样本构建，使用LLM生成的推理路径和答案，与grandtruth进行对比，一致的即视为正样本，否则负样本。
- 3.Vote
    - 训练好Verifier后，对与一个测试问题与LLM生成的多条推理路径，Verifier进行二元判别
    - 结合判别结果和投票结果，得出模型的最终预测。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-diverse.png width=50% />
</div>

实验结果显示，本论文的方法相对基于Greedy Decode和Self-Consistency能得到更优的效果。



##### 使用复杂的CoT


**《Complexity-based prompting for multi-step reasoning》**

面对这么多可选的CoT，简单的CoT示例和复杂的CoT示例，对新的样本推理结果会不会产生影响？答案是Yes。

论文探讨了一个问题，在包含简单推理路径的demonstrations和复杂推理路径的demonstrations下，哪个效果会表现较好？（这里的简单和复杂是指 推理链/推理步骤的长度）

本论文继承了Self-Consistency的思想，具体方法：
- 1.对于同一个测试问题，使用功能LLM（GPT-3）生成$N$条不同的推理链+答案；
- 2.对于生成的推理链+答案，按照推理链的长度进行倒序排序；
- 3.保留TopK条推理链+答案，并使用投票的方式，选取最终预测。

实验结果表明，本论文的方法效果优于以下方法： (1)人工构建Cot、(2)random Cot、(2)Complex CoT（数据集中最长的多个思维链作为demonstrations）。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-complexity-base.png width=70% />
</div>


##### 自动构建CoT

**《Automatic chain of thought prompting in large language models》**

上面提到的方法是基于人工构造CoT，那我们能否让模型自己来生成CoT？本论文就提供了这样一种自动生成CoT的思路。

本论文提到的Manual-CoT，可以等同于Few-shot CoT来理解。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/icl-demo-format-autocot-func.png width=70% />
</div>


由于Zero-Shot-CoT方法存在不稳定性，而Manual-CoT方法需要大量人工成本投入。作者提出了一种基于Auto-CoT的方法，自动构建包含问题和推理链的说明样例(demonstrations)。

整个过程分了两个阶段：

1.question cluster: 目的是将数据集中的question划分到不同簇中。
- 使用Sentence-Bert计算每个question的向量表示；
- 使用k-means方法将question记性簇划分；
- 最后对每个簇中的question，根据距离中心点距离，升序排序。

2.demostration sampling: 目的是从每个簇中选取一个代表性的question，基于LLMs，使用Zero-Shot-CoT生成推理链。
- 对于每一个簇$i$里的每一个问题$q^{(i)}_j$，使用Zero-Shot-CoT的方法，将$[Q:q^{(i)}_j,A:[P]]$（其中$[P]$表示"Let's think step by step"）输入到LLMs，LLMs生成该问题的推理链$r^{(i)}_j$和答案$a^{(i)}_j$；
- 若问题$q^{(i)}_j$不超过60个tokens，且推理链$r^{(i)}_j$不超过5个推理步骤，则将问题+推理链+答案，加入到demostrations列表中:$[Q:q^{(i)}_j,A:r^{(i)}_j。a^{(i)}_j]$；

- 遍历完所有簇，将得到k个demostrations，将其拼接上测试question，构造成新的Prompt，输入LLMs便可得到生成结果。

值得一提的是，Auto-CoT在多个开源推理任务的数据集上，效果与Manual-CoT相当，甚至某些任务表现得更好。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/icl-demo-format-autocot.png width=70% />
</div>

<br>


##### CoT中示例顺序的影响

**《Chain of thought prompting elicits reasoning in large language models》**

尽管CoT是ICL的一种特殊形式，但是与ICL有所不同的是，CoT中demonstrations的排序对其在新测试样本中的生成结果影响较小，论文对demonstrations进行重排序，在多数推理任务上仅导致小于2%的性能变化。（demonstrations顺序对ICL影响较大）


#### 3.1.2 CoT的增强策略


### 3.2 Zero-shot CoT

与Few-shot CoT不同，Zero-shot CoT并不需要人为构造demonstrations，只需要在prompt中加入一个特定的指令，即可驱动LLMs以思维链的方式生成结果。

当然这种不需要人工构造demonstrations的方式，效果相对Few-shot CoT会表现稍微差一点点。但是相对Zero-shot和Few-shot的方法而言，Zero-shot CoT在复杂任务推理上却能带来巨大的效果提升。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-compare.png width=60% />
</div>


**《Large language models are zero-shot reasoners》**

论文首先提出了Zero-shot CoT的方法，整个流程包含两部分：
- 1.Reasoning Extraction
    - 使用一个特定的"reasoning" prompt，是语言模型LLM生成原始问题的思维链，如"Let's think step by step."（让我们一步步来思考）
- 2.Answer Extraction
    - 基于第一步的结果，添加一个"answer" prompt，要求LLM生成正确的结果。
    - 这一个步骤中，LLM的输入格式为：quesiton + "reasoning" prompt + result(CoT) + "answer" prompt，输出为：result(answer)

值得一提的是，论文同时发现了，当模型LLM变得越来越大，对于使用Zero-shot的结果带来的增益不大，但是对使用Zero-shot CoT的结果带来的增益较大。


<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-zeroshot-step.png width=50% />
</div>


**《Scaling Instruction-Finetuned Language Models》**

既然在上一篇论文中，已经发现了LLM存在Zero-shot CoT的能力，那如果事先对LLM进行基于CoT的instruction tuning，那模型使用Zero-shot CoT方式在对unseen样本进行预测时，效果会不会更好？本论文给出了肯定的答案。

论文探索了以下可能影响LLM在unseen task上表现的因素：
- 1.任务数量
- 2.模型大小
- 3.指令微调（instruction tuning）

论文微调数据集包含了1836种指令任务，473个数据集和146种任务类型构成，数据集中包含了9个人工标注的CoT数据集。同时保留一个没出现过的held-out数据集作为模型评估数据集。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-zeroshot-scale-data.png width=50% />
</div>


<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/cot-zeroshot-scale-cot.png width=50% />
</div>


使用的模型是PaLM，而经过instruction tuning的模型，称为FlanPaLM（Finetuned Language PaLM）。

得到了以下结论：
- 1.增加微调任务数量，可以提高LLM表现。但任务数量超过一定值后，不管模型尺寸是否增大，受益都不大。推测原因有：
    - (1) 额外的任务多样化不足，没有为LLM提供新的知识；
    - (2) 多任务指令微调只是更好地激发了模型从预训练任务中学习到知识的表达能力，而微调任务超过一定值后，对表达能力没有太大帮助。
- 2.微调和未微调的PaLM，从8B增大到540B，在unseen任务上效果越来越好；
- 3.微调数据与CoT数据的关系
    - (1) 微调数据中删除CoT数据，会降低PaLM的推理能力
    - (2) 微调数据包含CoT数据，会全面提高所有评测任务的表现


## 5.总结

对于大模型LLM涌现的CoT能力，业界目前的共识是：当模型参数超过100B后，在复杂推理任务中使用CoT是能带来增益的；而当模型小于这个尺寸，CoT并不会带来效果增益。

还记得在Pretrain+Fine-tuning时代下，对于复杂数学推理任务，如MultiArith、GSM8K下，效果还是不太理想，而短短几年时间，LLM+CoT的模式已经大大提升了该领域的解决能力。随着LLM的继续发展，未来必定会发现更多LLM隐藏的能力和使用方法，让我们拭目以待。


## 6.Reference


[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)

[Large language models are zero-shot reasoners](https://arxiv.org/pdf/2205.11916.pdf)

[Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)

[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/pdf/2203.11171.pdf)

[On the advance of making language models better reasoners](https://arxiv.org/pdf/2206.02336.pdf)

[Chain of thought prompting elicits reasoning in large language models](https://arxiv.org/pdf/2201.11903.pdf)

[Complexity-based prompting for multi-step reasoning](https://arxiv.org/pdf/2210.00720.pdf)

[Chain of thought prompting elicits reasoning in large language models](https://arxiv.org/pdf/2201.11903.pdf)



