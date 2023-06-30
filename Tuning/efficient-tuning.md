# Efficient-Tuning

## 说明
由于LLM参数量都是在亿级以上，少则数十亿，多则数千亿。当我们想在用特定领域的数据微调模型时，如果想要full-tuning所有模型参数，看着是不太实际，一来需要相当多的硬件设备（GPU），二来需要相当长的训练时间。

因此，我们可以选择一条捷径，不需要微调LLM的全量参数，而只需要新增少量的参数，通过固定原始模型参数，而只需要微调新增的少量参数，从而达到接近使用全参数full-tuning的效果。

本章主要讲述在LLM时代，当下主流的微调方法。


## 1.Adapter tuning

**(1) 论文信息**

来自2019年，论文《[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)》

**(2) 思路**

- 固定Transformer的全部参数
- 在Transformer的每一个Block里嵌入一些新初始化的Adapter Network。
    - 其中Adapter由两层MLP组成，分别负责将Transformer的表征降维和升维

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/Adapter-tuning.png width=60% />

**(3) 优势**

- 只需要添加不到5%的可训练参数，即可以几乎达到全参数训练的效果 
- 在训练过程中大大节省了训练时间，做到时间有效性。
- 基本不降低模型在下游任务中的表现



## 2.Prefix tuning

**(1) 论文信息**

来自论文《[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf)》

**(2) 思路**

固定预训练参数，为每一个任务额外添加一个或多个embedding，且利用多层感知编码prefix。不再像prompt tuning继续输入LLM。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/Prefix-tuning.png width=80% />
</div>

**(3) 结构**

在seq前面加idx个虚拟token，以此构造一个连续的token，作为微调参数（结构一样是transformer）

固定LLM的参数

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/Prefix-tuning-func1.png width=30% />
</div>

由于发现直接加prefix层，模型不稳定，故在其后加了MLP层，用于reparametrization参数 $P_θ$
$$P_θ[i:]=MLP_θ(P'_θ[i,:])$$

原始$P_θ$维度为$ \mid P_{idx} \mid \times dim(h_{i})$，$P'_θ$维度为$ \mid P_{idx}\mid \times k$，经过$MLP$复原原始维度。

针对不同任务，有不同最优的k值，经过实验，作者建议
- Table-to-table任务，k=512
- Summarization任务，k=800


**(4) 优势**

- 在Table2Text任务上，只有0.1%参数量级的prompt tuning效果要优于微调

**(5) 缺点**

- 摘要任务上，prompt的效果要略差于微调


## 3.Prompt tuning

**(1) 论文信息**

来自论文《[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)》

**(2) 思路**

固定预训练LLM的参数，为每一个任务额外添加一个或多个embedding。之后拼接query正常输入LLM，并只训练这些embedding

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/prompt-tuning.png width=100% />
</div>

**(3) 优势**

- 效果优于GPT-3的few-shot learning
- 当模型参数量达100亿时，接近于全模型微调效果



## 4.P-tuning

**(1) 论文信息**

来自论文《[GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)》，发表于2021年

**(2) 思路**

固定LLM参数，用多层感知机和LSTM对prompt进行编码，编码后与其他向量进行拼接，正常输入LLM。

注意，训练之后只保留prompt编码之后的向量，无需保留编码器。

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/P-tuning.png width=100% />

**(3) 结构**

对于prompt模板，使用MLP+LSTM进行编码，替代原始的input embedding

对于原始的输入input和target，则使用原始的input embedding

**(4) 使用方式**

离散和连续template token混合时，显示地插入一下anchor（离散的token）有助于template的优化

**(5) 优势**

- 能缓解离散prompt方法，导致的模型输出结果到达局部最优

**(6) 缺点**

- 查找的最优提示，可能是次优的
- 在小参数量模型中表现差（小参数模型如Bert，330M），上了10B的模型效果才开始可以持平
- 序列标注等对推理和理解要求高的任务，prompt-tuning效果会变差



## 5.P-tuning v2

**(1) 论文信息**

来自论文《[P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)》，发表于2022年。

**(2) 思路**

- 固定LLM参数
- 类似Prefix-tuning
    - 在Deep FT层：在seq前面加n个虚拟token，以此构造一个连续的token，作为微调参数（结构一样是transformer）
- 在多种任务上下进行微调
- 完全变为生成模型，无需verbalizer

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/P-tuning-v2.png width=100% />

**(3) 优势**

- 在小、大模型上，效果均优于P-tuning。
- 当参数量达10B，效果相当于FT


## 6.LoRA

**(1) 论文信息**

来自论文《[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)》

**(2) 思路**

固定LLM参数，在每一个self-attention层中，加入一个low-rank的矩阵，即$B \times A$。在微调时，只更新$B \times A$的参数。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/lora.png width=50% />
</div>

**(3) 结构**

在每一层self-attention中，添加新的参数 $\bigtriangleup W$

$$h=W_0x+\bigtriangleup Wx=W_0x+BAx$$

其中，预训练模型的原始参数为$W_0 \in R^{d \times k}$。

LoRA的新增参数为$B \in R^{d \times r}$，$A \in R^{r \times k}$。$B$ 初始化为一个全0矩阵，$A$ 是一个高斯随机初始化的矩阵。$B$ 初始化为全0矩阵的目的是，在开始训练时，让$B \times A$等于0矩阵，即参数从0开始。

其中LoRA的中间维度$r$，远小于原始模型的维度，即$r\ll min(d,k)$

**(4) 学习目标**

原始的LLM，一般也是CLM (Causal Language Model/Conditional Language Model)，学习目标为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/lora-func1.png width=23% />
</div>

而加入LoRA后，学习目标为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/tuning/lora-func2.png width=30% />
</div>

**(5) 配置**

- 在多个部位$(Q/K/V/Output)$同时添加$\bigtriangleup W$ ，会比只在单一部分上添加权重$\bigtriangleup W$，效果要好
- 在wikiSQL/MultiNLI数据集上测试得出结论：小的γ值，能达到较好好的效果（一般为4-8）


**(6) 优势**

- 用于低资源的场景。也就是硬件设备资源有限的情况下。
- 更新参数量少。对于175B的GPT模型，使用该方法只需要额外更新0.01%的参数量。
- 是全参数微调（FT）的一种替代方案


**(7) 缺点**

- 全参数微调（FT）效果要比LoRA稍微要好


