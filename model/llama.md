# Llama v1/v2


## Llama v1

### 论文信息
Meta于2023年发布，论文

- [《LLaMA: Open and Efficient Foundation Language Models》](https://arxiv.org/abs/2302.13971)

### 整体思路

预训练任务基于casual LM的学习目标，即自回归学习

### 模型尺寸

发布的模型版本包括：7B、13B、32B、65B

上下文长度支持2048

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/gpt3-size.png width=80% />


### 模型结构

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama-architecture.png width=80% />

基于transformer decoder，即decoder-only架构。
- Pre-normalization
    - 为了提高训练稳定性，对每个 transformer 子层的输入进行归一化，使用 RMSNorm 归一化函数
- SwiGLU
    - 将 ReLU 非线性替换为 SwiGLU 激活函数，且使用  2/3 * 4d，而不是 PaLM 论文中的 4d
    - 原理：
        - 基于门控线性单元(GLU)的变体

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama-func1.png width=80% />

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama-func2.png width=80% />

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama-func3.png width=80% />

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama-func4.png width=80% />

- Rotary Embeddings
    - 模型的输入不再使用 positional embeddings，而是在网络的每一层添加了 positional embeddings (RoPE)

### 数据

预训练使用了1T-1.4T token（万亿）的数据，其中不同数据集占比如下

<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/llama1-token_size.png width=80% />

分词器使用 BPE

词表大小为 32K


### 训练时间
- 65B的模型，在2048个A100 GPU和80GB的内存上处理大约380个token/秒/GPU
- 1.4T token的数据集上进行训练大约需要21天，大部分数据1个epoch，少量数据2个epoch


### 优势
- 只是用开源的数据集进行预训练，保证了工作可复现
- 性能好
    - LLaMa-13B的性能媲美GPT3-175B
    - LLaMa-65B性能媲美Chinchilla 或 PaLM-540B




## Llama v2

### 整体思路
相对Llama1，更新点如下：
- 预训练语料从1->2 Trillion tokens
- Context Window 长度从2048->4096
- 收集了100k人类标注数据进行SFT（llama2-chat）
- 收集了1M人类偏好数据进行RLHF（llama2-chat）
- 在reasoning, coding, proficiency, and knowledge tests上表现超越MPT和Falcon
- 和falcon一样，使用了**Group query attention**，节省cache

### 论文信息

于Llama1后，几个月发布 [《Llama 2: Open Foundation and Fine-Tuned Chat Models》](https://arxiv.org/abs/2307.09288)


### 模型尺寸

发布的模型版本包括：7B/13B/34B/70B

### 模型结构

与Llama1一样



## 参考
-  https://mp.weixin.qq.com/s/qTJp_hBDiIvJ-ymxLr-Pmg
- https://zhuanlan.zhihu.com/p/63678464
- https://zhuanlan.zhihu.com/p/644440986




