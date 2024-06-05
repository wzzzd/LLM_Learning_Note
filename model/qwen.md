# Qwen系列模型


## Qwen 1

### 论文
[Qwen Technical Report](https://arxiv.org/pdf/2309.16609)

### 时间
2023.08

### 参数量
7B、14B、VL、1.8B、72B

### 输入长度
2048

### 模型结构
整体结构与llama类似
- 1）使用无限制嵌入untied embedding，
- 2）使用旋转位置嵌入，
- 3）除了注意力中的 QKV 之外没有偏差，
- 4）RMSNorm， 而不是 LayerNorm，
- 5）SwiGLU 而不是 ReLU，以及
- 6 ）采用闪光注意力来加速训练。该模型有32层，嵌入维度为4096，注意力头数量为32。

与llama的异同
- 1）计算Attention时，QKV的 matmul计算添加了Bias；llama没有Bias
- 2）Tokenizer 采用 titoken 来实现； llama 使用 sentencepiece 来实现
- 3）Embedding table 需要进行转换，才能与 llama 的 Embedding table 维度对齐
- 4）对layer参数的描述与llama不同

### 参考
- https://zhuanlan.zhihu.com/p/675332872
- https://blog.csdn.net/sinat_37574187/article/details/132144918


## Qwen 1.5

### 时间
2024.02

### 参数量
0.5/1.8/4/7/14/72B

### 输入长度
32k

### 模型结构
与Qwen1类似，整体结构与llama类似

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/qwen1.5-struct.png width=80% />
</div>

### 参考
- https://zhuanlan.zhihu.com/p/682602547
