# 各种Attention变体

## Self-Attention




## Muti Query Attention (MQA)

### 论文
- [《Fast Transformer Decoding: One Write-Head is All You Need》](https://arxiv.org/pdf/1911.02150.pdf)

### 作用

- 计算量低，加快decoder生成token的速度
    - 训练阶段：由于训练过程是并行的，影响不大
    - 推断阶段：主要减少了推理过程的计算量（qk），加快了生成速度。（减少了矩阵运算的次数）
- 节省显存
    - 每个k，由之前的维度 [batch size, head_num, d_model] 缩减到 [batch size, d_model] 。


### 思路

对所有query矩阵参数，共享一个keys和value矩阵。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-mqa-struct.png width=80% />
</div>

### 优势
- 减少了K/V的参数量
- 提高了推断速度

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-mqa-eff1.png width=60% />
</div>

### 劣势
- 轻微地降低了精度

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-mqa-eff2.png width=60% />
</div>




## Grouped-query attention（GQA）
### 论文
[《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》](https://arxiv.org/pdf/2305.13245.pdf)

### 思路

- 类似MQA，但是K/V矩阵由1个变成m个。当m=1，则为MQA；当m=head，则为MHA。
- 将query分为多个组，每一个组query共享一个K/V矩阵。即变成了多个MQA。
- 对K/V矩阵使用mean pool的方法进行分组/压缩，变为m个K矩阵。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-gqa-struct.png width=70% />
</div>

### 优势

减少了计算量和减少了内存（比MQA多，比MHA少）



## Flash Attention v1

### 论文

[《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》](https://arxiv.org/pdf/2205.14135.pdf)

### 效果

FlashAttention的运行速度比PyTorch标准注意力快 2-4 倍，所需内存减少5-20倍。

### 限制

只能在A100之后的显卡上用，不能用在V100
- 更大更快的 L1 缓存和SRAM能够让它在每个流处理器(SM)上提供相当于 V100 1.5 倍的总容量（192 KB vs. 128 KB）。
- A100 GPU 拥有 40 GB 的高速 HBM2 显存，与 Tesla V100 相比提升了 73%。
- A100 GPU一种新的异步拷贝指令，可以直接从HBM拷贝到SRAM，这大大简化了数据拷贝过程，减少了延迟，并提高了整体性能。
- A100 GPU 在SRAM中提供了硬件加速的 barrier，将 arrive 和 wait 操作分开，可将从HBM到SRAM的异步拷贝与 SM 中的计算穿插起来。

### 目的
- 自注意力机制（self-attention）的时间和存储复杂度在序列长度上属于二次型。所以，若想为其配备更长的上下文背景，需要优化计算过程，期望将计算过程降低到线性或接近线性。
- 现有的优化方法主要集中在两个方面：
    - 降低每秒所执行的浮点运算次数（FLops），
    - 降低显存中对HBO的访问(IO)开销

### 思路

缩减self-attention计算过程的对HBM的IO访问开销。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-flash-struct.png width=70% />
</div>


### 区别
#### 传统self-attention
- 公式

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/atte-flash-func1.png width=70% />
</div>

- 现状
    - 在计算中需要存储中间值S、P、O 到HBM中，再从HBM读取S、P、V，这会极大占用高带宽显存HBM。

#### flash-attention
- 目标：旨在避免从HBM中，读取和写入注意力矩阵
    - 1.在不访问整个输入的情况下计算softmax函数的缩减；
    - 2.在后向传播中不能存储中间注意力矩阵。
- 做法
    - 1.将输入分割成块，并在输入块上进行多次传递，从而以增量方式执行softmax缩减。
        - 计算softmax时候不需要全量input数据，可以分段计算；
    - 2.Flash Attention就提出了不使用中间注意力矩阵，通过存储归一化因子来减少HBM内存的消耗。
        - 反向传播的时候，不存储attention matrix (N^2的矩阵)，而是只存储softmax归一化的系数。




## Flash Attention v2

### 论文
[《FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning》](https://arxiv.org/pdf/2307.08691.pdf)

### 效果
FlashAttention-2在正向传递中实现了约2倍的速度提升，达到了理论最大吞吐量的73%，在反向传递中达到了理论最大吞吐量的63%。

在每个A100 GPU上的训练速度可达到225 TFLOPs/s。

### 做法
- 减少了non-matmul FLOPs的数量（消除了原先频繁rescale）。
    - 虽然non-matmul FLOPs仅占总FLOPs的一小部分，但它们的执行时间较长，这是因为GPU有专用的矩阵乘法计算单元，其吞吐量高达非矩阵乘法吞吐量的16倍。因此，减少non-matmul FLOPs并尽可能多地执行matmul FLOPs非常重要。
- 提出了在序列长度维度上并行化。
    - 该方法在输入序列很长（此时batch size通常很小）的情况下增加了GPU利用率。即使对于单个head，也在不同的thread block之间进行并行计算。
- 在一个attention计算块内，将工作分配在一个thread block的不同warp上，以减少通信和共享内存读/写。







## Flash Attention v3

### 博客

https://crfm.stanford.edu/2023/10/12/flashdecoding.html

### 思想
本文提出了Flash-Decoding，可以在推理过程中，显著加速attention操作（例如长序列生成速度提高8倍）。

其主要思想是最大化并行加载keys和values的效率，通过重新缩放组合得到正确结果。

### 做法

1.将keys和values分成较小的block

2.使用FlashAttention并行计算query与每个block的注意力（这是和FlashAttention最大的区别）。
- 对于每个block的每行（因为一行是一个特征维度），Flash Decoding会额外记录attention values的log-sum-exp（标量值，用于第3步进行rescale）

3.对所有output blocks进行reduction得到最终的output，需要用log-sum-exp值来重新调整每个块的贡献。



