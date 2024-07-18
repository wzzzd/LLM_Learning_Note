# 分布式训练方法

如今的大模型训练，离不开各种分布式的训练框架，一般来说，并行策略包含：数据并行、模型并行、流水线并行。

## 1.数据并行
数据并行分为了两种模式：Data Parallel（DP）和 Distributed Data Parallel（DDP）。

### **Data Parallel（DP）**
DP是一种单进程多线程的并行策略，只能在单机上进行训练，步骤如下：
* 单进程控制多GPU，即本质上是单进程多线程；
* 首先将模型加载到主 GPU 上，再复制到各个指定从 GPU；
* 将输入数据按照 Batch 维度进行拆分，各个 GPU 独立进行 forward 计算；
* 将结果同步给主 GPU 完成梯度计算和参数更新，将更新后的参数复制到各个 GPU。
由于其是单进程控制多个GPU，故会存在GPU之间负载不均衡的问题，主GPU负载较大。

### **Distributed Data Parallel（DDP）**

DDP采用 AllReduce 架构，多进程的方式，突破锁的束缚。在单机和多机上都可以使用。

负载分散在每个 GPU 节点上，通信成本（时间）是恒定的，与 GPU 数量无关，等于V/B（参数量/带宽）。

DDP不需要通过主GPU分发全模型的参数到每个GPU上。

使用ring-all-reduce的方式进行通讯，随着 GPU 数量 N 增加，总传输量恒定。也就是理论上，随着GPU数量的增加，ring all-reduce有线性加速能力。


## 2.张量并行

张量并行的原理是，将张量操作划分到多个设备上，以加速计算或增加模型大小；对模型每一层的层内参数进行切分，即对参数矩阵切片，并将不同切片放到不同GPU上；将原本在单卡中的矩阵乘法，切分到不同卡中进行矩阵乘法。训练过程中，正向和反向传播计算出的数据通过使用 All gather 或者 All reduce 的方法完成整合。

以transformer为例，该策略会把 Masked Multi Self Attention 和 Feed Forward 都进行切分以并行化。利用 Transformers 网络的结构，通过添加一些同步原语来创建一个简单的模型并行实现。

张量并行适用于模型单层网络参数较大的情况。同时缺点也是十分明显：
* 若环境是多机多卡，张量并行所需的all-reduce通信需要跨服务器进行链接，这比单机多GPU服务器内的高带宽通信要慢；
* 高度的模型并行会产生很多小矩阵乘法，这可能会降低GPU的利用率。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/parallel/parallel-tensor-process.png width=50% />
</div>


## 3.流水线并行

流水线原理是将不同的 layer 分配给指定 GPU 进行计算，流水线并行只需其之间点对点地通讯传递部分 activations。

具体步骤包括：
* 在流水线并行之中，一个模型的各层会在多个GPU上做切分。
* 一个批次（batch）被分割成较小的微批（microbatches），并在这些微批上进行流水线式执行。
* 通过流水线并行，一个模型的层被分散到多个设备上。
* 当用于具有相同transformer块重复的模型时，每个设备可以被分配相同数量的transformer层。
* 在流水线模型并行中，训练会在一个设备上执行一组操作，然后将输出传递到流水线中下一个设备，下一个设备将执行另一组不同操作。

流水线并行的方法，解决了超大模型无法在单设备上装下的难题，也解决了机器之间的通信开销的问题，使得每台机器的数据传输量跟总的网络大小、机器总数、并行规模无关。

常用的流水线方法有G-pipe、1F1B、Interleaved 1F1B等。

### 朴素实现

每次只有一张卡在运行，P个阶段的气泡占比= (P-1) / P。

但是存在显卡整体使用率不高的缺点。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/parallel/parallel-pipeline-nature.png width=70% />
</div>

### G-pipe

将输入分为若干micro batch流水线计算，以减少气泡。

P个阶段，M个输入的气泡占比=  (P-1) / (P -1 + M)。

该方法提升micro batch数量，会得到更小气泡占比，但是显存占用更大。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/parallel/parallel-pipeline-gpipe.png width=70% />
</div>


### 1F1B

该方法主旨是尽早进行backward，占用内存只和流水线段数有关。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/parallel/parallel-pipeline-1f1b.png width=70% />
</div>


### Interleaved 1F1B

将网络层交错切分，变相增加了micro batch数量。

以更多的通信量进一步降低了气泡占比。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/parallel/parallel-pipeline-inter1f1b.png width=70% />
</div>


