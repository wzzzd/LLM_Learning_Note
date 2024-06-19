
# Mistral 7B

## 时间
- 2023.10
## 参数量
- 7B

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mistral-args.png width=20% />
</div>

## 输入长度
- 8192
## 模型结构
- 滑动窗口注意力

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mistral-atte1.png width=60% />
</div>

- 滚动缓存

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mistral-atte2.png width=60% />
</div>

- 预填充与分块

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mistral-atte3.png width=60% />
</div>



# Mixtral 8x7B MoE
## 时间
- 2024.2
## 参数量
- 8x7B

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-args.png width=20% />
</div>

## 输入长度
- 32k
## 模型结构

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-struct.png width=60% />
</div>

对于给定的输入 x，MoE 模块的输出由专家网络输出的加权和决定，其中权重由门控网络的输出给出。

在每一层，对于每个token，路由器网络选择其中的两个组(“专家”)来处理token并通过组合相加得到它们的输出。

专家层
- 每一个专家都是一个transformer-decoder层
- 使用线性层的 Top-K logits，输出k个专家的结果，论文k=2

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-func1.png width=15% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-func2.png width=25% />
</div>


- 整体输出
<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-func3.png width=35% />
</div>

单层Mixtral模型结构
- Expert是指单层decoder层中的FFN层。
- FFN层其实就是SwiGLU层，即一个专家表示一个SwiGLU层
- 在各个层中仅有experts部分(FFN)是独立存在的，其余的部分(Attention等)则是各个expert均有共享的
- 理解：每一层decoder中的FFN层，都包含8个FFN专家，8个专家拥有各自的参数，剩下的如Attention相关参数是共享的。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/mixtral-struct2.png width=30% />
</div>





