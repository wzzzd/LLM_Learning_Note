# Layer Norm的种类

## Pre-LN

### 论文
[《ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE》](https://openreview.net/pdf?id=B1x8anVFPr)

### 思路
将LN过程移动到Multi-Head Attention和FFN前面进行

### 结论
当使用了Pre-LN后，warmup变成了非必须的操作

Pre-LN能提高Transformer的收敛速度

同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。
- 解释：
    - Pre Norm结构无形地增加了模型的宽度而降低了模型的深度
    - 我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。

### 结构

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/ln-preln-struct1.png width=40% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/ln-preln-struct2.png width=40% />
</div>


## Sandwich-LN
### 论文
[《CogView: Mastering Text-to-Image Generation via Transformers》](https://arxiv.org/pdf/2105.13290.pdf)

### 思路

在text-to-image任务中，Pre-LN的方式并不能带来像在NLP任务中的稳定性。

提出了Sandwich-LN确保了每层输入值在一个合理范围内。

其实就是在Attention和FFN层中，上下分别使用一个Layer Norm。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/ln-sandwich-struct1.png width=60% />
</div>


## DeepNorm
### 论文
[《DeepNet: Scaling Transformers to 1,000 Layers》](https://arxiv.org/pdf/2203.00555.pdf)

### 思路
- 提出了一个新的标准化方程(DEEPNORM)去修改transformer中的残差链接，能训练出1000层的transformer。
- 与Post-LN相比，DeepNorm在进行layer-norm之前会扩大残差连接。
- 相较于Post-LN模型，DeepNet的模型更新几乎保持恒定。
- 在初始化过程中降低了参数的比例。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/ln-deepnorm-struct1.png width=80% />
</div>

### 结构

公式

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/ln-deepnorm-func1.png width=60% />
</div>

- α、β是常数，β作为参数θ_l的权重因子调节参数权重大小
-  G_l是transformer的sub-layer，如attention、FFN等

