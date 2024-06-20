# 位置编码-绝对位置编码

## 正余弦位置编码
### 论文
[《Atenttion is all you need》](https://arxiv.org/pdf/1706.03762.pdf)


### 结构

在奇数和偶数的位置，分别采用不同的编码函数

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-sincos-func.png width=30% />
</div>

### 性质
由于sin/cos函数性质，位置向量值是有界的，位于[-1, 1]之间。

选用sin/cos的目的：
- 含有周期性和连续性，可以使得不同位置出现不同的值。
- 原论文解释为：其可以让模型相对简单地学习相对位置信息

随着word embedding越深，位置编码的变化频率越平缓，后期趋向于一致。

使用sin+cos交替的目的：
- 让embedding每个维度的值尽可能有所区别，同时不失周期性和连续性。
- 任何两个时间步之间距离反映的信息，在不同长度的句子中尽量做到一致。

若只使用sin，或只使用cos，在短句子中效果还可以；但是在长句中，会导致注意力的混乱，影响生成结果。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-sincos-attr1.png width=60% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-sincos-attr2.png width=60% />
</div>


### 使用
position为何可以直接与token、segment相加，作为模型输入
- 数学角度
    - 融合的一种方式，其实相加和拼接在数学本质上是一样的
    - 区别在于前者先做线性变换再做融合，后者先做融合再做线性变换
- 模型角度
    - 相加在一定程度上保持了三个embedding空间的独立性。
    - 体现了一种特征交叉的方法和信息融合。

### 缺陷
为何后续不用了?
- 外推性能不足。

- 正弦位置编码的相对位置表达能力被投影矩阵破坏掉了。（失去了相对位置的表达能力）
    - 注意力矩阵，可以被拆解为4个矩阵，其中最后一个矩阵包含了相对位置的信息
    - 当把U_i、U_j 移去后，能表达出位置pattern信息，当加上后，得到的投影矩阵看不出明显的位置pattern信息
<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-sincos-func2.png width=30% />
</div>

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-sincos-attr3.png width=60% />
</div>

## 可训练式编码-Bert

结构上，将positon embedding改成可训练的一组参数。
效果上，与transformer中的正余弦位置编码，效果差不多。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos-trainable-struct.png width=60% />
</div>





