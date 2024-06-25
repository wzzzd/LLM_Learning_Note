# 旋转编码-RoPE

## 计算逻辑

### 考虑在2维的情况下

#### 目标

在attention计算公式中，假设Q中当前位置为m，K当前位置为n，输入为x。将Q_m, K_n，分别表示成x’m, x'n

找到一个融合位置编码信息的x’m 和 x'n，假设其为一个关于位置θ的虚数表示

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func1.png width=30% />
</div>

目标是证明x’m和x’n的内积，能表示成相对位置的函数，也就是使得下面的式子成立

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func2.png width=40% />
</div>

#### 证明

1.从内积的角度

对x’m进行展开，得到二维向量转成虚数坐标表示，结合欧拉公式

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func3.png width=70% />
</div>

继续展开x’m，得

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func4.png width=60% />
</div>

同理，可得x’m和x’n的展开结果

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func5.png width=60% />
</div>

两者内积结果，便可得证。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func6.png width=70% />
</div>

2.从设计函数的角度

设计以下函数，输入是x’m，x’n，以及m-n，其中Re表示取虚数的实部

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func7.png width=40% />
</div>

继续代入和推导

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func8.png width=75% />
</div>

对式子3进行展开，得

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func9.png width=75% />
</div>

两部分结果相同，得证。

### 考虑在多维的情况下

由于内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接。

对Q按照两个元素为一组，进行旋转位置操作

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func10.png width=55% />
</div>

由于矩阵中多数元素为0，会导致许多计算是无效的，故可转换为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func11.png width=45% />
</div>

其中θ计算公式为（可以带来一定的远程衰减性）

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_func12.png width=25% />
</div>


## 远程衰减性

RoPE中，内积随着相对距离增大的远程衰减

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/pos_rope_img1.png width=50% />
</div>


## 外推性

1.RoPE为什么能输出超过训练位置长度的位置编码（为什么具有外推性）？

- 总结
    - RoPE 可以通过旋转矩阵来实现位置编码的外推，即可以通过旋转矩阵来生成超过预期训练长度的位置编码。
    - 这样可以提高模型的泛化能力和鲁棒性。
- 具体
    - 假设我们有一个 d 维的绝对位置编码 P_i ，其中 i 是位置索引。
    - 将 P_i 看成一个 d 维空间中的一个点。
    - 定义一个 d 维空间中的一个旋转矩阵 R ，它可以将任意一个点沿着某个轴旋转一定的角度。
    - 用 R 来变换 P_i ，得到一个新的点 Q_i= R *P_i 
    - 想要生成超过预训练长度的位置编码，只需要用 R 来重复变换最后一个预训练位置编码 P_n ，得到新的位置编码
        - Q_{n+1} = R * P_n 
        - Q_{n+2} = R * Q_{n+1} 
        - Q_{n+3} = R * Q_{n+2}  
        - 依此类推
    - 可以得到任意长度的位置编码序列 Q_1, Q_2, …, Q_m  ，其中 m 可以大于 n 。
    - 由于 R 是一个可逆矩阵，它保证了  Q_i  和 Q_j  的距离可以通过 R 的逆矩阵 R^{-1} 还原到  P_i 和 P_j  的距离
        - 即 || R^{-1} * Q_i - R^{-1} * Q_j || = || P_i - P_j || 。
        - 保证位置编码的可逆性和可解释性。

## 有效性

1.RoPE为什么有效？
- (1) 有效地保证位置信息的相对关系。
    - 相邻位置的编码之间有一定的相似性，而远离位置的编码之间有一定的差异性。
    - 增强模型对位置信息的感知和利用。
- (2) 通过旋转矩阵来实现位置编码的外推
    - 通过旋转矩阵来生成超过预训练长度的位置编码。
- (3) 与线性注意力机制兼容
    - 不需要额外的计算或参数来实现相对位置编码。
    - 降低了模型的计算复杂度和内存消耗。

