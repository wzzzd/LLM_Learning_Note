# GLM: General Language Model Pretraining with Autoregressive Blank Infilling》


## 论文：
* [《GLM: General Language Model Pretraining with Autoregressive Blank Infilling》](https://arxiv.org/pdf/2103.10360.pdf)


## 模型结构
同transformer结构，属于encoder-decoder模型
重新排列层和残差连接的顺序，
GeLUs替换ReLU激活函数
二维位置编码
* 第一维为原始的token顺序index
* 第二维，除开原始输入外，每一个segment的index为从1到segment长度

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/glm-130b.png width=40% />
</div>

## 学习目标
主要采用自回归的方式，填充空白token。

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/glm-func1.png width=40% />
</div>


* 对于input token，随机抽取m个span，并使用一个[mask] 占位符替换该span
* 在预测时，打乱被mask的span顺序
* 其中segment的预测公式为

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/glm-func2.png width=40% />
</div>

* 自注意力掩码
    * PartA在训练时，可以双向看到自身的信息
    * PartB在训练时，可以看到PartA的信息，以及PartB中已经被生成的信息。（看不到未来的信息）


## 学习过程
* NLU
    * 对模型预测的token，映射成positive或negtive，再计算CE loss

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/glm-nlu.png width=40% />
</div>

* NLG
    * 当成CLM任务，计算负对数损失

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/glm-nlg.png width=40% />
</div>

## 参考
* https://zhuanlan.zhihu.com/p/637382548


