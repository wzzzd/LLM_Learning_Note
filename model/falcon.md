# Falcon系列模型

## Falcon 1

### 论文
[The RefinedWeb Dataset for Falcon LLM:
Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/pdf/2306.01116)

### 时间
2023.06

### 参数量
1B、7B、40B

### 模型结构
正常的transformer decoder-only的结构，使用了旋转位置编码，使用了Flash Attention + Multi Query Attenion(MQA)。

### 思路

只要对数据进行严格的数据清洗，即便是只使用「互联网语料」也能够让 LLM 学会各种技能。

主要精力在于清理数据上，主要使用CommonCrawl数据集
- URL过滤
    - 制定 URL 黑名单和计算 URL 分数来决定内容是否保留。
- 内容抽取
    - 获取到这些 URL 中的「文本信息」，过滤并丢弃「目录」、「标题」、「广告」等无关的内容
- 语言识别（去除50%的数据）
    - 使用 [fastText] 训练了一个语言识别模型，去掉那些英语得分低于 0.65 的文章
    - 这类低质文章主要包括：非英语文章、以及那些不是由自然语言构成的网页（比如一些纯图像的博客和广告）。
- 低质过滤（去除24%的数据）
    - 篇章级别过滤
        - 去除一些文章内一直重复同一段内容的文章
        - 对每一篇文章，通过判断文章整体长度、标点符号占文章长度的比例等，来过滤掉那些不正规的文章
    - 句子级别过滤
        - 通过设定以下策略，过滤掉文章中的一些无用的句子
- 文章去重
    - 通过 MinHash 的方法对剩余文章进行 MinHash 值计算，每篇文章使用 9000 个 Hash 值
        - （用 20 个桶，每个桶 450 个值）
    - 使用确定性去重的方法，删掉那些重复片段超过 50 个 token 的文章
    - 根据 URL 进一步去除了一部分重复的文章

<div align=center>
<img src=https://github.com/wzzzd/LLM_Learning_Note/blob/main/img/model/falcon1-filter.png width=80% />
</div>

### 参考
    - https://zhuanlan.zhihu.com/p/637996787



## Falcon 2
### 时间
    * 2024.05
### 参数量
    * 11B、11B VLM
### 训练数据
    * 5,000B tokens of RefinedWeb,
### 输入长度
    * 8192
### 模型结构
    结构跟Falcon 1类似，都是基于transformer decoder-only的结构，使用了旋转位置编码，使用了Flash Attention2 + Multi Query Attenion(MQA)。

### 参考
- https://huggingface.co/tiiuae/falcon-11B
