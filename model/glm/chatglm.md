# ChatGLM-6B v1/v2/v3/v4


## V1
* 地址：https://github.com/THUDM/ChatGLM-6B
* 1T的token进行中英文训练
* 使用SFT+RLHF微调
* 支持2K的长度文本

## V2
* 地址：https://github.com/THUDM/ChatGLM2-6B
* 1.4T的中英文语料
* 使用SFT+RLHF微调
* 最长支持32K
* 使用Flash Attention，加快训练速度、增长上下文长度、节省GPU显存
* 使用Multi-Query Attention，降低了推理速度和显存占用

## V3
* 地址：https://github.com/THUDM/ChatGLM3
* 增加训练数据、训练步数，修改训练策略

## V4
* 收费，不开源
* 最长支持128K
* 基本工具调用能力：浏览器、AI绘图
