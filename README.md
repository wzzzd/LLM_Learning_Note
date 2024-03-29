# LLM_Learning_Note



# 介绍
用于记录Large Language Model相关的学习资料、内容，以及理解。

# 注意
如果使用Chrome浏览器，建议安装以下插件，并刷新页面
- [TeX All the Things](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn/related)
- [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)


# 模块
- 开源框架
    - Transformers
    - [DeepSpeed](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Parallel/deepspeed.md)
    - Megatron-LM
    - Jax
    - Colossal-AI
    - BMTrain
    - FastMoE
- LLM模型
    - GPT series
        - [GPT-1](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/gpt-1.md)
        - [GPT-2](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/gpt-2.md)
        - [GPT-3](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/gpt-3.md)
        - [Codex](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/codex.md)
        - [InstructGPT/ChatGPT](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/gpt-series/instructgpt-chatgpt.md)
        - GPT-4
    - Llama
        - [Llama1](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/llama.md)
        - [Llama2](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/llama.md)
    - GLM series
        - GLM-130B
        - ChatGLM-6B(V1/V2/V3)

- 模型结构
    - LN种类（Layer Nrom）
        - Post-LN
        - Pre-LN
        - Sandwich-LN
        - DeepNet
    - 注意力机制种类（Attention）
        - Self-Attention
        - Muti Query Attention(MQA)
        - Group Query Attention(GQA)
        - Flash Attention
    - 位置编码（Position Embedding）
        - 绝对位置编码
            - 三角函数式
            - 可训练式
        - 相对位置编码
            - Google
            - Transformer-XL
            - ALiBi
        - 旋转位置编码
            - RoPE

- 适应性训练（Adaptation tuning of LLMs）
    - 指令训练（Instuction tuning）
    - 对齐训练（Alignment tuning）
        -RLHF
            - Training language models to follow instructions with human feedback
            - Deep reinforcement learning from human preferences
    - [微调方法（Efficient tuning）](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [Adapter tuning](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [Prefix tuning](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [Prompt tuning](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [P-tuning](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [P-tuning v2](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
        - [LoRA](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Tuning/efficient-tuning.md)
- LLM的使用
    - [上下文学习（In-Context learning）](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Utilization/In-context-learning.md)
    - [思维链推理（Chain-of-Thought Prompting）](https://github.com/wzzzd/LLM_Learning_Note/blob/main/Utilization/chain-of-thought-prompting.md)
- LLM的能力评估（Capacity Evaluation）
- 量化
    - 模型、训练过程的显存占用
    - 各种精度介绍
    - 量化技术
    - 量化框架
- 并行
    - 数据并行
        - DP
        - DDP
    - 模型并行
        - 张量并行
        - 流水线并行
            - 朴素实现
            - G-pipe
            - 1F1B
            - Interleaved 1F1B
    - ZeRO（整合数据和模型并行）
    - 并行框架




