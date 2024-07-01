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
    - vLLM
    - ollama
- 分词器（Tokenizer）
    - [BPE](https://github.com/wzzzd/LLM_Learning_Note/blob/main/tokenizer/tokenizer.md)
    - [BBPE](https://github.com/wzzzd/LLM_Learning_Note/blob/main/tokenizer/tokenizer.md)
    - [WordPiece](https://github.com/wzzzd/LLM_Learning_Note/blob/main/tokenizer/tokenizer.md)
    - SentencePiece
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
        - Llama3
    - GLM series
        - [GLM-130B](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/glm/glm.md)
        - [ChatGLM(V1/V2/V3/V4)](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/glm/chatglm.md)
    - Falcon
        - [Falcon 1](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/falcon.md)
        - [Falcon 2](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/falcon.md)
    - Qwen
        - [Qwen v1](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/qwen.md)
        - [Qwen v1.5](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/qwen.md)
        - [Qwen v2](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/qwen.md)
    - Mistral
        - [Mistral 7B](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/mistral.md)
        - [Mixtral 8x7B MoE](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model/mistral.md)
- Embedding模型
    - BCEmbedding
    - BGE
    - GTE
    - E5
- 多模态模型

- 模型结构
    - [LN种类（Layer Nrom）](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/ln.md)
        - Post-LN
        - [Pre-LN](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/ln.md)
        - [Sandwich-LN](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/ln.md)
        - [DeepNet](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/ln.md)
    - [注意力机制种类（Attention）](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/attention.md)
        - [Self-Attention](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/attention.md)
        - [Muti Query Attention(MQA)](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/attention.md)
        - [Group Query Attention(GQA)](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/attention.md)
        - [Flash Attention v1/2/3](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/attention.md)
    - 位置编码（Position Embedding）
        - [绝对位置编码](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_abs.md)
            - [正余弦位置编码](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_abs.md)
            - [可训练式编码](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_abs.md)
        - [相对位置编码](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rel.md)
            - [Google](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rel.md)
            - [Transformer-XL](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rel.md)
            - [ALiBi](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rel.md)
            - [Kerple](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rel.md)
        - [旋转位置编码](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rope.md)
            - [RoPE](https://github.com/wzzzd/LLM_Learning_Note/blob/main/model-module/pos_emb_rope.md)

- 适应性训练（Adaptation tuning of LLMs）
    - 指令训练（Instuction tuning）
    - 对齐训练（Alignment tuning）
        - RLHF
        - DPO
        - RLAIF
        - RRHF
        - RSO
        - Rejection Sampling + SFT

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
- Scaling Law
- 




