# Awesome LLM Papers
중요한 LLM 관련 논문들을 팔로잉하고 있습니다🤗</br>
Following the newest & important LLM papers🤗

## Foundation Models & Finetuned Family

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | LLaMA | Meta | February. 2023 | \#Model<br>\#Foundation<br>
[\*Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) | Alpaca | Stanford University | March. 2023 | \#Model<br>\#Finetuning<br>\#Self-instruct
[\*Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/) | Vicuna | LMSYS Org. | March. 2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Chinchilla | - | May. 2022 | \#Model<br>\#Foundation<br>\#Methodology
[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | LIMA | Meta | May. 2023 | \#Model<br>\#Finetuning<br>\#Data-centric
[Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707) | Orca | Microsoft | June. 2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Platypus: Quick, Cheap, and Powerful Refinement of LLMs](https://platypus-llm.github.io/Platypus.pdf) | Platypus | Boston University | August. 2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Mistral 7B](https://arxiv.org/abs/2310.06825) | Mistral | Mistral.AI | Oct.2023 | \#Model<br>\#Finetuning<br>\#LightWeight
[Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045) | Orca 2 | Microsoft | Nov.2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) | ZEPHYR | HuggingFace | Oct. 2023 | \#Finetuning</br>\#dDPO</br>\#distilled
[OpenChat: Advancing Open-source Language Models with Mixed-Quality Data](https://arxiv.org/pdf/2309.11235.pdf) | OpenChat | Tsinghua Univ. | Sep. 2023 | \#Finetuning</br>\#C-RLFT</br>\#Mistral
[MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284) | MetaMath | Cambridge Univ. | Oct. 2023 | \#Finetuning</br>\#Math
[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) | TinyLlama | StatNLP Research Group | Jan. 2024 | \#Finetuning</br>\#Efficient


## PEFT

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | Adapter | - | Jun. 2019 | \#PEFT
[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) | Prefix-tuning | Stanford University | Jan. 2021 | \#PEFT
[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | Prompt-tuning | Google Research | Sep. 2021 | \#PEFT
[GPT Understands, Too](https://arxiv.org/abs/2103.10385) | P-tuning | - | March. 2021 | \#PEFT
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2205.05638) | LoRA | NC univ. \@chapel hill  | October. 2021 | \#PEFT
[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | QLoRA | University of Washington  | May. 2023 | \#PEFT<br>\#LoRA
[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638) | IA3 | NC univ. \@chapel hill | August. 2022 | \#PEFT<br>\#FewShot-learning
[Stack More Layers Differently: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695) | ReLoRA | Massachusetts Lowel | August. 2023 | \#PEFT<br>\#LoRA
[LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269) | LoRAHub | Sea AI Lab | Jun. 2023 | \#PEFT<br>\#LoRA<br>\#Compose
[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) | VeRA | University of Amsterdam | Oct. 2023 | \#PEFT


## Efficient Transformer

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Knowledge Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543.pdf) | MiniLLM | CoAI Group | Jun. 2023 | \#LightWeight<br>\#Distillation
[VLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | VLLM | UC Berkeley | Sep. 2023 | \#LightInference<br>\#Attention<br>\#KVcache
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | GQA | Google Research | Oct. 2023 | \#LightWeight<br>\#Attention<br>\#Distillation
[\*Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | FlashDecoding | Stanford University | Oct. 2023 | \#LightInference<br>\#Attention<br>\#Parallelization
[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | StreamingLLM | Massachusetts University | Sep. 2023 | \#LightInference<br>\#Attention<br>\#KVcache


## Positional Embedding & Input Length Control

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE | Zhuiyi Technology | August. 2022 | \#PE<br>\#RPE<br>\#ComplexPlane
[TRAIN SHORT, TEST LONG:<br>ATTENTION WITH LINEAR BIASES<br>ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/abs/2108.12409)| ALiBi | Facebook | April. 2022 | \#seq_len<br>\#Extrapolation<br>\#Efficient
[A Length-Extrapolatable Transformer](https://arxiv.org/abs/2212.10554) | xPos | Microsoft | December. 2022 | \#PE<br>\#RoPE<br>\#ComplexPlane
[\*Extending Context is Hard…but not Impossible](https://kaiokendev.github.io/context) | kaiokendev | - | February. 2023 |
[EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION](https://arxiv.org/abs/2306.15595) | post-kaiokendev | Meta | Jun. 2023 | \#seq_len<br>\#Interpolation<br>\#RoPE
[LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307) | LongLoRA | CUHK | Dec. 2023 | \#seq_len<br>\#LoRA<br>\#ContinualTraining


## LLM & Reinforcement Learning

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | InstructGPT | OpenAI | March. 2022 | \#Finetuning</br>\#PPO</br>\#Instruction
[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Voyager | NVIDIA | Oct. 2023 | \#Prompting</br>\#Game
[Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166) | Motif | Mila | Sep. 2023 | \#LLMfeedback</br>\#Game
[RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302) | RRHF | Alibaba DAMO Academy | Oct. 2023 | \#Finetuning</br>\#RLHF
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | DPO | Stanford University | May. 2023 | \#Finetuning</br>\#DPO
[A General Theoretical Paradigm to Understand Learning from Human Preferences](arxiv.org/abs/2310.12036) | ΨPO | Google DeepMind | Nov. 2023 | \#Finetuning</br>\#ΨPO
[Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond](https://arxiv.org/abs/2310.06147) | - | Cambridge Univ. | Oct. 2023 | \#Survey</br>\#RLHF
[Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964) | - | Fudan NLP Group | Jul. 2023 | \#Survey</br>\#PPO


## Mixture-of-Experts

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[\*Mixture of Experts: How an Ensemble of AI Models Decide As One](https://deepgram.com/learn/mixture-of-experts-ml-model-guide) | - | Deepgram | Oct. 2023 | \#MoE</br>\#History
[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | - | Google Brain | Jan. 2017 | \#MoE</br>\#Old-work
[On the Representation Collapse of Sparse Mixture of Experts](https://arxiv.org/abs/2204.09179) | - | Beijing Institute of Technology | Oct. 2022 | \#MoE</br>\#Methodology
[MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) | - | - | Nov. 2022 | \#MoE</br>\#Methodology
[Mixtral of experts](https://arxiv.org/abs/2401.04088) | Mixtral | Mistral AI | Jan. 2024 | \#MoE</br>\#Model
[QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) | QMoE | - | Oct. 2023 | \#MoE</br>\#Methodology</br>\#Quantization


## Expand and Enhance LLM

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[\*Swallow: LLaMA-2](https://zenn.dev/tokyotech_lm/articles/d6cb3a8fdfc907) | - | 藤井 | Dec. 2023 | \#VocabExpand</br>\#Article
[\*Tokenizer Expansion](https://seen-point-bd9.notion.site/Tokenizer-Expansion-ecb6d78211a54ba6b3cf8ebc0ec1d105#1f36bc3b970c465db5614a97cae2c55b) | - | Cheon Jaewon | Jan. 2024 | \#VocabExpand</br>\#How-to
[LLaMA Beyond English: An Empirical Study on Language Capability Transfer](https://arxiv.org/abs/2401.01055) | - | Fudan University | Jan. 2024 | \#VocabExpand</br>\#Cons
[LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415) | LLAMA PRO | Hong Kong Univ. | Jan. 2024 | \#LayerExpand
[SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling](https://arxiv.org/abs/2312.15166) | Solar | Upstage | Dec. 2023 | \#LayerExpand


## Cool New LLM Architecture & Sub-Module

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[RWKV: <br>Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) | RWKV | RWKV Foundation | May. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | RetNet | Microsoft | July. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) | Hyena | - | April. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) | 1-Bit Transformer | Microsoft | Oct. 2023 | \#Architecture<br>\#Quantized
[NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) | NEFT | Maryland University | Oct. 2023 | \#SubModule<br>\#NoisedVector<br>\#Finetuning
[Exponentially Faster Language Modelling](https://arxiv.org/abs/2311.10770) | FFFs | ETH Zurich | Nov. 2023 | \#FFNN<br>\#Speeding<br>
[Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906) | - | ETH Zurich | Nov. 2023 | \#SubModule<br>\#Speeding<br>\#LightWeight<br>
[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | Mamba | Carnegie Mellon Univ. | Dec. 2023 | \#Architecture<br>\#SSM


## Applications

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491) | Prometheus | KAIST university | Oct. 2023 | \#Evaluation
[REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652) | REPLUG | Washington university | May. 2023 | \#Retrieval<br>\#RAG
[Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) | - | Microsoft | Dec. 2023 | \#RAG<br>\#Embedding


## Etc.

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/abs/2310.02238) | - | Microsoft | Oct. 2023 | \#EraseMemory<br>\#Forgetting<br>\#Finetuning
[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916) | Task-Vector | Tel Aviv University | Oct. 2023 | \#In-Context Learning
[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) | TIES-Merging | NC univ. \@chapel hill | Oct. 2023 | \#ModelMerge
[UNcommonsense Reasoning: Abductive Reasoning about Uncommon Situations](https://arxiv.org/abs/2311.08469) | UNcommonsense | Cornell Univ. | Nov. 2023 | \#Reasoning<br>\#UnusualScenario
[Language Model Inversion](https://arxiv.org/abs/2311.13647) | - | Cornell Univ. | Nov. 2023 | \#ReverseEngineering<br>\#PromptHacking


## ...Maybe Rather Better Diffuse Than Attend

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Non-Autoregressive Text Generation with Pre-trained Language Models](https://arxiv.org/abs/2102.08220v1) | - | Cambridge Univ. | Feb. 2021 | \#NAG
[NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer](https://arxiv.org/abs/2106.02210v1) | NAST | CoAI group | Jun. 2021 | \#NAG
[ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation](https://arxiv.org/abs/2210.13304v2) | ELMER | Renmin Univ. | Oct. 2022 | \#NAG
[Directed Acyclic Transformer Pre-training for High-quality Non-autoregressive Text Generation](https://arxiv.org/abs/2304.11791v1) | PreDAT | CoAI group | April. 2023 | \#NAG
[Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576) | - | - | March. 2023 | \#NLP<br>\#Diffusion
[Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574v2) | - | Renmin Univ. | May. 2023 | \#NAG<br>\#Diffusion
[A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671) | - | Minnesota Univ. | Jun. 2023 | \#NLP<br>\#NAG<br>\#Diffusion


\*: not a paper
