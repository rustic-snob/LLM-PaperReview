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


## LLM & Reinforcement Learning

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | InstructGPT | OpenAI | March. 2022 | \#Finetuning</br>\#PPO</br>\#Instruction
[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Voyager | NVIDIA | Oct. 2023 | \#Prompting</br>\#Gam
[Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166) | Motif | Mila | Sep. 2023 | \#LLMfeedback</br>\#Game
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | DPO | Stanford University | May. 2023 | \#Finetuning</br>\#DPO


## Cool New LLM Architecture & Sub-Module

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[RWKV: <br>Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) | RWKV | RWKV Foundation | May. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | RetNet | Microsoft | July. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) | Hyena | - | April. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) | 1-Bit Transformer | Microsoft | Oct. 2023 | \#Architecture<br>\#Quantized
[NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) | NEFT | Maryland University | Oct. 2023 | \#SubModule<br>\#NoisedVector<br>\#Finetuning

## Etc.

| Paper | a.k.a | Affiliation | Published date | # | Desc.|
|-------|-------|-------------|----------------|---|------|
[Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/abs/2310.02238) | - | Microsoft | Oct. 2023 | \#EraseMemory<br>\#Forgetting<br>\#Finetuning
[Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491) | Prometheus | KAIST university | Oct. 2023 | \#Evaluation
[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916) | Task-Vector | Tel Aviv University | Oct. 2023 | \#In-Context Learning

\*: not a paper
