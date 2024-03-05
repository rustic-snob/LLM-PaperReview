# Awesome LLM Papers
중요한 LLM 관련 논문들을 팔로잉하고 있습니다🤗</br>
Following the newest & important LLM papers🤗

# Recommendation Annotation
단계 및 중요도를 고려한 논문 큐레이션을 제공합니다 (`HIGHLY RECOMM.` 열 참조)🤗</br>
Curated papers in terms of difficulty and coreness (See `HIGHLY RECOMM.` Column)🤗
| Lv. 1 | Lv. 2 | Lv. 3 |
|:-----:|:-----:|:-----:|
| 🔵 | 🟢 | 🟡(To be added...) |

# Table
- [Foundation Models & Finetuned Family](#foundation-models--finetuned-family)
- [PEFT](#peft)
- [Efficient Transformer](#efficient-transformer)
- [Positional Embedding & Input Length Control](#positional-embedding--input-length-control)
- [LLM & Reinforcement Learning](#llm--reinforcement-learning)
- [Mixture-of-Experts](#mixture-of-experts)
- [RAG](#rag)
- [Expand and Enhance LLM](#expand-and-enhance-llm)
- [Cool New LLM Architecture & Sub-Module](#cool-new-llm-architecture--sub-module)
- [Mamba & State Spaces Model](#mamba--state-spaces-model)
- [Etc.](#etc)
- [...Maybe Rather Better Diffuse Than Attend](#maybe-rather-better-diffuse-than-attend)

## Foundation Models & Finetuned Family

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | LLaMA | Meta | February. 2023 | \#Model<br>\#Foundation | 🔵
[\*Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) | Alpaca | Stanford University | March. 2023 | \#Model<br>\#Finetuning<br>\#Self-instruct | 🔵
[\*Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/) | Vicuna | LMSYS Org. | March. 2023 | \#Model<br>\#Finetuning<br>\#Methodology | 🟢
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Chinchilla | - | May. 2022 | \#Model<br>\#Foundation<br>\#Methodology | 🟢
[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | LIMA | Meta | May. 2023 | \#Model<br>\#Finetuning<br>\#Data-centric
[Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707) | Orca | Microsoft | June. 2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Platypus: Quick, Cheap, and Powerful Refinement of LLMs](https://platypus-llm.github.io/Platypus.pdf) | Platypus | Boston University | August. 2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Mistral 7B](https://arxiv.org/abs/2310.06825) | Mistral | Mistral.AI | Oct.2023 | \#Model<br>\#Finetuning<br>\#LightWeight | 🔵
[Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045) | Orca 2 | Microsoft | Nov.2023 | \#Model<br>\#Finetuning<br>\#Methodology
[Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) | ZEPHYR | HuggingFace | Oct. 2023 | \#Finetuning</br>\#dDPO</br>\#distilled
[OpenChat: Advancing Open-source Language Models with Mixed-Quality Data](https://arxiv.org/pdf/2309.11235.pdf) | OpenChat | Tsinghua Univ. | Sep. 2023 | \#Finetuning</br>\#C-RLFT</br>\#Mistral
[MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284) | MetaMath | Cambridge Univ. | Oct. 2023 | \#Finetuning</br>\#Math
[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) | TinyLlama | StatNLP Research Group | Jan. 2024 | \#Finetuning</br>\#Efficient
[UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377) | UltraFeedback | Tsinghua Univ. | Oct. 2023 | \#Finetuning</br>\#Dataset</br>\#Preference
[Qwen Technical Report](https://arxiv.org/abs/2309.16609) | Qwen | Alibaba | Sep. 2023 | \#Foundation</br>\#Servey-like


## PEFT

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | Adapter | - | Jun. 2019 | \#PEFT
[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) | Prefix-tuning | Stanford University | Jan. 2021 | \#PEFT
[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | Prompt-tuning | Google Research | Sep. 2021 | \#PEFT
[GPT Understands, Too](https://arxiv.org/abs/2103.10385) | P-tuning | - | March. 2021 | \#PEFT | 🔵
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2205.05638) | LoRA | NC univ. \@chapel hill  | October. 2021 | \#PEFT | 🔵
[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | QLoRA | University of Washington  | May. 2023 | \#PEFT<br>\#LoRA | 🟢
[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638) | IA3 | NC univ. \@chapel hill | August. 2022 | \#PEFT<br>\#FewShot-learning | 🟢
[Stack More Layers Differently: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695) | ReLoRA | Massachusetts Lowel | August. 2023 | \#PEFT<br>\#LoRA
[LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269) | LoRAHub | Sea AI Lab | Jun. 2023 | \#PEFT<br>\#LoRA<br>\#Compose
[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) | VeRA | University of Amsterdam | Oct. 2023 | \#PEFT
[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | DoRA | NVIDIA | March. 2024 | \#PEFT<br>\#LoRA<br>


## Efficient Transformer

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Knowledge Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543.pdf) | MiniLLM | CoAI Group | Jun. 2023 | \#LightWeight<br>\#Distillation
[VLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | VLLM | UC Berkeley | Sep. 2023 | \#LightInference<br>\#Attention<br>\#KVcache
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | GQA | Google Research | Oct. 2023 | \#LightWeight<br>\#Attention<br>\#Distillation | 🟢
[\*Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | FlashDecoding | Stanford University | Oct. 2023 | \#LightInference<br>\#Attention<br>\#Parallelization
[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | StreamingLLM | Massachusetts University | Sep. 2023 | \#LightInference<br>\#Attention<br>\#KVcache
[Cascade Speculative Drafting for Even Faster LLM Inference](https://arxiv.org/abs/2312.11462) | CS Drafting | Illinois Univ. | Dec. 2023 | \#LightInference
[Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118) | AQLM | HSE Univ. | Jan. 2024 | \#LightWeight<br>\#Quantize
[Initializing Models with Larger Ones](https://arxiv.org/abs/2311.18823) | - | Pennsylvania Univ | Nov. 2023 | \#LightWeight<br>\#Distillation-like


## Positional Embedding & Input Length Control

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE | Zhuiyi Technology | August. 2022 | \#PE<br>\#RPE<br>\#ComplexPlane | 🟢
[TRAIN SHORT, TEST LONG:<br>ATTENTION WITH LINEAR BIASES<br>ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/abs/2108.12409)| ALiBi | Facebook | April. 2022 | \#seq_len<br>\#Extrapolation<br>\#Efficient
[A Length-Extrapolatable Transformer](https://arxiv.org/abs/2212.10554) | xPos | Microsoft | December. 2022 | \#PE<br>\#RoPE<br>\#ComplexPlane
[\*Extending Context is Hard…but not Impossible](https://kaiokendev.github.io/context) | kaiokendev | - | February. 2023 | \#Interpolation<br>\#RoPE | 🟢
[EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION](https://arxiv.org/abs/2306.15595) | post-kaiokendev | Meta | Jun. 2023 | \#seq_len<br>\#Interpolation<br>\#RoPE
[LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307) | LongLoRA | CUHK | Dec. 2023 | \#seq_len<br>\#LoRA<br>\#ContinualTraining


## LLM & Reinforcement Learning

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | InstructGPT | OpenAI | March. 2022 | \#Finetuning</br>\#PPO</br>\#Instruction | 🔵
[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Voyager | NVIDIA | Oct. 2023 | \#Prompting</br>\#Game
[Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/abs/2310.00166) | Motif | Mila | Sep. 2023 | \#LLMfeedback</br>\#Game
[RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302) | RRHF | Alibaba DAMO Academy | Oct. 2023 | \#Finetuning</br>\#RLHF
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | DPO | Stanford University | May. 2023 | \#Finetuning</br>\#DPO
[A General Theoretical Paradigm to Understand Learning from Human Preferences](arxiv.org/abs/2310.12036) | ΨPO | Google DeepMind | Nov. 2023 | \#Finetuning</br>\#ΨPO
[Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417) | CPO | Johns Hopkins Univ. | Jan. 2024 | \#Finetuning</br>\#CPO
[Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond](https://arxiv.org/abs/2310.06147) | - | Cambridge Univ. | Oct. 2023 | \#Survey</br>\#RLHF
[Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964) | - | Fudan NLP Group | Jul. 2023 | \#Survey</br>\#PPO
[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | - | Meta | Jan. 2024 | \#Finetuning</br>\#Preference</br>\#SelfJudge
[A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860) | - | McGill Univ. | Dec. 2021 | \#OfflineRL


## Mixture-of-Experts

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[\*Mixture of Experts: How an Ensemble of AI Models Decide As One](https://deepgram.com/learn/mixture-of-experts-ml-model-guide) | - | Deepgram | Oct. 2023 | \#MoE</br>\#History | 🟢
[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | - | Google Brain | Jan. 2017 | \#MoE</br>\#Old-work
[On the Representation Collapse of Sparse Mixture of Experts](https://arxiv.org/abs/2204.09179) | - | Beijing Institute of Technology | Oct. 2022 | \#MoE</br>\#Methodology
[MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) | - | - | Nov. 2022 | \#MoE</br>\#Methodology
[Mixtral of experts](https://arxiv.org/abs/2401.04088) | Mixtral | Mistral AI | Jan. 2024 | \#MoE</br>\#Model | 🟢
[QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795) | QMoE | - | Oct. 2023 | \#MoE</br>\#Methodology</br>\#Quantization


## RAG

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | RAG | Facebook AI Research | April.2021 | \#RAG | 🔵
[REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652) | REPLUG | Washington university | May. 2023 | \#Retrieval<br>\#RAG
[Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) | - | Microsoft | Dec. 2023 | \#RAG<br>\#Embedding
[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | Self-RAG | Washington Univ. | Oct. 2023  | \#Retrieval<br>\#RAG
[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | - | Tongji Univ. | Jan. 2024 | \#RAG<br>\#Survey




## Expand and Enhance LLM

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[\*Swallow: LLaMA-2](https://zenn.dev/tokyotech_lm/articles/d6cb3a8fdfc907) | - | 藤井 | Dec. 2023 | \#VocabExpand</br>\#Article
[\*Tokenizer Expansion](https://seen-point-bd9.notion.site/Tokenizer-Expansion-ecb6d78211a54ba6b3cf8ebc0ec1d105#1f36bc3b970c465db5614a97cae2c55b) | - | Cheon Jaewon | Jan. 2024 | \#VocabExpand</br>\#How-to
[LLaMA Beyond English: An Empirical Study on Language Capability Transfer](https://arxiv.org/abs/2401.01055) | - | Fudan University | Jan. 2024 | \#VocabExpand</br>\#Cons
[LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415) | LLAMA PRO | Hong Kong Univ. | Jan. 2024 | \#LayerExpand
[SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling](https://arxiv.org/abs/2312.15166) | Solar | Upstage | Dec. 2023 | \#LayerExpand | 🟢
[The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://arxiv.org/abs/2312.13558) | LASER | MIT | Dec. 2023 | \#LayerDecompose


## Cool New LLM Architecture & Sub-Module

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[RWKV: <br>Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) | RWKV | RWKV Foundation | May. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) | RetNet | Microsoft | July. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) | Hyena | - | April. 2023 | \#Architecture<br>\#Recurrent<br>\#Efficient
[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) | 1-Bit Transformer | Microsoft | Oct. 2023 | \#Architecture<br>\#Quantized
[NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) | NEFT | Maryland University | Oct. 2023 | \#SubModule<br>\#NoisedVector<br>\#Finetuning
[Exponentially Faster Language Modelling](https://arxiv.org/abs/2311.10770) | FFFs | ETH Zurich | Nov. 2023 | \#FFNN<br>\#Speeding<br>
[Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906) | - | ETH Zurich | Nov. 2023 | \#SubModule<br>\#Speeding<br>\#LightWeight<br>
[Cached Transformers: Improving Transformers with Differentiable Memory Cache](https://arxiv.org/abs/2312.12742) | Cached Transformers | CUHK | Dec. 2023 | \#Recurrent<br>\#Vision<br>\#Efficiency


## Mamba & State Spaces Model

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) | HiPPO | Stanford University | Oct. 2020 | \#Architecture
[Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](https://arxiv.org/abs/2110.13985) | LSSL | Stanford University | Oct. 2021 | \#Architecture
[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) | S4 | Stanford University | Aug. 2022 | \#Architecture<br>\#SSM
[Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052) | H3 | Stanford University | April. 2023 | \#Architecture<br>\#SSM
[Effectively Modeling Time Series with Simple Discrete State Spaces](https://arxiv.org/abs/2303.09489) | - | Stanford University | March. 2023 | \#Architecture<br>\#SSM<br>\#TimeSeries
[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | Mamba | Carnegie Mellon Univ. | Dec. 2023 | \#Architecture<br>\#SSM<br>\#Mamba
[MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081) | MoE-Mamba | IDEAS NCBR | Jan. 2024 | \#Architecture<br>\#Mamba<br>\#MoE
[MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660) | MambaByte | Cornell Univ. | Jan. 2024 | \#Architecture<br>\#ByteToken<br>\#Mamba


## Etc.

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491) | Prometheus | KAIST university | Oct. 2023 | \#Evaluation
[Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/abs/2310.02238) | - | Microsoft | Oct. 2023 | \#EraseMemory<br>\#Forgetting<br>\#Finetuning
[In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916) | Task-Vector | Tel Aviv University | Oct. 2023 | \#In-Context Learning
[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) | TIES-Merging | NC univ. \@chapel hill | Oct. 2023 | \#ModelMerge | 🟢
[UNcommonsense Reasoning: Abductive Reasoning about Uncommon Situations](https://arxiv.org/abs/2311.08469) | UNcommonsense | Cornell Univ. | Nov. 2023 | \#Reasoning<br>\#UnusualScenario
[Language Model Inversion](https://arxiv.org/abs/2311.13647) | - | Cornell Univ. | Nov. 2023 | \#ReverseEngineering<br>\#PromptHacking
[\*Deep Random micro-Glitch Sampling](https://www.reddit.com/r/LocalLLaMA/comments/18toidc/stop_messing_with_sampling_parameters_and_just/) | DRµGS | qrios | Dec. 2023 | \#DecodingStrate<br>\#Sampling
[TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) | TinyStories | Microsoft | May. 2023 | \#Dataset<br>\#Synthetic
[Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models](https://arxiv.org/abs/2401.06102) | Patchscopes | Google Research | Jan. 2024 | \#Representation
[\*Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) | - | TurnTrout | May. 2023 | \#Representation<br>\#Steering
[Learning Universal Predictors](https://arxiv.org/abs/2401.14953) | - | Google Deepmind | Jan. 2024 | \#Predictor

## ...Maybe Rather Better Diffuse Than Attend

| Paper | a.k.a | Affiliation | Published date | # | HIGHLY<br>RECOMM.|
|-------|-------|-------------|----------------|---|:------:|
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | DDPM | UC Berkeley | Dec. 2020 | \#Diffusion
[Non-Autoregressive Text Generation with Pre-trained Language Models](https://arxiv.org/abs/2102.08220v1) | - | Cambridge Univ. | Feb. 2021 | \#NAG
[NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer](https://arxiv.org/abs/2106.02210v1) | NAST | CoAI group | Jun. 2021 | \#NAG
[ELMER: A Non-Autoregressive Pre-trained Language Model for Efficient and Effective Text Generation](https://arxiv.org/abs/2210.13304v2) | ELMER | Renmin Univ. | Oct. 2022 | \#NAG
[Directed Acyclic Transformer Pre-training for High-quality Non-autoregressive Text Generation](https://arxiv.org/abs/2304.11791v1) | PreDAT | CoAI group | April. 2023 | \#NAG
[Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576) | - | - | March. 2023 | \#NLP<br>\#Diffusion
[Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574v2) | - | Renmin Univ. | May. 2023 | \#NAG<br>\#Diffusion
[Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) | DiffusionDPO | Salesforce AI | Nov. 2023 | \#Diffusion<br>\#DPO
[A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671) | - | Minnesota Univ. | Jun. 2023 | \#NLP<br>\#NAG<br>\#Diffusion


\*: not a paper
