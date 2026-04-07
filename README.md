# Representation Engineering (RepE): LLM Control via Hidden State Manipulation

This repository contains a solo-developed implementation and extension of **Representation Engineering (RepE)**, as proposed by [Zou et al. (2023)](https://arxiv.org/abs/2310.01405). This project moves beyond "black-box" prompt engineering by identifying and manipulating the internal cognitive states of Large Language Models (LLMs) through their high-dimensional hidden layer activations.

## 🚀 Project Overview
The objective was to extract "concept vectors" (directions in activation space) that represent abstract concepts like emotions and political leanings. By isolating these vectors, I implemented a mechanism to "read" a model's internal state and "steer" its output mechanistically during inference.

### Key Engineering & Research Highlights:
*   **Representational Reading:** Built a pipeline using Linear Artificial Tomography (LAT) and Principal Component Analysis (PCA) to isolate concept directions within the residual streams of **Mistral-7B**.
*   **Vector Compositionality:** Successfully proved that internal representations are arithmetically compositional. By averaging individual vectors (e.g., `Happiness` + `Sadness`), I forced the model to generate nuanced "bittersweet" narratives that are difficult to elicit consistently via standard prompting.
*   **Memory & Resource Optimization:** Solved hardware constraints (OOM errors) associated with processing massive activation datasets. Implemented **4-bit quantization (bitsandbytes)** and a **data chunking/batching strategy** to perform PCA on high-dimensional tensors on consumer-grade hardware.
*   **High-Level Concept Mapping:** Extended the original research from simple emotions to complex societal concepts by curating a custom dataset from Wikipedia to map a linear axis for political ideology.

## 🛠 Technical Workflow

### 1. Activation Collection
Contrastive prompts (e.g., "Pretend you are honest" vs. "Pretend you are dishonest") are fed into the model. I capture the hidden states $H$ at the last token position for specific layers.

### 2. PCA & Reading Vectors
I calculate the difference vectors between contrastive pairs:
$$D = H_{positive} - H_{negative}$$
Applying PCA to these differences allows me to extract the first principal component, which serves as the **Reading Vector**. This vector represents the specific "axis" of that concept in the model's 4096-dimensional hidden space.

### 3. Representational Control (Steering)
To change model behavior, I inject the reading vector back into the hidden states during the forward pass:
$$H'_{layer} = H_{layer} + \alpha \cdot v_{reading}$$
where $\alpha$ is a scaling coefficient. This allow for deterministic control over the "intensity" of a concept without changing the input prompt.

## 📊 Key Results
*   **Truthfulness Benchmarking:** Demonstrated that suppressing "Anger" and "Fear" vectors led to improved performance on the **TruthfulQA** benchmark, suggesting a mechanistic link between simulated emotional states and factual reasoning.
*   **Compositional Generation:** Proved that $v_{bittersweet} \approx \frac{v_{happy} + v_{sad}}{2}$ results in coherent, context-aware bittersweet narratives.
*   **Ideological Steering:** Successfully scaled model responses along a linear political axis, moving from moderate to radical viewpoints in a controlled fashion.

## 💻 Tech Stack
*   **Language:** Python
*   **Libraries:** PyTorch, Hugging Face Transformers, Scikit-learn, Accelerate
*   **Models:** Mistral-7B-v0.1 (Quantized)
*   **Hardware Platform:** Google Colab TPUs / NVIDIA T4 GPUs

## 📚 References
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). *arXiv preprint arXiv:2310.01405*.
