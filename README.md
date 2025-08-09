# HippoMemory: A Brain-Inspired Memory-Augmented LLM Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🧠 **HippoMemory** is an **experimental, brain-inspired memory augmentation framework** designed to endow Large Language Models (LLMs) with human-like short-term and long-term memory capabilities. By simulating the consolidation, forgetting, and retrieval mechanisms of the human hippocampus, HippoMemory aims to overcome the inherent context length limitations of LLMs.

> **⚠️ Important Note: This is a Research Prototype**
>
> This repository is **not a production-ready system**. It is a **work in progress** primarily intended to:
>
> 1.  **Inspire Discussion:** Spark ideas and discussion around biologically-inspired AI architectures.
> 2.  **Demonstrate Concepts:** Serve as a proof-of-concept for the *ideas* of entropy-driven compression, temporal gating, and differentiable graph memory.
> 3.  **Provide a Foundation:** Offer a modular codebase for others to experiment with and build upon.
>
> The current implementation has **not been rigorously trained or tested** on large-scale datasets. The focus is on the *design philosophy* and *architectural components*, rather than the immediate performance of this specific version. I encourage you to explore the underlying concepts and consider how they might be refined and applied.
>
> **I am actively working on this project and will continue to refine, train, and evaluate it over time.**

---

## 🌟 Core Features

- ✅ **Entropy-Driven Memory Compression**: Dynamically filters and compresses key information using multi-dimensional entropy analysis and structural knowledge.
- ✅ **Temporal Hippocampal Gating**: Simulates the brain's short-term memory system with time-aware decay, multi-hop reasoning, and adaptive forgetting.
- ✅ **Trainable Graph Memory**: A fully differentiable, end-to-end trainable graph-based long-term memory system with self-supervised relation discovery.
- ✅ **Dynamic Memory Fusion**: Intelligently computes the fusion weights of current input, short-term, and long-term memory for context enhancement.
- ✅ **Modular & Extensible**: Clean separation of components for easy customization and future integration with external storage.

---

## 🧠 How It Works

The HippoMemory system processes user input through a series of biologically inspired stages:

1.  **Input Encoding & Compression**: The `MemoryCompressor` analyzes the text, computes token-level information entropy, extracts knowledge graph (KG) triples, and outputs compressed, high-value memory fragments.
2.  **Short-Term Memory Update**: The `HippocampalGate` module acts as the "hippocampus," updating the short-term memory pool by fusing new fragments with existing ones, applying time decay, and performing adaptive forgetting.
3.  **Long-Term Memory Consolidation**: The `GraphMemory` module acts as the "neocortex," converting memory fragments into nodes and relations, and storing them in a dynamic knowledge graph for persistent, structured storage.
4.  **Memory Fusion & Response**: For each query, the system retrieves relevant information from both short-term and long-term memory, fuses them with the current input via a `DynamicRouter`, and passes the enhanced context to the LLM.

The entire process is visualized below:

```mermaid
graph TD
    A[User Input] --> B(MemoryCompressor)
    B --> B1[Token Embedding]
    B --> B2[Entropy Calculation]
    B --> B3[KG Triple Extraction]
    B --> C[HippocampalGate]
    C --> C1[Time Decay]
    C --> C2[Memory Update]
    C --> D[Short-Term Memory]
    D --> C
    B --> E[GraphMemory]
    E --> E1[Entity Mapping]
    E --> E2[Relation Discovery]
    E --> E3[Dynamic Graph Storage]
    E --> F[Long-Term Knowledge Graph]
    F --> E
    D --> G[HippoMemorySystem]
    F --> G
    C --> G
    G --> H(Dynamic Router)
    H --> I[Augmented Context]
    I --> J[Large Language Model]
    style A fill:#f9f,stroke:#333
    style J fill:#bbf,stroke:#333,color:#fff
    style E fill:#cf9,stroke:#333
    style C fill:#ffcc00,stroke:#333
