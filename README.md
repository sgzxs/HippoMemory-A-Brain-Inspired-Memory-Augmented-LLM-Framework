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

This might not be very clear, as I'm not yet accustomed to using Markdown for drawing graphs. If you have any questions, please feel free to raise them.

```mermaid
graph TD
    subgraph "HippoMemory System"
        direction TB
        
        subgraph "1. Input Processing & Memory Compression"
            direction LR
            A["User Input Text"] --> B1["MemoryCompressor\n(EnhancedMemoryCompressor)"]
            B1 --> B2["Tokenizer & Encoder\n(MiniLM)"]
            B2 --> B3["WindowEntropyCalculator\n(Information Entropy)"]
            B2 --> B4["NLP (spaCy)\n(Entity/Relation Extraction)"]
            B3 --> B5["Fusion Network\n(Importance Fusion)"]
            B4 --> B5
            B5 --> B6["Dynamic Compression\n(Adaptive Threshold)"]
            B6 --> C1["Compressed Memory Fragments\n(Embeddings)"]
        end

        subgraph "2. Short-Term Memory Management (Hippocampus-like)"
            direction TB
            C1 --> D1["HippocampalGate\n(TemporalHippocampalGating)"]
            D2["Short-Term Memory Buffer\n(short_term_memory)"] --> D1
            D3["Time Delta\n(time_delta)"] --> D1
            D1 --> D4["Gating Logic\n(Forget/Retain/Compress)"]
            D4 --> D5["Updated Short-Term Memory"]
            D5 --> D2
            D5 --> E1
        end

        subgraph "3. Long-Term Memory Storage & Retrieval (Graph Network)"
            direction TB
            C1 --> E1["TrainableGraphMemory\n(or GraphMemory)"]
            E1 --> E2["Entity Mapping\n(Entity Recognition/Mapping)"]
            E2 --> E3["DynamicGraphStorage\n(Graph Storage & Updates)"]
            E3 --> E4["GNN\n(Graph Neural Network Inference)"]
            E4 --> E5["Long-Term Memory Representation"]
            E5 --> F2
        end

        subgraph "4. Dynamic Routing & Context Augmentation"
            direction LR
            F1["Current LLM Input Embedding"] --> F2[DynamicRouter]
            D5 --> F2
            E5 --> F2
            F2 --> F3["Compute Fusion Weights\n(Current, STM, LTM)"]
            F3 --> F4["Augmented Context Vector"]
        end
    end

    subgraph "External Interaction"
        direction LR
        F4 --> G["Large Language Model (LLM)"]
        G --> H["Final Response"]
        H --> A
        I[Timestamp] --> D3
    end

    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style D2 fill:#ffeb3b,stroke:#333
    style E3 fill:#c8e6c9,stroke:#333
    style E4 fill:#c8e6c9,stroke:#333
    style F4 fill:#ffcc80,stroke:#333
    
    classDef module fill:#bbdefb,stroke:#333,stroke-width:2px;
    classDef data fill:#ffcdd2,stroke:#333;
    classDef process fill:#e1bee7,stroke:#333;
    
    class B1,D1,E1,F2 module;
    class C1,D5,E5,F4,G data;
    class B3,B5,B6,D4,E2,E4,F3 process;
