# Variable Latent Perceiver AR: Curriculum Learning for Efficient Sequence Modeling

## Project Overview

This project implements and evaluates a novel approach to sequence modeling using Perceiver AR (Autoregressive) architectures with variable latent dimensions and curriculum learning strategies. The research explores how dynamically adjusting the cross-attention sequence length during training can improve both computational efficiency and model performance.

## Research Contributions

- **Variable Latent Perceiver AR**: Implementation of a Perceiver AR model that can dynamically adjust its cross-attention sequence length
- **Curriculum Learning Strategies**: Novel training approaches including uniform sampling and progressive curriculum learning

## Architecture

### Perceiver AR Model
- **Cross-Attention Mechanism**: Efficiently processes variable-length sequences through learned latent representations
- **Transformer Backbone**: 6-layer transformer with 8 attention heads and 512 embedding dimensions
- **Variable Latent Dimensions**: Supports cross-attention lengths from 32 to 1024 tokens

### Training Strategies
1. **Fixed Prefix**: Constant cross-attention length (256, 512 tokens)
2. **Variable Uniform**: Random sampling between 64-512 tokens
3. **Curriculum Learning**: Progressive increase from 64 to 512 tokens over training

## Dataset

- **enwik8**: Character-level language modeling on Wikipedia text
- **Training**: 90M characters
- **Validation**: 5M characters  
- **Test**: 5M characters

## Results

The model achieves competitive performance on character-level language modeling:

- **Best Perplexity**: Achieved through curriculum learning approach
- **Memory Efficiency**: Significant reduction in peak memory usage compared to fixed-length approaches
- **Training Stability**: Curriculum learning shows more stable convergence patterns

## Technical Specifications

- **Framework**: PyTorch
- **Model Size**: ~15M parameters
- **Training**: Adam optimizer with gradient clipping
- **Hardware**: CUDA-compatible GPUs
- **Monitoring**: Weights & Biases integration

## Usage

### Training
```bash
python Train.py
```

### Evaluation
```bash
python Exp.py
```

### Demo
```bash
python demo.py
```

### Requirements
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── PerceiverAR.py      # Core Perceiver AR implementation
├── Transformer.py      # Standard Transformer baseline
├── Train.py           # Training script with curriculum learning
├── Exp.py             # Comprehensive evaluation experiments
├── Utils.py           # Data loading and utility functions
├── demo.py            # Demonstration script
└── README.md          # This file
```

## Research Impact

This work contributes to the field of efficient sequence modeling by:
- Demonstrating the effectiveness of variable latent dimensions in Perceiver architectures
- Introducing curriculum learning strategies for cross-attention mechanisms
- Providing empirical evidence of memory-computation trade-offs in attention-based models

## Future Work

- Extension to larger-scale datasets (e.g., Books, Common Crawl)
- Investigation of adaptive curriculum learning schedules
- Application to other sequence modeling tasks (translation, summarization)
- Integration with modern attention optimizations (Flash Attention, etc.)

## Citation

```bibtex
@article{hawthorne2022perceiver_ar,
  title={General-purpose, long-context autoregressive modeling with Perceiver AR},
  author={Hawthorne, Curtis and Jaegle, Andrew and Cangea, C{\u{a}}t{\u{a}}lina and Borgeaud, Sebastian and Nash, Charlie and Malinowski, Mateusz and Dieleman, Sander and Vinyals, Oriol and Botvinick, Matthew and Simon, Ian and Sheahan, Hannah and Zeghidour, Neil and Alayrac, Jean-Baptiste and Carreira, Jo{\~a}o and Engel, Jesse},
  journal={arXiv preprint arXiv:2202.07765},
  year={2022},
  url={https://arxiv.org/abs/2202.07765}
}

```