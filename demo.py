"""
Demo script for Variable Latent Perceiver AR project

This script demonstrates the key capabilities of the implemented models
and provides a quick way to test the system.
"""

import torch
import torch.nn.functional as F
from PerceiverAR import PerceiverAR
from Transformer import Transformer
from Utils import sample_batch, decode_tokens

def demo_perceiver_ar():
    """Demonstrate Perceiver AR model capabilities"""
    print("=" * 60)
    print("PERCEIVER AR DEMONSTRATION")
    print("=" * 60)
    
    model = PerceiverAR(
        vocab_size=256,
        max_seq_len=1024,
        emb_size=512,
        heads=8,
        num_layers=6,
        perceive_depth=1,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model architecture: {model}")
    
    batch_size, seq_len = 2, 256
    x = torch.randint(0, 256, (batch_size, seq_len))
    
    latent_sizes = [32, 64, 128, 256]
    
    for latent_size in latent_sizes:
        print(f"\nTesting with latent size: {latent_size}")
        
        with torch.no_grad():
            output = model(x, latent_size)
            
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output logits range: [{output.min():.3f}, {output.max():.3f}]")
        
        if latent_size <= 128:
            try:
                sample_output = F.softmax(output[0, -10:], dim=-1)
                sample_tokens = torch.multinomial(sample_output, 1).squeeze()
                sample_text = decode_tokens(sample_tokens)
                print(f"Sample generated text: {sample_text}")
            except Exception as e:
                print(f"Text generation failed: {e}")

def demo_transformer():
    """Demonstrate Transformer baseline model"""
    print("\n" + "=" * 60)
    print("TRANSFORMER BASELINE DEMONSTRATION")
    print("=" * 60)
    
    model = Transformer(
        vocab_size=256,
        seq_length=512,
        emb_size=512,
        heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model architecture: {model}")
    
    batch_size, seq_len = 2, 256
    x = torch.randint(0, 256, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min():.3f}, {output.max():.3f}]")

def demo_training_strategies():
    """Demonstrate different training strategies"""
    print("\n" + "=" * 60)
    print("TRAINING STRATEGIES DEMONSTRATION")
    print("=" * 60)
    
    strategies = [
        ("Fixed Prefix 256", lambda step: 256),
        ("Fixed Prefix 512", lambda step: 512),
        ("Variable Uniform (64-512)", lambda step: torch.randint(64, 513, (1,)).item()),
        ("Curriculum (64→512)", lambda step: min(512, 64 + (step // 1000) * 32))
    ]
    
    for name, strategy_fn in strategies:
        print(f"\n{name}:")
        for step in [0, 1000, 5000, 10000]:
            latent_size = strategy_fn(step)
            print(f"  Step {step:5d}: Latent size = {latent_size:3d}")

def main():
    """Run all demonstrations"""
    print("VARIABLE LATENT PERCEIVER AR PROJECT DEMO")
    print("This script demonstrates the key capabilities of the implemented models.")
    
    try:
        demo_perceiver_ar()
        demo_transformer()
        demo_training_strategies()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This might be due to missing dependencies or GPU requirements.")

if __name__ == "__main__":
    main()
