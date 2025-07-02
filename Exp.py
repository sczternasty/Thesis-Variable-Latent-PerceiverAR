import os
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import tqdm
import math
import numpy as np

from PerceiverAR import PerceiverAR
from Transformer import Transformer
from Utils import enwik8, sample_batch

def decode_tokens(tokens):
    return "".join(chr(max(32, t)) for t in tokens)

@torch.no_grad()
def evaluate(model, data,  context_size, device, num_batches=500, latent_size=None):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    for i in tqdm.trange(num_batches):
        X, y = sample_batch(data, length=context_size, batch_size=32)

        x = X.to(device)
        y = y.to(device)
        if latent_size is not None:
            log_probs = model(x, latent_size)
        else:
            log_probs = model(x)
        loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), y.reshape(-1), reduction='mean')

        total_nll += loss.item()
        total_tokens += y.numel()

    elapsed = time.perf_counter() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    avg_loss = total_nll / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    bpb = (avg_loss / torch.log(torch.tensor(2.0))).item()
    throughput = total_tokens / elapsed

    return {
        "nll": avg_loss,
        "perplexity": perplexity,
        "bpb": bpb,
        "throughput": throughput,
        "peak_mem_gb": peak_mem,
        "inference_time": elapsed / num_batches
    }

def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_train, data_val, data_test = enwik8()
    print(f"Train size: {data_train.size(0)}, Val size: {data_val.size(0)}, Test size: {data_test.size(0)}")

    experiments = [
        {
            "name": "TX-256",
            "model_type": "transformer",
            "context_size": 256,
            "model_args": {
                "vocab_size": 256,
                "seq_length": 256,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 8
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\Transformer_256_stepfinal.pt"
        },
        {
            "name": "TX-512",
            "model_type": "transformer",
            "context_size": 512,
            "model_args": {
                "vocab_size": 256,
                "seq_length": 512,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 8
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\Transformer_512_stepfinal.pt"
        },
        {
            "name": "PAR-256",
            "model_type": "perceiver",
            "model_args": {
                "vocab_size": 256,
                "max_seq_len": 1024,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 6,
                "perceive_depth": 1,
                "dropout": 0.1
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\fixed_prefix_256_stepfinal.pt"
        },
        {
            "name": "PAR-512",
            "model_type": "perceiver",
            "model_args": {
                "vocab_size": 256,
                "max_seq_len": 1024,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 6,
                "perceive_depth": 1,
                "dropout": 0.1
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\fixed_prefix_512_stepfinal.pt"
        },
        {
            "name": "VL-Uniform",
            "model_type": "perceiver",
            "model_args": {
                "vocab_size": 256,
                "max_seq_len": 1024,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 6,
                "perceive_depth": 1,
                "dropout": 0.1
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\variable_uniform_64_512_stepfinal.pt"
        },
        {
            "name": "VL-Curriculum",
            "model_type": "perceiver",
            "model_args": {
                "vocab_size": 256,
                "max_seq_len": 1024,
                "emb_size": 512,
                "heads": 8,
                "num_layers": 6,
                "perceive_depth": 1,
                "dropout": 0.1
            },
            "checkpoint_path": r"S:\PycharmProjects\Transformer-from-scratches\checkpoints_2.0\curriculum_64_to_512_stepfinal.pt"
        }
    ]


    latent_sizes =  [32, 64, 96, 128, 160, 256, 384, 512, 768, 896, 1000]

    transformer_contexts = [256, 512, 1024]

    results = []

    for exp in experiments:
        print(f"\n{'=' * 50}")
        print(f"Evaluating {exp['name']}")
        print(f"{'=' * 50}")

        if not os.path.exists(exp['checkpoint_path']):
            print(f"Checkpoint not found: {exp['checkpoint_path']}")
            continue

        try:
            if exp['model_type'] == 'transformer':
                model = Transformer(**exp['model_args'])
                model = load_model_checkpoint(model, exp['checkpoint_path'], device)

                context_size = exp['context_size']
                print(f"Evaluating Transformer with context size: {context_size}")

                metrics = evaluate(model, data_val, context_size, device)
                metrics.update({
                    "model": exp["name"],
                    "context_size": context_size,
                    "cross_seq_len": None
                })

                print(f"Results: {metrics}")
                results.append(metrics)

            elif exp['model_type'] == 'perceiver':
                model = PerceiverAR(**exp['model_args'])
                model = load_model_checkpoint(model, exp['checkpoint_path'], device)

                for ls in latent_sizes:
                    if ls >= 1024:
                        continue

                    print(f"Evaluating PerceiverAR with latent size: {ls}")

                    context_size = min(1024, max(ls + 100, 512))

                    metrics = evaluate(model, data_test, context_size, device, latent_size = ls)
                    metrics.update({
                        "model": exp["name"],
                        "context_size": context_size,
                        "cross_seq_len": ls
                    })

                    print(f"results: {metrics}")
                    results.append(metrics)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {exp['name']}: {str(e)}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        print(f"\nSaved {len(results)} results to evaluation_results.csv")

        print("\nSummary of results:")
        print(df.groupby('model')[['bpb', 'perplexity', 'peak_mem_gb']].agg(['mean', 'min', 'max']))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()