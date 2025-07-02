import gzip
import random
import time
import math
import os
import numpy as np
import torch
import torch.optim as optim
import wandb
import tqdm

from PerceiverAR import PerceiverAR
from Utils import enwik8, sample_batch, sample_sequence

NUM_BATCHES = int(1e4)
BATCH_SIZE = 32
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024


def decode_tokens(tokens):
    return "".join(chr(max(32, t)) for t in tokens)

data_train, data_val, data_test = enwik8()
print(f"Train size: {data_train.size(0)}, Val size: {data_val.size(0)}, Test size: {data_test.size(0)}")

bucket = 10000 / 14
experiments = [
    {
        "name": "fixed_prefix_256",
        "prefix_fn": lambda step: 256
    },
{
        "name": "fixed_prefix_512",
        "prefix_fn": lambda step: 512
    },
    {
        "name": "variable_uniform_64_512",
        "prefix_fn": lambda step: random.randint(64, 512)
    },
    {
        "name": "curriculum_64_to_512",
        "prefix_fn": lambda step: min(512, 64 + int(step // bucket) * 32)
    }
]

CHECKPOINT_EVERY = 1_000
CKPT_DIR = "./checkpoints_2.0"
os.makedirs(CKPT_DIR, exist_ok=True)


def save_ckpt(run_name, step, model, optimizer):
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_step{step}.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


def compute_loss(model, X, y, cross_seq_len, device):

    X = X.to(device)
    y = y.to(device)

    log_probs = model(X, cross_seq_len)

    loss = torch.nn.functional.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        y.reshape(-1),
        reduction='mean'
    )

    return loss


def generate_sample(model, val_data, cross_seq_len, max_context, device, length=512, temp=0.8):
    model.eval()

    start_idx = torch.randint(0, val_data.size(0) - max_context, (1,))
    seed = val_data[start_idx:start_idx + max_context]

    generated = sample_sequence(
        model=model,
        seed=seed,
        cross_seq_len=cross_seq_len,
        max_context=max_context,
        length=length,
        temp=temp,
        device=device
    )

    return decode_tokens(generated)

def run_experiment(exp_cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PerceiverAR(
        vocab_size=256,
        max_seq_len=SEQ_LEN,
        emb_size=512,
        heads=8,
        num_layers=6,
        perceive_depth=1,
        dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

    wandb.init(
        project="perceiver-ar-enwik8-custom",
        name=exp_cfg["name"],
        config={
            "batch_size": BATCH_SIZE,
            "grad_accum": GRADIENT_ACCUMULATE_EVERY,
            "learning_rate": LEARNING_RATE,
            "seq_len": SEQ_LEN,
            "vocab_size": 256,
            "emb_size": 512,
            "heads": 8,
            "num_layers": 6,
            "perceive_depth": 1,
            **{k: v for k, v in exp_cfg.items() if k != "prefix_fn"}
        }
    )

    for step in tqdm.trange(NUM_BATCHES):
        model.train()
        cross_seq_len = exp_cfg["prefix_fn"](step)

        total_loss = 0
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            X, y = sample_batch(data_train, length=SEQ_LEN, batch_size=BATCH_SIZE)

            loss = compute_loss(model, X, y, cross_seq_len, device)
            loss = loss / GRADIENT_ACCUMULATE_EVERY
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            mem_alloc = mem_reserved = max_mem_alloc = 0

        total_loss *= GRADIENT_ACCUMULATE_EVERY
        bpb = total_loss / math.log(2)
        ppl = math.exp(total_loss)

        log_dict = {
            "step": step,
            "loss": total_loss,
            "bpb": bpb,
            "ppl": ppl,
            "cross_seq_len": cross_seq_len,
        }

        if torch.cuda.is_available():
            log_dict.update({
                "gpu_mem_GB": mem_alloc,
                "gpu_reserved_GB": mem_reserved,
                "gpu_max_alloc_GB": max_mem_alloc,
            })

        wandb.log(log_dict)

        if (step % CHECKPOINT_EVERY) == 0 and step > 0:
            save_ckpt(exp_cfg["name"], step, model, optimizer)

        # Validation
        if step % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                X_val, y_val = sample_batch(data_val, length=SEQ_LEN, batch_size=BATCH_SIZE)
                val_loss = compute_loss(model, X_val, y_val, cross_seq_len, device)

            wandb.log({
                "val_loss": val_loss.item(),
                "val_bpb": val_loss.item() / math.log(2)
            })

        if step % GENERATE_EVERY == 0:
            try:
                generated_text = generate_sample(
                    model=model,
                    val_data=data_val,
                    cross_seq_len=cross_seq_len,
                    max_context=min(512, SEQ_LEN - 1),
                    device=device,
                    length=GENERATE_LENGTH,
                    temp=0.8
                )
                wandb.log({"generated": generated_text})
            except Exception as e:
                print(f"Generation failed at step {step}: {e}")

    save_ckpt(exp_cfg["name"], "final", model, optimizer)
    wandb.finish()

if __name__ == "__main__":
    for cfg in experiments:
        run_experiment(cfg)