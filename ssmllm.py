from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


MODEL_NAME = "gpt2"
DATASET_SPLIT = "train[:10%]"
MAX_LENGTH = 256
BATCH_SIZE = 8


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_tokenized_dataset(tokenizer: AutoTokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = load_dataset("roneneldan/TinyStories", split=DATASET_SPLIT)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset


class LanguageModelingDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset["input_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


def build_dataloader(tokenizer: AutoTokenizer) -> DataLoader:
    dataset = build_tokenized_dataset(tokenizer)
    train_ds = LanguageModelingDataset(dataset)
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


class GatedSSMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.u = nn.Linear(d_model, d_model)
        self.f = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        forget_gate = torch.sigmoid(self.f(x))
        update = torch.tanh(self.u(x))
        output_gate = torch.sigmoid(self.o(x))
        h = forget_gate * x + (1 - forget_gate) * update
        return self.dropout(self.norm(output_gate * h))


class CrossHeadRouter(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.router = nn.Linear(d_model * num_heads, d_model)

    def forward(self, head_outputs):
        return self.router(torch.cat(head_outputs, dim=-1))


class TokenWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        return self.ff(x)


class ParallelSSMHeads(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([GatedSSMBlock(d_model) for _ in range(num_heads)])
        self.router = CrossHeadRouter(d_model, num_heads)
        self.tokenwise_ffn = TokenWiseFeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        fused = self.router(head_outputs)
        fused = fused + self.tokenwise_ffn(fused)
        return self.norm(fused)


class HybridBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.parallel_ssm = ParallelSSMHeads(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ssm = self.parallel_ssm(x)
        return self.norm(x + x_ssm)


class LazySSMLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model=768, depth=2, num_heads=3, max_seq_len=MAX_LENGTH - 1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.Sequential(*[HybridBlock(d_model, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        _, seq_len = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        preds = torch.argmax(logits, dim=-1)
        repeat_penalty = (preds[:, 1:] == preds[:, :-1]).float().mean()
        loss = loss + 10.0 * repeat_penalty
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        total_steps = max(1, getattr(self.trainer, "estimated_stepping_batches", 1))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, repetition_penalty=5.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.max_seq_len :]
            logits = self(idx_cond)[:, -1, :]
            for batch_idx in range(idx.size(0)):
                for token in set(idx[batch_idx].tolist()):
                    logits[batch_idx, token] /= repetition_penalty
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 25)
            next_token = topk_indices.gather(1, torch.multinomial(topk_probs, 1))
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def sample_text(model: LazySSMLanguageModel, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    device = next(model.parameters()).device
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = model.generate(tokens, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def build_trainer() -> Trainer:
    use_gpu = torch.cuda.is_available()
    return Trainer(
        max_epochs=5,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else "32-true",
        callbacks=[ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)],
    )


def main():
    tokenizer = build_tokenizer()
    train_loader = build_dataloader(tokenizer)
    model = LazySSMLanguageModel(vocab_size=tokenizer.vocab_size)
    trainer = build_trainer()
    trainer.fit(model, train_loader)
    print(
        sample_text(
            model,
            tokenizer,
            "The arabs steam through the golden horn, about to besiege constantinople. Emperor Leo 3 engages the arab fleet with cunning tactics, including:",
        )
    )


if __name__ == "__main__":
    main()
