# LazyAttentionSSM Test Bed

Research sandbox for the LazyAttentionSSM architecture across three prototype tracks: time-series forecasting, language modeling, and text-conditioned image generation. The repository is intentionally lightweight and notebook-driven while benchmarking is still in progress.

## Repository Contents

- `cryptopred-ipynb (6).ipynb`: windowed crypto-market prediction experiment and transformer comparison
- `ssmllm.py`: standalone TinyStories language-model prototype using the LazyAttentionSSM blocks
- `ssmimggen (5).ipynb`: text-conditioned image-generation experiment built around an SSM-style UNet path
- `requirements.txt`: base Python dependencies for the local Anaconda data-science workflow

## Environment

This refresh is based on the local Anaconda `datascience312` workflow (Python 3.12):

```bash
conda activate datascience312
pip install -r requirements.txt
```

## Notes

- `ssmllm.py` is the most portable entrypoint in the repository.
- `cryptopred-ipynb (6).ipynb` is designed around Kaggle parquet inputs.
- `ssmimggen (5).ipynb` was built for Kaggle or TPU-backed environments and will need extra runtime setup such as `torch-xla`.
- No formal license file is included yet, so treat the repository as all rights reserved until one is added.
