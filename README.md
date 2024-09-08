# BigVGAN

A single-file implementation of BigVGAN generator (see https://arxiv.org/abs/2206.04658).
The original implementation can be found at https://github.com/NVIDIA/BigVGAN/tree/v2.4.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Python 3.8 or later](https://www.python.org).
Clone or download and extract the repository, open a terminal and run:

```
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

```python
import torch
import torchaudio
from bigvgan import BigVGAN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigVGAN().to(device)
model.eval().requires_grad_(False)
wav_path = "sample.wav"
sig, orig_sample_rate = torchaudio.load(wav_path)
sig = sig.to(device)
with torch.no_grad():
    feats = model.extract_features(sig, orig_sample_rate)
    rec_sig = model(feats).cpu()
torchaudio.save("reconstruction.wav", rec_sig[0, :], model.sample_rate)
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
