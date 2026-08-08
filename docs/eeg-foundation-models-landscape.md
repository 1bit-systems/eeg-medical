# EEG Foundation Models — Landscape

Survey of EEG foundation models (2023–2025), condensed from
[gayalkuruppu/eeg-foundation-models](https://github.com/gayalkuruppu/eeg-foundation-models),
which accompanies:

> Kuruppu, Wagh, Kremen, Varatharajah. **EEG Foundation Models: A Critical Review of
> Current Progress and Future Directions.** Journal of Neural Engineering, 2025.
> [arXiv:2507.11783](https://arxiv.org/abs/2507.11783) ·
> [NeurIPS 2025 BBFM Workshop](https://openreview.net/forum?id=Iu6qVgtgUD)

Where ZUNA1.1 (our base) sits in this space: a masked **diffusion autoencoder**
(380M, 4D-RoPE over xyz+t, channel-agnostic) — unusual among the list below,
which is dominated by masked-autoencoder/BERT-style (EEGPT, LaBraM, BIOT,
BrainBERT, NeuroLM), Mamba/SSM (EEGMamba, FEMBA, Mentality), and reconstruction
(Brant) families.

## 2025

| Model | Venue | Family / Note | Link |
|---|---|---|---|
| LUNA | NeurIPS 2025 | topology-agnostic, efficient | [paper](https://openreview.net/forum?id=uazfjnFL0G) · [code](https://github.com/pulp-bio/biofoundation) · [HF](https://huggingface.co/thorir/LUNA) |
| FEMBA | EMBC 2025 | bidirectional Mamba | [paper](https://arxiv.org/abs/2502.06438v2) · [code](https://github.com/pulp-bio/BioFoundation) · [HF](https://huggingface.co/thorir/FEMBA) |
| LaBraM++ | arXiv | codebook-based | [paper](https://arxiv.org/abs/2505.16724) |
| MVPFormer | arXiv | multi-variate parallel attention, iEEG | [paper](https://arxiv.org/abs/2506.20354) · [code](https://github.com/IBM/multi-variate-parallel-transformer) |
| MIRepNet | arXiv | motor imagery | [paper](https://arxiv.org/abs/2507.20254) · [code](https://github.com/staraink/MIRepNet) · [HF](https://huggingface.co/starself/MIRepNet) |
| M4CEA | IEEE JBHI | childhood epilepsy | [paper](https://ieeexplore.ieee.org/abstract/document/11083595) · [code](https://github.com/Evigouse/M4CEA_Project) |
| EEGMamba | Neural Networks | Mamba | [paper](https://www.sciencedirect.com/science/article/pii/S0893608025006963) · [code](https://github.com/wjq-learning/EEGMamba) · [HF](https://huggingface.co/weighting666/EEGMamba) |
| CSBrain | NeurIPS 2025 | cross-scale spatiotemporal | [paper](https://openreview.net/forum?id=agcXjEHmyW) · [code](https://github.com/yuchen2199/CSBrain) |
| DIVER-0 | ICML GenBio WS | channel-equivariant | [paper](https://openreview.net/pdf?id=QTfifcFEiE) |
| CodeBrain | arXiv | decoupled interpretability | [paper](https://arxiv.org/abs/2506.09110) |
| EEG FM for BCI | arXiv | features of electrophysiology | [paper](https://arxiv.org/abs/2506.01867) |
| Nested DL | arXiv | brain signal | [paper](https://arxiv.org/abs/2410.03191) |
| ALFEE | arXiv | adaptive large FM | [paper](https://arxiv.org/abs/2505.06291) |
| CBraMod | ICLR 2025 | criss-cross | [paper](https://openreview.net/forum?id=NPNUHgHF2w) · [code](https://github.com/wjq-learning/CBraMod) · [HF](https://huggingface.co/weighting666/CBraMod) |
| GEFM | arXiv | graph-enhanced | [paper](https://arxiv.org/abs/2411.19507) |
| Large Cognition Model | arXiv | | [paper](https://arxiv.org/abs/2502.17464) |
| LEAD | arXiv | Alzheimer's detection | [paper](https://arxiv.org/abs/2502.01678) · [code](https://github.com/DL4mHealth/LEAD) |

## 2024

| Model | Venue | Family / Note | Link |
|---|---|---|---|
| EEGPT | NeurIPS 2024 | universal representation | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) · [code](https://github.com/BINE022/EEGPT) |
| BrainWave | arXiv | clinical | [paper](https://arxiv.org/pdf/2402.10251) |
| FoME | arXiv | adaptive temporal-lateral attention | [paper](https://arxiv.org/pdf/2409.12454) |
| NeuroLM | ICLR 2025 | language↔EEG | [paper](https://arxiv.org/pdf/2409.00101) · [code](https://github.com/935963004/NeuroLM) · [HF](https://huggingface.co/Weibang/NeuroLM) |
| Mentality | ICLR WS TS4H | Mamba | [paper](https://openreview.net/pdf?id=O6T38rRiFp) |
| Large Brain Model (LaBraM) | ICLR 2024 | codebook BERT | [paper](https://arxiv.org/pdf/2405.18765) · [code](https://github.com/935963004/LaBraM) |
| EEGFormer | AAAI SS 2024 | transferable | [paper](https://openreview.net/forum?id=MXRy6bYBfB) |

## 2023

| Model | Venue | Family / Note | Link |
|---|---|---|---|
| BIOT | ICLR 2023 | intracranial, self-supervised | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf) · [code](https://github.com/ycq091044/BIOT) |
| Brant | NeurIPS 2023 | intracranial reconstruction | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/535915d26859036410b0533804cee788-Paper-Conference.pdf) · [HF](https://huggingface.co/Daoze/Brant/tree/main) |
| Neuro-GPT | ISBI 2024 | | [paper](https://arxiv.org/pdf/2311.03764) · [code](https://github.com/wenhui0206/NeuroGPT) |
| BrainBERT | ICLR 2023 | intracranial | [paper](https://arxiv.org/abs/2302.14367) · [code](https://github.com/czlwang/BrainBERT) |

## Takeaway for this project

- Channel/topology-agnostic design is the emerging consensus (LUNA, DIVER-0, ZUNA).
- Mamba/SSM is the hot alternative family to transformers for long EEG sequences.
- Clinical evaluation (seizure/epilepsy: M4CEA, LEAD) is a thin slice of the
  literature — TUH fine-tuning + clinical benchmarks remains a differentiator.
- None of these run fully locally on edge NPU hardware; that's the 1bit angle.
