# TIGER-Temporally-Improved-Graph-Entity-Linker

The implementation of our approach is based on the original codebase [BLINK](https://github.com/facebookresearch/BLINK) and [AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN).<br>

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

In this work, we introduce TIGER: a Temporally Improved Graph Entity Linker. By incorporating structural information between entities into the model, we enhance the learned representation, making entities more distinguishable over time. The core idea is to integrate graph-based information onto text-based information, from which both distinct and shared embeddings based on an entities' feature and structural relationships and their interaction. Experiments on three datasets, show that our model can effectively prevent temporal degradation, demonstrating a 2.55% performance boost over a strong baseline when the time gap is one year, and an improvement to 18.83% as the interval expands to nine years.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n tiger39 -y python=3.9 && conda activate tiger39
pip install -r requirements.txt
```

### 2. Download the data

[Our Dataset](https://drive.google.com/drive/folders/1DeHi-cvVOAdYFA4GljaBvpuG0wiYpgch?usp=sharing)<br>
[ZESHEL](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel)<br>
[WikiLinksNED](https://github.com/yasumasaonoe/ET4EL)<br>

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| [Our Dataset](https://drive.google.com/drive/folders/1DeHi-cvVOAdYFA4GljaBvpuG0wiYpgch?usp=sharing) | 3.12 GB            |
| [ZESHEL](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel) | 1.55 GB            |
| [WikiLinksNED](https://github.com/yasumasaonoe/ET4EL) | 1.1 GB             |

### 3. Reproduce the experiments

```
train.sh
```

## Using your own data

If you want to use your own dataset, you only need to use the code in Dataset Construction. Construct your own dataset according to the description of the dataset construction process in the Supplementary Material.
