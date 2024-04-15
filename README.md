# TIGER-Temporally-Improved-Graph-Entity-Linker

The implementation of our approach is based on the original codebase [BLINK](https://github.com/facebookresearch/BLINK) and [AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN).<br>

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

In this work, we introduce TIGER: a Temporally Improved Graph Entity Linker. By incorporating structural information between entities into the model, we enhance the learned representation, making entities more distinguishable over time. The core idea is to integrate graph-based information onto text-based information, from which both distinct and shared embeddings based on an entities' feature and structural relationships and their interaction. Experiments on three datasets show that our model can effectively prevent temporal degradation, demonstrating a 16.24% performance boost over the state-of-the-art in a temporal setting when the time gap is one year and an improvement to 20.93% as the gap expands to three years.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n tiger39 -y python=3.9 && conda activate tiger39
pip install -r requirements.txt
```

### 2. Download the data

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| [Our Dataset](https://drive.google.com/drive/folders/1DeHi-cvVOAdYFA4GljaBvpuG0wiYpgch?usp=sharing) | 3.12 GB            |
| [ZESHEL](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel) | 1.55 GB            |
| [WikiLinksNED](https://github.com/yasumasaonoe/ET4EL) | 1.1 GB             |

### 3. Reproduce the experiments

```
train.sh
```

<table>
    <tr>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>0</td>
        <td>1</td>
        <td>2</td>
        <td>3</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>Continual Entities</td>
        <td></td>
        <td></td>
        <td></td>
        <td>New Entities</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>@1</td>
        <td>BLINK</td>
        <td>0.177</td>
        <td>0.181</td>
        <td>0.182</td>
        <td>0.177</td>
        <td>0.132</td>
        <td>0.132</td>
        <td>0.132</td>
        <td>0.142</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.229</td>
        <td>0.234</td>
        <td>0.228</td>
        <td>0.221</td>
        <td>0.172</td>
        <td>0.169</td>
        <td>0.167</td>
        <td>0.192</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.29</td>
        <td>0.292</td>
        <td>0.297</td>
        <td>0.304</td>
        <td>0.186</td>
        <td>0.195</td>
        <td>0.188</td>
        <td>0.217</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>26.76</td>
        <td>24.83</td>
        <td>30.22</td>
        <td>37.53</td>
        <td>8.6</td>
        <td>15.15</td>
        <td>12.47</td>
        <td>12.73</td>
    </tr>
    <tr>
        <td>@2</td>
        <td>BLINK</td>
        <td>0.26</td>
        <td>0.265</td>
        <td>0.268</td>
        <td>0.263</td>
        <td>0.197</td>
        <td>0.197</td>
        <td>0.198</td>
        <td>0.211</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.32</td>
        <td>0.328</td>
        <td>0.327</td>
        <td>0.322</td>
        <td>0.239</td>
        <td>0.247</td>
        <td>0.258</td>
        <td>0.261</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.404</td>
        <td>0.409</td>
        <td>0.414</td>
        <td>0.425</td>
        <td>0.274</td>
        <td>0.285</td>
        <td>0.277</td>
        <td>0.314</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>26.31</td>
        <td>24.54</td>
        <td>26.54</td>
        <td>31.79</td>
        <td>14.52</td>
        <td>15.38</td>
        <td>7.38</td>
        <td>20.32</td>
    </tr>
    <tr>
        <td>@4</td>
        <td>BLINK</td>
        <td>0.357</td>
        <td>0.364</td>
        <td>0.367</td>
        <td>0.362</td>
        <td>0.277</td>
        <td>0.277</td>
        <td>0.278</td>
        <td>0.294</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.429</td>
        <td>0.436</td>
        <td>0.43</td>
        <td>0.429</td>
        <td>0.329</td>
        <td>0.34</td>
        <td>0.333</td>
        <td>0.354</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.52</td>
        <td>0.524</td>
        <td>0.53</td>
        <td>0.543</td>
        <td>0.374</td>
        <td>0.389</td>
        <td>0.381</td>
        <td>0.421</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>21.13</td>
        <td>20.38</td>
        <td>23.23</td>
        <td>26.36</td>
        <td>13.65</td>
        <td>14.38</td>
        <td>14.36</td>
        <td>18.79</td>
    </tr>
    <tr>
        <td>@8</td>
        <td>BLINK</td>
        <td>0.463</td>
        <td>0.469</td>
        <td>0.475</td>
        <td>0.47</td>
        <td>0.37</td>
        <td>0.37</td>
        <td>0.374</td>
        <td>0.392</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.546</td>
        <td>0.544</td>
        <td>0.554</td>
        <td>0.548</td>
        <td>0.423</td>
        <td>0.44</td>
        <td>0.439</td>
        <td>0.472</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.628</td>
        <td>0.632</td>
        <td>0.637</td>
        <td>0.652</td>
        <td>0.483</td>
        <td>0.498</td>
        <td>0.49</td>
        <td>0.533</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>15.11</td>
        <td>16.14</td>
        <td>14.97</td>
        <td>18.9</td>
        <td>14.34</td>
        <td>13.17</td>
        <td>11.76</td>
        <td>13.04</td>
    </tr>
    <tr>
        <td>@16</td>
        <td>BLINK</td>
        <td>0.571</td>
        <td>0.576</td>
        <td>0.581</td>
        <td>0.578</td>
        <td>0.472</td>
        <td>0.471</td>
        <td>0.474</td>
        <td>0.491</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.645</td>
        <td>0.645</td>
        <td>0.656</td>
        <td>0.652</td>
        <td>0.539</td>
        <td>0.541</td>
        <td>0.554</td>
        <td>0.551</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.724</td>
        <td>0.728</td>
        <td>0.733</td>
        <td>0.744</td>
        <td>0.592</td>
        <td>0.604</td>
        <td>0.599</td>
        <td>0.638</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>12.36</td>
        <td>12.95</td>
        <td>11.76</td>
        <td>14.24</td>
        <td>9.74</td>
        <td>11.73</td>
        <td>8.22</td>
        <td>15.76</td>
    </tr>
    <tr>
        <td>@32</td>
        <td>BLINK</td>
        <td>0.675</td>
        <td>0.68</td>
        <td>0.685</td>
        <td>0.683</td>
        <td>0.576</td>
        <td>0.576</td>
        <td>0.577</td>
        <td>0.593</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.732</td>
        <td>0.739</td>
        <td>0.744</td>
        <td>0.741</td>
        <td>0.641</td>
        <td>0.646</td>
        <td>0.637</td>
        <td>0.673</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.807</td>
        <td>0.809</td>
        <td>0.812</td>
        <td>0.821</td>
        <td>0.694</td>
        <td>0.704</td>
        <td>0.702</td>
        <td>0.732</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>10.24</td>
        <td>9.38</td>
        <td>9.14</td>
        <td>10.8</td>
        <td>8.27</td>
        <td>9.06</td>
        <td>10.13</td>
        <td>8.77</td>
    </tr>
    <tr>
        <td>@64</td>
        <td>BLINK</td>
        <td>0.769</td>
        <td>0.774</td>
        <td>0.778</td>
        <td>0.776</td>
        <td>0.677</td>
        <td>0.676</td>
        <td>0.679</td>
        <td>0.694</td>
    </tr>
    <tr>
        <td></td>
        <td>SpEL</td>
        <td>0.82</td>
        <td>0.827</td>
        <td>0.825</td>
        <td>0.824</td>
        <td>0.732</td>
        <td>0.733</td>
        <td>0.739</td>
        <td>0.754</td>
    </tr>
    <tr>
        <td></td>
        <td>TIGER</td>
        <td>0.871</td>
        <td>0.872</td>
        <td>0.874</td>
        <td>0.881</td>
        <td>0.783</td>
        <td>0.791</td>
        <td>0.79</td>
        <td>0.813</td>
    </tr>
    <tr>
        <td></td>
        <td>Boost (%)</td>
        <td>6.25</td>
        <td>5.43</td>
        <td>5.91</td>
        <td>6.86</td>
        <td>7.08</td>
        <td>7.85</td>
        <td>6.82</td>
        <td>7.84</td>
    </tr>
    <tr>
        <td>Total Average Boost (%)</td>
        <td></td>
        <td>16.88</td>
        <td>16.24</td>
        <td>17.4</td>
        <td>20.93</td>
        <td>10.89</td>
        <td>12.39</td>
        <td>10.16</td>
        <td>13.89</td>
    </tr>
</table>

## Using your own data

If you want to use your own dataset, you only need to use the code in Dataset Construction. Construct your own dataset according to the description of the dataset construction process in the Supplementary Material.
