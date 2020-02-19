
## Reinforcement Learning Enhanced Quantum-inspired Algorithm for Combinatorial Optimization

This is the official implementation of the paper ["Reinforcement Learning Enhanced Quantum-inspired Algorithm for Combinatorial Optimization"](https://arxiv.org/abs/2002.04676)


### Setup the environment

```
conda env create --name sim -f environment.yml
conda activate sim
pip install git+https://github.com/BeloborodovDS/baselines.git
conda install -c conda-forge tensorflow
```

### Run experiments:

- `make train_ref`: pre-train the agent an random problems (R3, FILM)
- `make train_R2`: pre-train the agent an random problems (R2, FILM)
- `make train_nofilm`: pre-train the agent an random problems (R3, no FILM)
- `make experiment_ref`: train the agent on graphs G1-G10 (fine-tune, R3, FILM)
- `make experiment_R2`: train the agent on graphs G1-G10 (fine-tune, R2, FILM)
- `make experiment_nofilm`: train the agent on graphs G1-G10 (fine-tune, R3, no FILM)
- `make experiment_scratch`: train the agent on graphs G1-G10 (from scratch, R3, FILM)
- `make experiment_R2_scratch`: train the agent on graphs G1-G10 (from scratch, R2, FILM)
- `make experiment_nofilm_scratch`: train the agent on graphs G1-G10 (from scratch, R3, no FILM)

See `plot.ipynb` for plots and tables from the paper.

### Data sources

Gset (data/G{i}.txt): [dataset link](https://web.stanford.edu/~yyye/yyye/Gset/)

Best cuts (data/gbench.txt): [paper link](https://www.researchgate.net/publication/257392755_Breakout_Local_Search_for_the_Max-Cutproblem)
