
## Setup the environment

```
conda env create --name sim -f environment.yml
conda activate sim
pip install git+https://github.com/BeloborodovDS/baselines.git
conda install -c conda-forge tensorflow
```

## Data sources

Gset (data/G{i}.txt): [dataset link](https://web.stanford.edu/~yyye/yyye/Gset/)

Best cuts (data/gbench.txt): [paper link](https://www.researchgate.net/publication/257392755_Breakout_Local_Search_for_the_Max-Cutproblem)
