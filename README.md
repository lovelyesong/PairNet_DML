# PairNet_DML: Causal Estimation with DML Utilizing PairNet-Trained Representations
Yesong Choe, Yeahoon Kwon

(Note that PairNet is developed by Lokesh Nagalapatti, Pranava Singhal, Avishek Ghosh, Sunita Sarawagi)

# To install catenets, use the following commands:
```
conda create -n catenets numpy pandas python=3.9 scipy 
pip3 install torch
pip install jax
pip install jaxlib
pip install loguru
pip install pytest scikit_learn
pip install gdown
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ott-jax
```
---


### To run this, generate a bootstrapped dataset first using "fetch_bonus.ipynb"

```
python run_experiments_benchmarks_NeurIPS.py --experiment bonus --file_name pairnet --n_exp 500
```


