# Automatically Learning Compact Quality-aware Surrogates for Optimization Problems

This is the implementation of the paper **Automatically Learning Compact Quality-aware Surrogates for Optimization Problems** submitted to NeurIPS 2020. The paper includes three examples: adversarial modeling in network security games (NSG folder), movie recommendation with a submodular objective (movie folder), and a convex portfolio optimization (portfolio folder). Among these three, NSG uses synthetic data, movie recommendation uses the data from MovieLens (ml-25m), and portfolio optimization uses the data downloaded from Quandl using quandl API.

The commands to run each example are included in each folder. You will have to download the data from [MovieLens](https://grouplens.org/datasets/movielens/) in movie recommendation and apply for an API key from [Quandl](https://docs.quandl.com/docs/getting-started) in portfolio optimization before running the code.

All the implementations are written in Python3.

Here is a list of dependency:
- [qpth](https://locuslab.github.io/qpth/)
- [cvxpylayers](https://github.com/cvxgrp/cvxpylayers)
- scipy
- tqdm
- [quandl](https://pypi.org/project/Quandl/)

