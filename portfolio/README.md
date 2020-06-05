# Portfolio Optimization

There are four methods implemented in this domain:
1. two-stage (TS)
2. decision-focused (DF)
3. surrogate learning (surrogate)

Before running the code, you should apply for an [Quandl API](https://docs.quandl.com/docs#section-authentication) in order to download all the data between 2004-2017 from Quandl.

All TS, DF, and surrogate methods are implemented in `facilityUtils.py` file, which can be run by:

Two-stage method:
```
python3 main.py --epochs=100 --filepath='test' --lr=0.01 --n=50 --num-samples=0 --method=0
```

Decision-focused method
```
python3 main.py --epochs=100 --filepath='test' --lr=0.01 --n=50 --num-samples=0 --method=1
```

Surrogate learning method
```
python3 main.py --epochs=100 --filepath='test' --lr=0.01 --n=50 --num-samples=0 --T=5 --method=2
```

The parameter $n$ refers to the number of securities taken as candidate set. We aim to pick a portfolio of available securities to maximize the future return with a risk penalty term. The features are composed of previous stock prices and the rolling averages. The dataset includes all the daily prices of all 505 companies in SP500 between 2004 to 2017. We use a fully connected neural network with two layers to predict the future reture, and learn an embedding for each security to be used to compute the covariance matrix, where the covariance of two securities is the cosine similarity of their embeddings.
