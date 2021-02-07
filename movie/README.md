# Bulk Movie Recommendation

There are four methods implemented in this domain:
1. two-stage (TS)
2. decision-focused (DF)
3. surrogate learning (surrogate)

Before running the code, you should download the MovieLens 25m data from [MovieLens](https://grouplens.org/datasets/movielens/) and put the dataset under folder `data/ml-25m/`.

All TS, DF, and surrogate methods are implemented in `facilityUtils.py` file, which can be run by:

- Two-stage
```
python3 main.py --epochs=100 --filepath='test' --budget=10 --lr=0.01 --n=50 --m=100 --features=200 --num-samples=0 --seed=1 --method=0
```

- Decision-focused
```
python3 main.py --epochs=100 --filepath='test' --budget=10 --lr=0.01 --n=50 --m=100 --features=200 --num-samples=0 --seed=1 --method=1
```

- Surrogate learning
```
python3 main.py --epochs=100 --filepath='test' --budget=10 --lr=0.01 --n=50 --m=100 --features=200 --num-samples=0 --seed=1 --T=5 --method=2
```

The parameters `n` refers to how many candidate movies are under consideration. `m` refers to how many users in each group or training and testing instance. We run a bulk recommendation to broadcast budget=10 movies from `n` candidate movies to all `m` users in the same group. The movies are randomly selected from all movies in the MovieLens dataset. We further select num-features movies as anchor users' features, which are given in advance to identify different users. The dataset is filtered by choosing only the movies that are either in the candidate set or the movies that are used as users' features. The remaining users are then groups into groups with size `m`. We randomly select num-samples groups as our entire training, validation, testing dataset, where when num-samples is set to be 0, the entire dataset is used. Lastly, `T` refers to the reparameterization size used in the surrogate approach. Similarly, random seed can be set by the variable `seed`. It can help fix the randomness of candidate set and the group division.
