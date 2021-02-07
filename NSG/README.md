# Adversarial Learning in Network Security Games

There are four methods implemented in this domain:
1. two-stage (TS)
2. decision-focused (DF)
3. block decision-focused (block)
4. surrogate learning (surrogate)


TS, DF, block are implemented in `blockQP.py` file, while the surrogate approach is implemented in 'surrogate.py'.

To run TS, DF, and block methods, please run the following commands respectively:

Two-stage method:
```
python3 blockQP.py --number-nodes=50 --number-sources=5 --number-targets=5 --budget=3 --filename='test' --feature-size=16 --number-samples=50 --noise=0.2 --learning-rate=0.01 --epochs=100 --seed=1 --method=0
**REMARK**: it is normal if you find the objective to be `-inf` in the two-stage mode. This is because we disable the computation of objective in the training epochs, where two-stage method does not need it and it is extremely slow to evaluate the objective/utility.
```

Decision-focused method:
```
python3 blockQP.py --number-nodes=50 --number-sources=5 --number-targets=5 --budget=3 --filename='test' --feature-size=16 --number-samples=50 --noise=0.2 --learning-rate=0.01 --epochs=100 --seed=1 --method=1
```

Block decision-focused learning method:
```
python3 blockQP.py --number-nodes=50 --number-sources=5 --number-targets=5 --budget=3 --filename='test' --feature-size=16 --number-samples=50 --noise=0.2 --learning-rate=0.01 --epochs=100 --seed=1 --method=2
```


To run surrogate approach, please run the following commands:
```
python3 surrogate.py --number-nodes=50 --number-sources=5 --number-targets=5 --budget=3 --filename='test' --feature-size=16 --number-samples=50 --noise=0.2 --learning-rate=0.01 --epochs=100 --T=5
```

In the above arguments, number of nodes, sources, targets, noise, feature size, and number of samples are related to the data and sample generation part. Filename indicates the filename prefix that you would like to save within `results` folder. `T` is the reparameterization size chosen in the surrogate approach. The larger the `T` is, the longer it takes to run the training and inference. Larger `T` can also imply better representational power but less generalizability as shown in our paper. We set `T` to be 10% of the problem size throughout the experiments. Random seed can be fixed by assigning the random seed to `seed`. It can help reduce the variance of different initial setup and compare the performance from the same data generation.
