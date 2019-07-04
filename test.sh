SEED=394
python3 pathProbabilities.py --epochs=50 --method=0 --seed=$SEED --budget=0.5 --distribution=1 --number-nodes=15 --number-graphs=1 --number-samples=100 --learning-rate=0.01 --prob=0.3 --feature-size=3 --number-sources=6 --number-targets=2
