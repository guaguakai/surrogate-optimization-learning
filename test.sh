SEED=3824
filename="0703-2300"
python3 pathProbabilities.py --epochs=25 --method=0 --seed=$SEED --filename=$filename --budget=2 --distribution=1 --number-nodes=15 --number-graphs=1 --number-samples=50 --learning-rate=0.01 --prob=0.4 --feature-size=3 --number-sources=2 --number-targets=2
