SEED=3822
FILENAME="0716-0900"
METHOD=1
BUDGET=1
NODES=15

python3 pathProbabilities.py --epochs=100 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=2 --number-targets=2
