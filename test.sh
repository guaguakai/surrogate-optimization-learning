SEED=3824
FILENAME="0704-2300"
METHOD=0
BUDGET=2
NODES=15

python3 pathProbabilities.py --epochs=25 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=100 --number-samples=1 --learning-rate=0.01 --prob=0.35 --feature-size=3 --number-sources=2 --number-targets=2
