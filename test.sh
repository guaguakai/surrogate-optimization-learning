FILENAME="0719"
BUDGET=2
NODES=20
SEED=1
METHOD=1
MINCUT=1

python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=1 --number-targets=1 --mincut=$MINCUT

