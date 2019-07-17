SEED=383
FILENAME="0717-1000"
METHOD=1
BUDGET=2
NODES=20

python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=6 --number-targets=2
