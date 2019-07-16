SEED=1200
FILENAME="0717-0300"
METHOD=0
BUDGET=2
NODES=20

python3 pathProbabilities.py --epochs=20 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=0 --number-nodes=$NODES --number-graphs=100 --number-samples=1 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=2 --number-targets=2
