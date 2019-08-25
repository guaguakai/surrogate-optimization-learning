FILENAME="0825-server"
BUDGET=2
NODES=30

echo $VAR
SEED=$VAR
MINCUT=0
METHOD=1
python3 pathProbabilities.py --epochs=30 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=10 --number-samples=10 --learning-rate=0.001 --prob=0.2 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT

