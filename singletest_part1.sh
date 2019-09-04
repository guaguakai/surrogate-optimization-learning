FILENAME="0904-server"
BUDGET=3
# NODES=30
EPOCHS=51
PROB=0.2

echo $NODES
echo $VAR
SEED=$VAR
MINCUT=0
METHOD=0
python3 pathProbabilities.py --epochs=$EPOCHS --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=100 --learning-rate=0.001 --prob=$PROB --feature-size=16 --number-sources=2 --number-targets=2 --mincut=$MINCUT

