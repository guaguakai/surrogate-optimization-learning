# FILENAME="0905-server-var0.1"
# BUDGET=2
# NODES=30
EPOCHS=51
# PROB=0.2

echo $NODES
echo $VAR
SEED=$VAR
MINCUT=1
METHOD=1
python3 blockQP.py --epochs=$EPOCHS --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=100 --learning-rate=0.001 --prob=$PROB --feature-size=16 --number-sources=2 --number-targets=2 --mincut=$MINCUT --noise=$NOISE

