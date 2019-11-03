# FILENAME="0905-server-var0.1"
# BUDGET=2
# NODES=30
# EPOCHS=101
# PROB=0.2
LR=0.001

echo $NODES
echo $VAR
SEED=$VAR
METHOD=2
python3 blockQP.py --epochs=$EPOCHS --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=$SAMPLES --learning-rate=$LR --prob=$PROB --feature-size=16 --number-sources=5 --number-targets=5 --noise=$NOISE --cut-size=$CUTSIZE --block-selection=$SELECTION
