FILENAME='1102-server'

NODES=30
NOISE=0.2
BUDGET=2
PROB=0.2

python3 exp2.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
