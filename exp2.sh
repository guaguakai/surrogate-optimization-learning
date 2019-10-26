FILENAME='1025-server'

NODES=20
NOISE=0.0
BUDGET=2
PROB=0.2
CUTSIZE=0.1

python3 exp2.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
