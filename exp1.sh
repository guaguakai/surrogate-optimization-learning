FILENAME='1026-1600-server'

NODES=20
NOISE=0.5
BUDGET=2
PROB=0.2
CUTSIZE=0.5

python3 exp1.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB --cut-size=$CUTSIZE
