FILENAME='1104-server'

NODES=30
BUDGET=2
PROB=0.2
CUTSIZE='0.5n'
NOISE=0.2

python3 exp4.py --filename=$FILENAME --number-nodes=$NODES --budget=$BUDGET --prob=$PROB --cut-size=$CUTSIZE --noise=$NOISE
