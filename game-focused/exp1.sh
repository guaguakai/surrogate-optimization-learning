FILENAME='2020-0518-1200-stochastic-server'

NODES=100
NOISE=0.2
BUDGET=3
PROB=0.2
CUTSIZE='10'
SELECTION='coverage'

python3 exp1.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB --cut-size=$CUTSIZE --block-selection=$SELECTION
