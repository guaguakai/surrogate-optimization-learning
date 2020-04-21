FILENAME='2020-0410-stochastic-server'

NODES=80
NOISE=0.2
BUDGET=2
PROB=0.2
CUTSIZE='10'
SELECTION='coverage'

python3 exp1.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB --cut-size=$CUTSIZE --block-selection=$SELECTION
