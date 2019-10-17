FILENAME='1016-add-server'

NODES=10
NOISE=0.1
BUDGET=2
PROB=0.2

python3 generate_bar.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
