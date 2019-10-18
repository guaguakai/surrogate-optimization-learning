FILENAME='1017-1800-server'

NODES=40
NOISE=0.5
BUDGET=4
PROB=0.2

python3 generate_bar.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
