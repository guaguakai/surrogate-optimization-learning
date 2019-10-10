FILENAME='1010-1400-server'

NODES=10
NOISE=0.0
BUDGET=3
PROB=0.3

python3 generate_fig4.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
