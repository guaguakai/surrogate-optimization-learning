FILENAME='1004-1400-server'

NODES=20
NOISE=0.05
BUDGET=3
PROB=0.3

python3 generate_fig4.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
