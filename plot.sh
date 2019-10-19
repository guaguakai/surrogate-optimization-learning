FILENAME='1018-server'

NODES=20
NOISE=0.2
BUDGET=2
PROB=0.2

python3 generate_bar.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
python3 generate_fig4.py --filename=$FILENAME --number-nodes=$NODES --noise=$NOISE --budget=$BUDGET --prob=$PROB
