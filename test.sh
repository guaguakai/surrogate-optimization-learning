SEED=3824
FILENAME="0704-1900"
METHOD=0
BUDGET=1

python3 pathProbabilities.py --epochs=25 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=15 --number-graphs=1 --number-samples=10 --learning-rate=0.1 --prob=0.35 --feature-size=3 --number-sources=2 --number-targets=2
