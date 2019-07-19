FILENAME="0719"
BUDGET=2
NODES=20
MINCUT=1

for VAR in {10..10}
	do
		echo $VAR
		SEED=$VAR
		METHOD=0
		python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=10 --number-samples=5 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=1 --number-targets=1 --mincut=$MINCUT
		METHOD=1
		python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=10 --number-samples=5 --learning-rate=0.05 --prob=0.35 --feature-size=3 --number-sources=1 --number-targets=1 --mincut=$MINCUT
	done

