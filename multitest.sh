FILENAME="0722"
BUDGET=2
NODES=20
MINCUT=1

for VAR in {21..30}
	do
		echo $VAR
		SEED=$VAR
		METHOD=0
		python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.01 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT
		METHOD=1
		python3 pathProbabilities.py --epochs=50 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.01 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT
	done

