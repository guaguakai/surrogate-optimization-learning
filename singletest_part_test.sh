FILENAME="0823-test"
BUDGET=2
NODES=15

for VAR in {21..25}
	do
		echo $VAR
		SEED=$VAR
		MINCUT=0
		METHOD=0
		python3 pathProbabilities.py --epochs=30 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=100 --learning-rate=0.001 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT

	done
