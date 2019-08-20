FILENAME="0819-server"
BUDGET=2
NODES=15

for VAR in {1..10}
	do
		echo $VAR
		SEED=$VAR
		MINCUT=0
		METHOD=1
		python3 pathProbabilities.py --epochs=26 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.002 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT

	done
