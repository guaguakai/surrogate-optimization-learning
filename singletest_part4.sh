FILENAME="0821-local"
BUDGET=2
NODES=15

for VAR in {1..5}
	do
		echo $VAR
		SEED=$VAR
		MINCUT=1
		METHOD=1
		python3 pathProbabilities.py --epochs=30 --fixed-graph=2 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=0 --number-nodes=$NODES --number-graphs=1 --number-samples=100 --learning-rate=0.005 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT

	done
