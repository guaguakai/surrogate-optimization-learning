FILENAME="0824-2100-local"
BUDGET=2
NODES=30

for VAR in {1..10}
	do
		echo $VAR
		SEED=$VAR
		MINCUT=1
		METHOD=1
		python3 pathProbabilities.py --epochs=30 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=5 --number-samples=20 --learning-rate=0.001 --prob=0.25 --feature-size=10 --number-sources=2 --number-targets=2 --mincut=$MINCUT

	done
