FILENAME="0723-timetest"
BUDGET=2
MINCUT=1

# for NODES in 10 12 14 16 18 20
for NODES in 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
	echo $NODES
	for VAR in {1..5}
	do
		echo $VAR
		SEED=$VAR
		METHOD=0
		python3 pathProbabilities.py --epochs=1 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.01 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT
		METHOD=1
		python3 pathProbabilities.py --epochs=1 --fixed-graph=0 --method=$METHOD --seed=$SEED --filename=$FILENAME --budget=$BUDGET --distribution=1 --number-nodes=$NODES --number-graphs=1 --number-samples=50 --learning-rate=0.01 --prob=0.3 --feature-size=5 --number-sources=2 --number-targets=2 --mincut=$MINCUT
	done
done

