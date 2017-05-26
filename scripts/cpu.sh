#ncores=$1 # set this to #logical cores of your machine (with hyper-threading if available)
#export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
#numactl --interleave=all ../wombat \
../wombat \
	-train ../data/text8 \
	-output ../output/vectors-cpu-$1.txt -binary 0 \
 	-cbow 0 -size 100 -window 8 -sample 1e-4 \
	-negative 5 \
	-hs 0 \
	-iter 10 \
	-num-threads $1 \
	-debug 2 \
	-num-phys $1
