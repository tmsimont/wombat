cuda-gdb --args ../wombat \
	-train ../data/text8 \
	-output ../output/vectors-cuda-$1$2$3.txt -binary 0 \
 	-cbow 0 -size 100 -window 8 -sample 1e-4 \
	-negative 5 \
	-hs 1 \
	-iter 10 \
	-num-threads $1 \
	-tcbs-per-thread 1 \
	-items-in-tcb 125 \
	-batch-size $2 \
	-batches-per-thread $3 \
	-senbs 1 \
	-debug 2 \
	-time-events 0 \