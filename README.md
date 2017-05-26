<p align="center">
	<img src="https://raw.githubusercontent.com/tmsimont/wombat/master/wombat.png" />
</p>

---

To compile, use make:

For g++:
```
make
```

For CUDA:
```
make cuda
```

For MKL support and ICPC:
```
make intel
```

Once made, you can use the scripts in /scripts to run test programs:

Testing g++ or icpc compiled program:
```
./cpu.sh [num threads]
```

Testing CUDA (requries 6.0 CUDA capability):
```
./cuda.numCPUT-batchSize-batchesPerT.sh  [num cpu threads] [batch size] [batches per thread]
```

For all programs, to get test data:
```
./get-data.sh
```



