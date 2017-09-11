<p align="center">
	<img src="https://raw.githubusercontent.com/tmsimont/wombat/master/wombat.png" />
</p>

---

**wo**rd **m**atrix **bat**ches

---

This code was developed as part of my Master's thesis research.

A summary of the methods used and motivation for this code will be 
published in the conference proceedings at the 
[2017 HPEC conference](http://www.ieee-hpec.org/).
I will present this work on September 13, 2017.

The work builds upon ideas presented in [BIDMach](https://github.com/BIDData/BIDMach/)
and further refined in [Intel's pWord2Vec](https://github.com/IntelLabs/pWord2Vec).


*Note that in its current state it is in a bit of a mess. Substantial clean up is needed.*


This code supports:

 * Both CPU and GPU matrix-based fast Word2Vec
 * Both SkipGram and Hierarchical Softmax Word2Vec architectures

This code **does not** support:

 * Distributed computing techniques (see pWord2Vec)
 * CBOW Word2Vec architectures


## Installation

The make file (hackishly) supports g++, CUDA or ICPC.

Different source files are used for different compilers.

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



