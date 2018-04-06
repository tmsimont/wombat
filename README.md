
<img align=left width=150 src="https://raw.githubusercontent.com/tmsimont/wombat/master/wombat.png" />

---

**wo**rd **m**atrix **bat**ches

---

<br clear="left" />

This code was developed as part of my Master's thesis research.

A paper is available that describes the methods in this package on IEEE:  
[Efficient and accurate Word2Vec implementations in GPU and shared-memory multicore architectures](http://ieeexplore.ieee.org/document/8091076/)

The work builds upon ideas presented in [BIDMach](https://github.com/BIDData/BIDMach/)
and further refined in [Intel's pWord2Vec](https://github.com/IntelLabs/pWord2Vec).


*Note that in its current state the code is in a bit of a mess. A lot of remnants of some related expiriments are left in the code... Substantial refactoring is needed.*


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



