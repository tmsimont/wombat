#include "src/console.h"
#include "src/batch_model.h"
#include "src/cuda_batch_model.h"

int main(int argc, char **argv) {
	if (readConsoleArgs(argc, argv)) {
		CUDABatchModel t;
		t.init();
		t.train();

		double now = omp_get_wtime();
		printf("\nFinal: Alpha: %f  Progress: %.2f%%  Words/sec: %.2fk\n",  alpha,
						word_count_actual / (real) (iter * train_words + 1) * 100,
						word_count_actual / ((now - start) * 1000));
		if (time_events) printFinalTimers();

		saveModel();
	}
	return 0;
}
