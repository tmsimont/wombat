// Copyright 2017 Trevor Simonton

#include "src/console.h"


#include "src/buffers/sen_buffer.h"
int sentences_in_buffer = 10;
int sen_buffer_item_size = 1;

#include "src/pht_model.h"
int num_phys = 2;

#include "src/batch_model.h"
int batch_size = 256;
int batches_per_thread = 8;

#include "src/buffers/tc_buffer.h"
int tc_buffer_item_size;
int tcbs_per_thread = 1;
int items_in_tcb = 125;

double start;

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      return a;
    }
  return -1;
}

int readConsoleArgs(int argc, char **argv) {
  if (argc == 1) {
    printf("parallel word2vec (sgns) in shared memory system\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that\n");
    printf("\t\tappear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3,\n");
    printf("\t\tuseful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common\n");
    printf("\t\tvalues are 3 - 10 (0 = not used)\n");
    printf("\t-iter <int>\n");
    printf("\t\tNumber of training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int>\n");
    printf("\t\ttimes; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info\n");
    printf("\t\tduring training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded;\n");
    printf("\t\tdefault is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not\n");
    printf("\t\tconstructed from the training data\n");
    printf("\t-batch-size <int>\n");
    printf("\t\tThe batch size used for mini-batch training;\n");
    printf("\t\tdefault is 11 (i.e., 2 * window + 1)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt\n");
    printf("\t\t-size 200 -window 5 -sample 1e-4 -negative 5\n");
    printf("\t\t-binary 0 -iter 3\n\n");
    return -1;
  }

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;

  int i;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
    hidden_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
    strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
    strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
    strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
    alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
    strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
    window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
    sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-hs", argc, argv)) > 0)
    hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
    iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  //if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
    //batch_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-num-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-senbs", argc, argv)) > 0)
    sentences_in_buffer  = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-tcbs-per-thread", argc, argv)) > 0)
    tcbs_per_thread  = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-items-in-tcb", argc, argv)) > 0)
    items_in_tcb  = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-num-phys", argc, argv)) > 0)
    num_phys  = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
    batch_size  = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-batches-per-thread", argc, argv)) > 0)
    batches_per_thread  = atoi(argv[i + 1]);

  printf("num threads: %d\n", num_threads);
  printf("num physical cores: %d\n", num_phys);
  printf("tcbs per thread: %d\n", tcbs_per_thread);
  printf("items in tcb: %d\n", items_in_tcb);
  printf("hs: %d\n", hs);
  printf("number of iterations: %d\n", iter);
  printf("hidden size: %d\n", hidden_size);
  printf("number of negative samples: %d\n", negative);
  printf("window size: %d\n", window);
  printf("starting learning rate: %.5f\n", alpha);
  printf("starting training using file: %s\n\n", train_file);

  return 1;
}
