// Copyright 2017 Trevor Simonton

#include "timer.h"

std::vector<StopWatch> produceTimers;
std::vector<StopWatch> consumeWaitTime;
std::vector<StopWatch> consumeTime;
std::vector<StopWatch> lockTime;

void InitTimers() {

  for (int i = 0; i < num_threads; ++i) {
    produceTimers.push_back(StopWatch());
    consumeWaitTime.push_back(StopWatch());
    consumeTime.push_back(StopWatch());
    lockTime.push_back(StopWatch());
  }

}

void printTimers() {

  for (int i = 0; i < num_threads; ++i) {
    double p = produceTimers[i].pctRecorded();
    double cw = consumeWaitTime[i].pctRecorded();
    double c = consumeTime[i].pctRecorded();
    double l = lockTime[i].pctRecorded();
    printf("thread[%d]:\tproducing: %.8f\t| consumeWait: %.8f\t| consuming: %.8f\t| lock-get: %.8f\t| recorded:%.8f\n",
      i,
      p,
      cw,
      c,
      l,
      (p+cw+c+l)
      );
  }



  //printf("\033[2J\033[1;1H");
  for (int i = 0; i < num_threads + 1; ++i) {
    fputs("\033[A\033[2K",stdout);
  }
  //rewind(stdout);
  //ftruncate(1,0); 
}
void printFinalTimers() {
  double Ap = 0;
  double Acw = 0;
  double Ac = 0;
  double Al = 0;

  for (int i = 0; i < num_threads; ++i) {
    double p = produceTimers[i].pctRecorded();
    Ap += p;
    double cw = consumeWaitTime[i].pctRecorded();
    Acw += cw;
    double c = consumeTime[i].pctRecorded();
    Ac += c;
    double l = lockTime[i].pctRecorded();
    Al += l;
    printf("thread[%d]:\tproducing: %.8f\t| consumeWait: %.8f\t| consuming: %.8f\t| lock-get: %.8f\t| recorded:%.8f\n",
      i,
      p,
      cw,
      c,
      l,
      (p+cw+c+l)
      );
  }
  printf("Average:\tproducing: %.8f\t| consumeWait: %.8f\t| consuming: %.8f\t| lock-get: %.8f\t| recorded:%.8f\n",
    Ap / num_threads,
    Acw / num_threads,
    Ac / num_threads,
    Al / num_threads,
    (Ap+Acw+Ac+Al) / num_threads
    );
}
