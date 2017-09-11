// Copyright 2017 Trevor Simonton

#ifndef TIMER_H_
#define TIMER_H_

#include "src/common.h"
#include "omp.h"
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include "src/w2v-functions.h"

extern double start;
extern int time_events;

class StopWatch {
public:
  double last, stopped = 0;
  double accumulated = 0;
  StopWatch() {
    last = 0;
  }
  void record() {
    if (last == 0) last = omp_get_wtime();
  }
  void pause() {
    if (last == 0) return;
    accumulated += omp_get_wtime() - last;
    last = 0;
  }
  double timed() {
    return accumulated;
  }
  double running() {
    return stopped - start;
  }
  void stop() {
    stopped = omp_get_wtime();
  }
  float pctRecorded() {
    if (stopped)
      return (float) timed() * 100 / (running() + 1);
    return (float) timed() * 100 / ((omp_get_wtime() - start)+ 1);
  }
};

extern std::vector<StopWatch> produceTimers;
extern std::vector<StopWatch> consumeWaitTime;
extern std::vector<StopWatch> consumeTime;
extern std::vector<StopWatch> lockTime;

void InitTimers();
void printTimers();
void printFinalTimers();


#endif
