set(HEADERS
  arguments.h
  neuralnet/network.h
  neuralnet/weights.h
  vocabulary/wordbag.h
  vocabulary/word2vec.wordbag.h
  training/data/stdVector.sentence.h
  )

set(SOURCES
  arguments.cxx
  neuralnet/network.cxx
  neuralnet/weights.cxx
  vocabulary/word2vec.wordbag.builder.cxx
  vocabulary/word2vec.wordbag.cxx
  training/data/stdVector.sentence.cxx
  main.cxx
  )

add_executable(wombat ${SOURCES} ${HEADERS})
add_library(wombatLib ${SOURCES})
