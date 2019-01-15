set(HEADERS
  arguments.h
  neuralnet/network.h
  neuralnet/weights.h
  vocabulary/dictionary.h
  vocabulary/word2vec.dictionary.h
  )

set(SOURCES
  arguments.cxx
  neuralnet/network.cxx
  neuralnet/weights.cxx
  vocabulary/dictionary.cxx
  vocabulary/word2vec.dictionary.cxx
  main.cxx
  )

add_executable(wombat ${SOURCES} ${HEADERS})
add_library(wombatLib ${SOURCES})
