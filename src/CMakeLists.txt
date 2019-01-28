set(HEADERS
  contiguous_buffer/int32_ring_buffer.hpp
  arguments.h
  neuralnet/network.h
  neuralnet/weights.h
  vocabulary/wordbag.h
  vocabulary/word2vec.wordbag.h
  training/data/stdVector.sentence.h
  training/data/contiguous_buffer_backed.word_with_context.hpp
  training/data/contiguous_word_with_context_buffer.hpp
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
