set(HEADERS
  contiguous_buffer/int32_ring_buffer.hpp
  arguments.h
  neuralnet/network.h
  neuralnet/weights.h
  vocabulary/wordbag.h
  vocabulary/word2vec.wordbag.h
  training/data/parsing/sentence_parser.h
  training/data/source/sentence_producer.h
  training/data/source/file_backed.sentence_producer.h
  training/data/structure/stdVector.sentence.h
  training/data/structure/contiguous_buffer_backed.word_with_context.h
  training/data/structure/contiguous_word_with_context_buffer.h
  )

set(SOURCES
  arguments.cxx
  neuralnet/network.cxx
  neuralnet/weights.cxx
  vocabulary/word2vec.wordbag.builder.cxx
  vocabulary/word2vec.wordbag.cxx
  training/data/parsing/sentence_parser.cxx
  training/data/source/file_backed.sentence_producer.cxx
  training/data/structure/stdVector.sentence.cxx
  training/data/structure/contiguous_buffer_backed.word_with_context.cxx
  training/data/structure/contiguous_word_with_context_buffer.cxx
  main.cxx
  )

add_executable(wombat ${SOURCES} ${HEADERS})
add_library(wombatLib ${SOURCES})
