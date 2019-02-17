set(HEADERS
  contiguous_buffer/int32_ring_buffer.hpp
  arguments.h
  neuralnet/network.h
  neuralnet/weights.h
  training/data/parsing/sentence_parser.h
  training/data/source/sentence_source.h
  training/data/source/word_sampling.sentence_source.h
  training/data/structure/stdVector.sentence.h
  training/data/structure/contiguous_buffer_backed.word_with_context.h
  training/data/structure/contiguous_word_with_context_buffer.h
  vocabulary/stream_backed.word_source.h
  vocabulary/wordbag/wordbag.h
  vocabulary/wordbag/word2vec.wordbag.h
  vocabulary/wordbag_producer.h
  util.h
  )

set(SOURCES
  arguments.cxx
  neuralnet/network.cxx
  neuralnet/weights.cxx
  training/data/parsing/sentence_parser.cxx
  training/data/source/word_sampling.sentence_source.cxx
  training/data/structure/stdVector.sentence.cxx
  training/data/structure/contiguous_buffer_backed.word_with_context.cxx
  training/data/structure/contiguous_word_with_context_buffer.cxx
  vocabulary/stream_backed.word_source.cxx
  vocabulary/wordbag/word2vec.wordbag.builder.cxx
  vocabulary/wordbag/word2vec.wordbag.cxx
  vocabulary/wordbag_producer.cxx
  util.cxx
  main.cxx
  )

add_executable(wombat ${SOURCES} ${HEADERS})
add_library(wombatLib ${SOURCES})
