enable_testing()

include_directories ("${CMAKE_SOURCE_DIR}/src")

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
find_package(GMock REQUIRED)
include_directories(${GMOCK_INCLUDE_DIRS})

set(TESTS
  contiguous_buffer/int32_ring_buffer.cxx
  word2vec.wordbag.cxx
  training/data/structure/stdVector.sentence.cxx
  training/data/parsing/sentence_parser.cxx
  training/data/structure/contiguous_word_with_context_buffer.cxx
  )

add_executable(test ${TESTS})

target_link_libraries(test GTest::GTest GTest::Main wombatLib)
