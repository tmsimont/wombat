enable_testing()

include_directories ("${CMAKE_SOURCE_DIR}/src")

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
find_package(GMock REQUIRED)
include_directories(${GMOCK_INCLUDE_DIRS})

set(TESTS
  word2vec.wordbag.cxx
  )

add_executable(test ${TESTS})

target_link_libraries(test GTest::GTest GTest::Main wombatLib)
