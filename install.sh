# git submodule init
# git submodule update
# cd googletest
# mkdir build
# cd build
# cmake ../
# cd ..
# cmake --build build
# cd ..
cmake -H. -Bbuild \
  -DGTEST_LIBRARY="googletest/build/googlemock/gtest/libgtest.a" \
  -DGTEST_MAIN_LIBRARY="googletest/build/googlemock/gtest/libgtest_main.a" \
  -DGTEST_INCLUDE_DIR="googletest/googletest/include/" \
  -DGMOCK_LIBRARY="googletest/build/googlemock/libgmock.a" \
  -DGMOCK_MAIN_LIBRARY="googletest/build/googlemock/libgmock_main.a" \
  -DGMOCK_INCLUDE_DIR="googletest/googlemock/include/"
cmake --build build
