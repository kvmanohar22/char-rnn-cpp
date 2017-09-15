cmake_minimum_required(VERSION 2.8)

project(RNN)
include_directories("include")

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O2")

set(srcs src/rnn_cell.cpp
    src/utils.cpp
    src/softmax_layer.cpp)

add_library(vanillaRNN SHARED ${srcs})

add_executable(rnn main.cpp)
target_link_libraries(rnn vanillaRNN)