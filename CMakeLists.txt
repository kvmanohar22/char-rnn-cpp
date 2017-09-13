cmake_minimum_required(VERSION 2.8)

project(RNN)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11")

include_directories("include")
set(srcs src/rnn_cell.cpp
    src/utils.cpp)

add_library(vanillaRNN SHARED ${srcs})

add_executable(rnn main.cpp)
target_link_libraries(rnn vanillaRNN)