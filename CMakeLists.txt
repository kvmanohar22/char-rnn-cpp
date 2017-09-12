cmake_minimum_required(VERSION 2.8)

project(RNN)

include_directories("include")
set(srcs src/rnn_cell.cpp)

add_library(vanillaRNN SHARED ${srcs})

add_executable(rnn main.cpp)
target_link_libraries(rnn vanillaRNN)