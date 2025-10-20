cmake_minimum_required(VERSION 3.22)
project(llama_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(llama_jni SHARED
        llama_jni.cpp
)

target_include_directories(llama_jni PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
)

add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/llama_build
)

target_link_libraries(llama_jni PRIVATE llama)

target_link_libraries(llama_jni PRIVATE log)
