cmake_minimum_required(VERSION 3.18)
project(OCR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/base64/include
    ${PROJECT_SOURCE_DIR}/cpp-httplib
    ${PROJECT_SOURCE_DIR}/json/include
)

add_executable(ocrClient ocrClient_o.cpp)
