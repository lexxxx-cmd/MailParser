cmake_minimum_required(VERSION 3.18)

project(MailParser VERSION 0.1 LANGUAGES CXX CUDA)

# Qt basic setup
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt packages
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools)
# 加网络
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Network)

# Set paths (configurable from command line)
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/Thridparty/OpenCV/opencv/build" CACHE PATH "OpenCV installation directory")
set(TENSORRT_ROOT "E:/TensorRT/TensorRT-10.12.0.36" CACHE PATH "TensorRT installation directory")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" CACHE PATH "CUDA installation directory")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
add_definitions(-DTRT_10)

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")

# Include directories
include_directories(
    #${PROJECT_SOURCE_DIR}/YOLO_src/
    #${PROJECT_SOURCE_DIR}/YOLO_plugin/
    ${PROJECT_SOURCE_DIR}/Thridparty/SDK_Mircovision/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
#file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/YOLO_src/*.cpp ${PROJECT_SOURCE_DIR}/YOLO_src/*.cu)
#file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/YOLO_plugin/*.cu)
file(GLOB_RECURSE UI_SOURCES UI/*.cpp UI/*.h UI/*.ui)

# Plugin library

#add_library(myplugins SHARED ${PLUGIN_SRCS})
#target_link_libraries(myplugins
#    ${TENSORRT_ROOT}/lib/nvinfer_10.lib
#    ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
#)

# Translation files
set(TS_FILES MailParser_en_AS.ts)
qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})

# Main executable
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    find_package(Qt6 REQUIRED COMPONENTS Core)

    qt_add_executable(MailParser
        MANUAL_FINALIZATION
        main.cpp
        ${UI_SOURCES}
 #       ${SRCS}
        Core/mvcamera.h Core/mvcamera.cpp
        Core/yolo.h Core/yolo.cpp
        Core/common.hpp
        Core/filesystem.hpp
        Core/ocrclient.cpp
    )
else()
    add_executable(MailParser
        main.cpp
        ${UI_SOURCES}
#        ${SRCS}
    )
endif()

# Link libraries
target_link_libraries(MailParser PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    ${OpenCV_LIBS}

    ${TENSORRT_ROOT}/lib/nvinfer_10.lib
    ${TENSORRT_ROOT}/lib/nvinfer_plugin_10.lib
    #${TENSORRT_ROOT}/lib/nvinfer_dispatch_10.lib
    #${TENSORRT_ROOT}/lib/nvinfer_lean_10.lib
    #${TENSORRT_ROOT}/lib/nvinfer_vc_plugin_10.lib
    ${TENSORRT_ROOT}/lib/nvonnxparser_10.lib

    ${CUDAToolkit_LIBRARY_DIR}/cudart.lib

    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVCamProptySheet.lib
    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVGigE.lib
    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVTickDlg.lib
#    myplugins
)
target_link_libraries(MailParser PRIVATE Qt6::Core Qt6::Network)

# Link directories
target_link_directories(MailParser PRIVATE
    ${TENSORRT_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    ${CUDAToolkit_LIBRARY_DIR}
)

# macOS/iOS specific settings
if(${QT_VERSION} VERSION_LESS 6.1.0)
    set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.MailParser)
endif()

set_target_properties(MailParser PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# Installation
install(TARGETS MailParser
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MailParser)
endif()
