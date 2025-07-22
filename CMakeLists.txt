cmake_minimum_required(VERSION 3.16)

project(MailParser VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools)
# 设置OpenCV路径
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/Thridparty/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)


# 设置 TensorRT 安装路径
set(TENSORRT_ROOT "E:/TensorRT/TensorRT-10.12.0.36" CACHE PATH "TensorRT installation directory")

# 设置 CUDA 安装路径（如果使用 CUDA）
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" CACHE PATH "CUDA installation directory")
list(APPEND ALL_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_ROOT}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/include
)
list(APPEND ALL_LIBS
        ${OpenCV_LIBRARIES}
        nvinfer_10.lib
        nvinfer_plugin_10.lib
        nvinfer_vc_plugin_10.lib
        nvonnxparser_10.lib
)

set(TS_FILES MailParser_en_AS.ts)
set(UI_SOURCES
        UI/mainwindow.cpp
        UI/mainwindow.h
        UI/mainwindow.ui
)
set(PROJECT_SOURCES
        main.cpp
        ${UI_SOURCES}
        ${TS_FILES}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    find_package(Qt6 REQUIRED COMPONENTS Core Widgets Quick)
find_package(Qt6 REQUIRED COMPONENTS Core)

    qt_add_executable(MailParser
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        Core/mvcamera.h Core/mvcamera.cpp
        Core/yolo.h Core/yolo.cpp
        Core/common.hpp
        Core/filesystem.hpp




    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET MailParser APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation

    qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
else()
    if(ANDROID)
        add_library(MailParser SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(MailParser
            ${PROJECT_SOURCES}
        )
    endif()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
endif()
include_directories(${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/include
                    ${ALL_INCLUDE_DIRS})
target_link_directories(
                        MailParser
                        PUBLIC
                        ${TENSORRT_ROOT}/lib
                        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
                )
target_link_libraries(MailParser PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    # microvision
    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVCamProptySheet.lib
    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVGigE.lib
    ${CMAKE_SOURCE_DIR}/Thridparty/SDK_Mircovision/lib/MVTickDlg.lib
    ${ALL_LIBS}
)
target_link_libraries(MailParser PRIVATE Qt6::Core Qt6::Widgets Qt6::Quick)
target_link_libraries(MailParser PRIVATE Qt6::Core)


# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
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

include(GNUInstallDirs)
install(TARGETS MailParser
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MailParser)
endif()
