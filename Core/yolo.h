#ifndef YOLO_H
#define YOLO_H

#include<iostream>
#include <QObject.h>
#include <QThread>
#include<vector>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "common.hpp"
#include <fstream>

#include <chrono>
#include <cmath>
#include <iostream>
using namespace det;
class YOLO : public QObject{
    Q_OBJECT
public slots:
    void                 pipeline(const QImage& image);
    void                 needOcr();
signals:
    void                 resReady(const QImage& image);
    void                 roiReady(const cv::Mat& image);
public:
    explicit YOLO(const std::string& engine_file_path);
    ~YOLO();
    // 构造方法
    bool                 buildFromOnnx(const std::string& onnx_path);
    bool                 buildFromEngine(const std::string& engine_path);
    bool                 saveEngine(const std::string& engine_path);
    // 推理方法
    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs);
    static void          draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    std::vector<Object>  objs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;

    nvinfer1::IBuilder*          builder = nullptr;
    nvinfer1::INetworkDefinition* network = nullptr;
    nvinfer1::IBuilderConfig*    config = nullptr;
    nvonnxparser::IParser*       parser = nullptr;

    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};

    // 拿取路径文件拓展名用于决定构建方式
    std::string getFileExtension(const std::string& filepath);

    bool mb_NeedOcr = false;
};
#endif // YOLO_H
