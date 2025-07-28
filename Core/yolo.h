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
using namespace det;
class YOLO : public QObject{
    Q_OBJECT
public slots:
    void                 pipeline(const QImage& image);
signals:
    void                 resReady(const QImage& image);
public:
    explicit YOLO(const std::string& engine_file_path);
    ~YOLO();
    // 构造方法
    bool                 buildFromOnnx(const std::string& onnx_path);
    bool                 loadFromEngine(const std::string& engine_path);
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
                             const std::vector<Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    std::vector<Object>  objs;
    const std::vector<std::string> CLASS_NAMES = {
                                                  "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
                                                  "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
                                                  "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
                                                  "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
                                                  "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
                                                  "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
                                                  "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
                                                  "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
                                                  "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
                                                  "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
                                                  "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
                                                  "teddy bear",     "hair drier", "toothbrush"};
    const std::vector<std::vector<unsigned int>> COLORS = {
                                                           {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
                                                           {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
                                                           {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
                                                           {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
                                                           {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
                                                           {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
                                                           {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
                                                           {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
                                                           {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
                                                           {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
                                                           {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
                                                           {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
                                                           {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
                                                           {80, 183, 189},  {128, 128, 0}};

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
};
#endif // YOLO_H
