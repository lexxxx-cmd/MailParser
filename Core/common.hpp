#ifndef COMMON_HPP
#define COMMON_HPP
#include "NvInfer.h"
#include "filesystem.hpp"
#include "opencv2/opencv.hpp"
#include <qimage.h>

#define CHECK(call)                                                                                                    \
do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
    }                                                                                                              \
} while (0)

    class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};
    class Converter {
public:
        static cv::Mat QImage2Mat(const QImage& image) {
            // 空图像检查
            if (image.isNull()) return cv::Mat();


            // 判断是否为灰度图像
            if (image.isGrayscale()) {
                // 灰度图处理：统一转为Grayscale8格式
                QImage grayImage = image.convertToFormat(QImage::Format_Grayscale8);
                cv::Mat result(grayImage.height(),
                               grayImage.width(),
                               CV_8UC1,
                               (void*)grayImage.constBits(),
                               grayImage.bytesPerLine());

                return result.clone(); // 克隆确保数据安全
            } else {
                // 彩色图处理：统一转为RGB888格式
                QImage colorImage = image.convertToFormat(QImage::Format_RGB888);
                cv::Mat result(colorImage.height(),
                               colorImage.width(),
                               CV_8UC3,
                               (void*)colorImage.constBits(),
                               colorImage.bytesPerLine());
                return result.clone(); // 克隆确保数据安全
            }

        }

        static QImage cvMatToQImage(const cv::Mat &mat)
        {
            // 空矩阵检查
            if (mat.empty()) return QImage();


            // 确保数据类型为8位
            cv::Mat processedMat;
            if (mat.type() != CV_8UC1 && mat.type() != CV_8UC3 && mat.type() != CV_8UC4) {
                // 非标准格式，转换为8位
                mat.convertTo(processedMat, CV_8U);

            } else {
                processedMat = mat;
            }

            switch (processedMat.channels()) {
            case 1:
            {
                // 单通道：直接转为灰度QImage
                QImage result(processedMat.data,
                              processedMat.cols,
                              processedMat.rows,
                              processedMat.step,
                              QImage::Format_Grayscale8);


                return result.copy(); // 复制确保数据独立
            }
            case 3:
            {
                // 三通道：直接转为RGB QImage（不做颜色空间转换）
                QImage result(processedMat.data,
                              processedMat.cols,
                              processedMat.rows,
                              processedMat.step,
                              QImage::Format_RGB888);


                return result.copy(); // 复制确保数据独立
            }
            case 4:
            {
                // 四通道：去除Alpha通道，转为3通道处理
                cv::Mat rgb;
                cv::cvtColor(processedMat, rgb, cv::COLOR_BGRA2RGB);

                QImage result(rgb.data,
                              rgb.cols,
                              rgb.rows,
                              rgb.step,
                              QImage::Format_RGB888);


                return result.copy();
            }
            default:
            {
                // 异常通道数：强制转为灰度图
                cv::Mat gray;
                if (processedMat.channels() > 1) {
                    cv::cvtColor(processedMat, gray, cv::COLOR_BGR2GRAY);
                } else {
                    gray = processedMat;
                }

                QImage result(gray.data,
                              gray.cols,
                              gray.rows,
                              gray.step,
                              QImage::Format_Grayscale8);


                return result.copy();
            }
            }
        }
    };
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

namespace det {
struct Binding {
    size_t         size  = 1;
    size_t         dsize = 1;
    nvinfer1::Dims dims;
    std::string    name;
};

struct Object {
    cv::Rect_<float> rect;
    int              label = 0;
    float            prob  = 0.0;
};

struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};
}  // namespace det
#endif // COMMON_HPP
