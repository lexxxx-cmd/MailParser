#ifndef COMMON_HPP
#define COMMON_HPP
#include "NvInfer.h"
#include "filesystem.hpp"
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <qimage.h>
#include <regex>
#include <QJsonObject>

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

inline bool isChinese(const std::string& str) {
    // 匹配中文字符的Unicode范围
    std::regex chinese_regex("[\u4e00-\u9fa5]");
    return std::regex_search(str, chinese_regex);
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}
// 优化版本：使用面积预计算
class FastNMS {
private:
    // 静态辅助函数，计算两个矩形框的IoU
    static float calculate_iou_fast(int i, int j,
                                    const std::vector<cv::Rect_<float>>& boxes,
                                    const std::vector<float>& areas) {
        const auto& box1 = boxes[i];
        const auto& box2 = boxes[j];

        float x1 = (std::max)(box1.x, box2.x);
        float y1 = (std::max)(box1.y, box2.y);
        float x2 = (std::min)(box1.x + box1.width, box2.x + box2.width);
        float y2 = (std::min)(box1.y + box1.height, box2.y + box2.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0f;
        }

        float intersection = (x2 - x1) * (y2 - y1);
        float union_area = areas[i] + areas[j] - intersection;

        return intersection / union_area;
    }

public:
    // 静态NMS方法
    static std::vector<int> nms(const std::vector<cv::Rect_<float>>& boxes,
                                const std::vector<float>& scores,
                                float score_threshold,
                                float nms_threshold) {

        if (boxes.empty()) {
            return {};
        }

        // 预计算所有框的面积
        std::vector<float> areas(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            areas[i] = boxes[i].width * boxes[i].height;
        }

        // 创建索引并按分数排序
        std::vector<int> indices;
        for (int i = 0; i < boxes.size(); i++) {
            if (scores[i] >= score_threshold) {
                indices.push_back(i);
            }
        }

        std::sort(indices.begin(), indices.end(), [&scores](int i, int j) {
            return scores[i] > scores[j];
        });

        std::vector<bool> suppressed(boxes.size(), false);
        std::vector<int> keep;

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices[i];

            if (suppressed[idx]) {
                continue;
            }

            keep.push_back(idx);

            // 抑制重叠的框
            for (int j = i + 1; j < indices.size(); j++) {
                int jdx = indices[j];

                if (suppressed[jdx]) {
                    continue;
                }

                float iou = calculate_iou_fast(idx, jdx, boxes, areas);
                if (iou > nms_threshold) {
                    suppressed[jdx] = true;
                }
            }
        }

        return keep;
    }
};
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

class RecognitionResult {
private:
    std::chrono::system_clock::time_point _timestamp;
    QString _text, _zipcode, _barcode, _address, _receiver, _grade;
    double _score;
public:
    void setTimeStamp(const std::chrono::system_clock::time_point timestamp) {
        _timestamp = timestamp;
    }

    // 转换为UTC Unix时间戳（毫秒）
    int64_t getTimeStamp() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   _timestamp.time_since_epoch()
                   ).count();
    }

    void setText(const QString& text) {
        _text = text;
    }

    QString getText() const {
        return _text;
    }

    void setZipCode(const QString& zipcode) {
        _zipcode = zipcode;
    }

    QString getZipCode() const {
        return _zipcode;
    }

    void setBarCode(const QString& barcode) {
        _barcode = barcode;
    }

    QString getBarCode() const {
        return _barcode;
    }

    void setAddress(const QString& address) {
        _address = address;
    }

    QString getAddress() const {
        return _address;
    }

    void setReceiver(const QString& receiver) {
        _receiver = receiver;
    }

    QString getReceiver() const {
        return _receiver;
    }

    void setScore(const double& score) {
        _score = score;
    }

    double getScore() const {
        return _score;
    }

    void setGrade(const QString& grade) {
        _grade = grade;
    }

    QString getGrade() const {
        return _grade;
    }
};


class InfoComplete {

    // InfoExtractor() = default;

private:
    static QString get_grade(double score) {
        if (score >= 4.0) return "优秀";
        if (score >= 3.0) return "良好";
        if (score >= 2.0) return "及格";
        return "不及格";
    }
public:
    /**
     * @brief 模拟Python的 evaluate_ocr_result
     * (已更新为 Python 示例中的真实评分逻辑)
     * @param result 包含已提取信息的 QJsonObject
     * @param image_info 图像信息
     * @return 评估结果 QJsonObject
     */
    static void evaluate_ocr_result(RecognitionResult& result) {
        // (void)image_info; // Python 示例中未使用 image_info

        double score;

        // 关键转换点：对应 Python 的 @catch_exceptions 装饰器
        try {
            // --- 对应 InfoComplete.get_value() 的逻辑 ---
            score = 4.0;

            // 关键转换点：.toString() 会将 QJsonValue::Null 转换为空 QString
            // 这与 Python 的 (str(x) if x is not None else '') 逻辑等效
            QString postcode = result.getZipCode();
            QString barcode = result.getBarCode();
            QString receiver = result.getReceiver();
            QString address = result.getAddress();

            // 检查邮编
            if (postcode.length() != 6) {
                score -= 1.0;
            }

            // 检查条码
            if (barcode.isEmpty()) {
                score -= 1.0;
            }

            // 检查收件人
            if (receiver.isEmpty()) {
                score -= 1.0;
            } else if (receiver.length() >= 5) {
                score -= 0.5;
            }

            // 检查地址
            if (address.isEmpty()) {
                score -= 1.0;
            }

        } catch (const std::exception& e) {
            qWarning() << "Scoring system error: " << e.what();
            score = 0.0; // 发生异常时返回默认分数
        } catch (...) {
            qWarning() << "Unknown scoring system error.";
            score = 0.0;
        }

        // --- 组装返回的 QJsonObject ---

        // 获取等级
        QString grade = get_grade(score);

        result.setScore(score);
        result.setGrade(grade);
    }

    /**
     * @brief 将 RecognitionResult (中间状态) 转换为 QJsonObject
     * 用于传递给 evaluate_ocr_result
     */
    // QJsonObject result_to_json(const RecognitionResult& result) {
    //     QJsonObject obj;
    //     obj["timestamp"] = result.timestamp;
    //     obj["zip_code"] = result.zip_code;
    //     obj["barcode"] = result.barcode;
    //     obj["address"] = result.address;
    //     obj["receiver"] = result.receiver;

    //     QJsonArray rawTextsArray;
    //     for(const QString& s : result.raw_texts) {
    //         rawTextsArray.append(s);
    //     }
    //     obj["raw_texts"] = rawTextsArray;
    //     return obj;
    // }
};
#endif // COMMON_HPP
