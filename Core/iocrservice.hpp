#ifndef IOCRSERVICE_HPP
#define IOCRSERVICE_HPP

#include <QObject>
#include "common.hpp"
#include <opencv2/opencv.hpp>

enum class OcrType{
    LocalOcr,
    XydOcr
};

class IOcrService : public QObject
{
    Q_OBJECT
public:
    explicit IOcrService(QObject* parent = nullptr) : QObject(parent) {};
    virtual ~IOcrService() = default;
public slots:
    virtual void sendOCRRequest(const QImage& image, const QString &filepath) = 0;

signals:
    //void ocrResReady(const QString& ocr);
    void ocrResReady(const RecognitionResult& res);
    //void ocrResReady2(const QJsonObject& ocrResult, const QString& imagePath);
    void errorOccur(const QString& error);

private:

};
#endif // IOCRSERVICE_HPP
