#ifndef OCRCLIENT_H
#define OCRCLIENT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <QNetworkAccessManager>
#include "common.hpp"

class OcrClient : public QObject
{
    Q_OBJECT
public:
    OcrClient();
    ~OcrClient();
public slots:
    void sendOCRRequest(const cv::Mat& image);

signals:
    void ocrResReady(const QString& ocr);
    void errorOccur(const QString& error);

private:
    void handleResponse();

    QNetworkAccessManager *manager = nullptr;
    QNetworkRequest request;

    size_t format;
};

#endif // OCRCLIENT_H
