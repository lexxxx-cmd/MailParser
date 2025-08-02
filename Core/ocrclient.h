#ifndef OCRCLIENT_H
#define OCRCLIENT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <QNetworkAccessManager>

class OcrClient : public QObject
{
    Q_OBJECT
public:
    OcrClient();
    ~OcrClient();
public slots:
    void sendOCRRequest(const cv::Mat& image);

signals:
    void ocrResReady();

private:
    void handleResponse();

    QNetworkAccessManager *manager = nullptr;
    QNetworkRequest request;
};

#endif // OCRCLIENT_H
