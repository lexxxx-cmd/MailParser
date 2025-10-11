#ifndef OCRCLIENT_H
#define OCRCLIENT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <QNetworkAccessManager>
#include "common.hpp"
#include "iocrservice.hpp"

class OcrClient : public IOcrService
{
    Q_OBJECT
public:
    OcrClient(QObject *parent = nullptr);
    ~OcrClient() override;
public slots:
    void sendOCRRequest(const QImage& image) override;

private:
    void handleResponse();

    QNetworkAccessManager *manager = nullptr;
    QNetworkRequest request;

    size_t format;
};

#endif // OCRCLIENT_H
