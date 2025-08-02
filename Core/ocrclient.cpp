#include "ocrclient.h"
#include "../Core/common.hpp"
#include <QBuffer>
#include <QString>
#include <QJsonObject>
#include <QNetworkReply>
#include <QJsonArray>
#include <QFile>


OcrClient::OcrClient() {

    manager = new QNetworkAccessManager(this);
    request.setUrl(QUrl("http://localhost:8080/ocr"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
}
OcrClient::~OcrClient() {
    delete manager;
}


void OcrClient::sendOCRRequest(const cv::Mat& image) {
    cv::Mat images = cv::imread("./demo.jpg");
    QImage img = Converter::cvMatToQImage(images);

    QByteArray imagedata;
    QBuffer buffer(&imagedata);
    buffer.open(QIODevice::WriteOnly);

    img.save(&buffer,"JPEG",85);
    QString imagebase64 = imagedata.toBase64();

    // 构造JSON请求
    QJsonObject requestJson;
    requestJson["file"] = imagebase64;
    requestJson["fileType"] = 1;

    QNetworkReply *reply = manager->post(request,QJsonDocument(requestJson).toJson());


    // 处理响应
    connect(reply, &QNetworkReply::finished, [=]() {
        if(reply->error() == QNetworkReply::NoError) {
            if(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200){
                QByteArray response = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(response);
                QJsonObject result = jsonDoc.object()["result"].toObject();

                QJsonArray ocrResults = result["ocrResults"].toArray();
                for(const QJsonValue& res : ocrResults) {
                    QString text = res.toObject()["prunedResult"].toString();
                    qDebug() << "识别结果:" << text;

                    QString imgData = res.toObject()["ocrImage"].toString();
                    QByteArray decoded = QByteArray::fromBase64(imgData.toLatin1());
                    QFile output("result.jpg");
                    if(output.open(QIODevice::WriteOnly)) {
                        output.write(decoded);
                        output.close();
                    }
                }
            } else {
                qDebug() << "状态码非200，失败:" << reply->errorString();
            }
        } else {
            qDebug() << "请求失败:" << reply->errorString();
        }
        reply->deleteLater();
    });
}
