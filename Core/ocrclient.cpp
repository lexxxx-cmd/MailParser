#include "ocrclient.h"
#include "../Core/common.hpp"
#include <QBuffer>
#include <QString>
#include <QJsonObject>
#include <QNetworkReply>
#include <QJsonArray>
#include <QFile>


OcrClient::OcrClient() {
    format = 0;

    manager = new QNetworkAccessManager(this);
    request.setUrl(QUrl("http://localhost:8080/ocr"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
}
OcrClient::~OcrClient() {
    delete manager;
}


void OcrClient::sendOCRRequest(const cv::Mat& image) {
    // cv::Mat images = cv::imread("./demo.jpg");
    QImage img = Converter::cvMatToQImage(image);

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
    auto s = std::chrono::system_clock::now();
    connect(reply, &QNetworkReply::finished, [=]() {
        if(reply->error() == QNetworkReply::NoError) {
            if(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200){
                QByteArray response = reply->readAll();
                QJsonDocument jsonDoc = QJsonDocument::fromJson(response);
                QJsonObject result = jsonDoc.object()["result"].toObject();

                QJsonArray ocrResults = result["ocrResults"].toArray();
                for(const QJsonValue& res : ocrResults) {
                    QJsonObject pruneRes = res.toObject()["prunedResult"].toObject();
                    // 只有[1]多边形顶点坐标和[6]识别文字有用
                    QJsonArray dt_polys = pruneRes["dt_ploys"].toArray();
                    QJsonArray rec_texts = pruneRes["rec_texts"].toArray();
                    QString text;
                    for (auto it = rec_texts.begin(); it != rec_texts.end(); it++) {
                        QString tmp = (*it).toString();
                        text += tmp;

                        if (std::isalnum(static_cast<unsigned char>(tmp.toStdString().back()))) {
                            text += "\n";
                        } else {
                            text += " ";
                        }
                    }
                    emit ocrResReady(text);
                    qDebug() << "识别结果:" << text;
                    /*
                    QString imgData = res.toObject()["ocrImage"].toString();
                    QByteArray decoded = QByteArray::fromBase64(imgData.toLatin1());
                    format++;
                    QString m = "E:/Qt/repos/MailParser/ocrRes/result" + QString::number(format) + ".jpg";
                    QFile output(m);
                    if(output.open(QIODevice::WriteOnly)) {
                        output.write(decoded);
                        output.close();
                    }
                    */
                }
            } else {
                emit errorOccur("状态码非200，客户端链接失败!");
                qDebug() << "状态码非200，失败:" << reply->errorString();
            }
        } else {
            emit errorOccur("请求失败,服务端未开启!");
            qDebug() << "请求失败,服务端未开启:" << reply->errorString();
        }
        reply->deleteLater();
    });
    auto e = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / 1000.;
    qDebug() << "time: " << tc;
}
