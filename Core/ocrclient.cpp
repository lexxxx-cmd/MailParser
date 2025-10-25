#include "ocrclient.h"
#include "../Core/common.hpp"
#include <QBuffer>
#include <QString>
#include <QJsonObject>
#include <QNetworkReply>
#include <QJsonArray>
#include <QFile>


OcrClient::OcrClient(QObject *parent) : IOcrService(parent) {
    format = 0;

    manager = new QNetworkAccessManager(this);
    request.setUrl(QUrl("http://localhost:8080/ocr"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
}
OcrClient::~OcrClient() {
    delete manager;
    manager = nullptr;
}


void OcrClient::sendOCRRequest(const QImage& image) {
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
                    QJsonObject resultJson = extractInfoFromTexts(rec_texts);

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

                    RecognitionResult tmp;
                    tmp.setTimeStamp(std::chrono::system_clock::now());
                    // emit ocrResReady(text);
                    emit ocrResReady(tmp);
                    qDebug() << "识别结果:" << text;

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

/**
 * @brief 从文本行中提取地址、邮编、收件人等信息
 * @param rec_texts 包含多行识别文本的QJsonArray
 * @return 包含提取信息的QJsonObject
 */
QJsonObject OcrClient::extractInfoFromTexts(const QJsonArray& rec_texts)
{
    // 将QJsonArray转换为QStringList以便处理
    QStringList texts;
    for (const QJsonValue& val : rec_texts) {
        texts.append(val.toString());
    }

    QString joinedText = texts.join(" ");
    QJsonObject result;

    // 初始化返回结果
    result["timestamp"] = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    result["zip_code"] = QJsonValue::Null;
    result["barcode"] = QJsonValue::Null;
    result["address"] = QJsonValue::Null;
    result["receiver"] = QJsonValue::Null;
    result["raw_texts"] = rec_texts;

    // 提取邮政编码 - 6位数字
    QRegularExpression zipCodeRegex(R"(\b\d{6}\b)");
    QRegularExpressionMatch zipCodeMatch = zipCodeRegex.match(joinedText);
    if (zipCodeMatch.hasMatch()) {
        result["zip_code"] = zipCodeMatch.captured(0);
    }

    // 提取邮件条码 - 13位字母和数字
    QRegularExpression barcodeRegex(R"(\b[A-Z0-9]{13}\b)");
    QRegularExpressionMatch barcodeMatch = barcodeRegex.match(joinedText);
    if (barcodeMatch.hasMatch()) {
        result["barcode"] = barcodeMatch.captured(0);
    }

    // 提取地址
    QStringList addressLines;
    QString cityLine;
    const QStringList addressKeywords = {"路", "街", "弄", "巷", "号", "室", "区", "市", "省", "县", "大学", "学校", "公司", "大厦", "广场", "理工", "科技", "学院"};

    for (const QString& line : texts) {
        bool containsKeyword = false;
        for (const QString& keyword : addressKeywords) {
            if (line.contains(keyword)) {
                containsKeyword = true;
                break;
            }
        }

        if (containsKeyword && line.length() >= 3) {
            if (!QRegularExpression(R"(^[\d\s\-\(\)]+$)").match(line).hasMatch() &&
                !QRegularExpression(R"(^[A-Z0-9]{10,}$)").match(line).hasMatch()) {
                if (!line.contains("先生") && !line.contains("女士") && !line.contains("小姐") && !line.contains("收") && !line.contains("电话")) {
                    if (QRegularExpression(R"([省市])").match(line).hasMatch() && line.length() <= 5) {
                        cityLine = line;
                    } else {
                        addressLines.append(line);
                    }
                }
            }
        }
    }

    if (addressLines.isEmpty()) {
        for (const QString& line : texts) {
            if (QRegularExpression(R"([省市区县])").match(line).hasMatch() && line.length() > 4) {
                if (!line.contains("先生") && !line.contains("女士") && !line.contains("小姐") && !line.contains("收") && !line.contains("电话")) {
                    addressLines.append(line);
                }
            }
        }
    }

    if (!cityLine.isEmpty()) {
        if (addressLines.isEmpty()) {
            addressLines.append(cityLine);
        } else if (!addressLines.contains(cityLine)) {
            addressLines.prepend(cityLine);
        }
    }

    if (!addressLines.isEmpty()) {
        result["address"] = addressLines.join(" ");
    }

    // 提取收件人
    QString receiverLine;
    for (const QString& line : texts) {
        if (line.contains("收") || line.contains("先生") || line.contains("女士") || line.contains("小姐")) {
            receiverLine = line;
            break;
        }
    }

    if (receiverLine.isEmpty()) {
        for (int i = 0; i < texts.size(); ++i) {
            if ((texts[i].contains("亲启") || texts[i].contains("敬启")) && i > 0) {
                const QString& prevLine = texts[i - 1];
                if (QRegularExpression(R"(^[\u4e00-\u9fa5]{2,5}$)").match(prevLine).hasMatch()) {
                    receiverLine = prevLine;
                    break;
                }
            }
        }
    }

    if (receiverLine.isEmpty()) {
        const QStringList addressStopWords = {"路", "街", "弄", "巷", "号", "室", "省", "市", "区", "县"};
        for (const QString& line : texts) {
            if (QRegularExpression(R"(^[\u4e00-\u9fa5]{2,5}$)").match(line).hasMatch()) {
                bool isAddress = false;
                for (const QString& keyword : addressStopWords) {
                    if (line.contains(keyword)) {
                        isAddress = true;
                        break;
                    }
                }
                if (!isAddress) {
                    receiverLine = line;
                    break;
                }
            }
        }
    }

    // if (!receiverLine.isEmpty()) {
    //     result["receiver"] = cleanReceiverName(receiverLine);
    // }

    // // 添加图像信息并进行评分
    // QJsonObject imageInfo;
    // imageInfo["width"] = 640;
    // imageInfo["height"] = 480;
    // imageInfo["format"] = "jpg";
    // imageInfo["file_size"] = 0;

    // result["evaluation"] = evaluateOcrResult(result, imageInfo);

    return result;
}
