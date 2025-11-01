#include "xydocrclient.h"
#include "common.hpp"
#include <QBuffer>
#include <QString>
#include <QJsonObject>
#include <QNetworkReply>
#include <QJsonArray>

xydOcrClient::xydOcrClient(QObject *parent)
    : IOcrService{parent}
{
    format = 0;

    manager = new QNetworkAccessManager(this);
    request.setUrl(QUrl("http://222.66.108.206:4000/xyd_server/OcrGeneralServlet"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    m_systemname = "test";
    privatekey = "11111";
    analysisType = 1;
    m_barcode_len = 13;
    qInfo() << "使用远程ocr";

}

xydOcrClient::~xydOcrClient() {
    delete manager;
    manager = nullptr;
}

void xydOcrClient::sendOCRRequest(const QImage& image, const QString &filepath) {
    // cv::Mat images = cv::imread("./demo.jpg");
    // QImage img = Converter::cvMatToQImage(images);

    QByteArray imagedata;
    QBuffer buffer(&imagedata);
    buffer.open(QIODevice::WriteOnly);

    image.save(&buffer,"JPEG",100);
    QString imagebase64 = imagedata.toBase64();

    // 构造JSON请求
    QString req_time = QDateTime::currentDateTime().toString("yyyyMMddhhmmss");
    QString req_trans_no = m_systemname + req_time;
    QJsonObject headObj;
    headObj["system_name"] = m_systemname;
    headObj["req_time"] = QDateTime::currentDateTime().toString("yyyyMMddhhmmss");; // 假设是 QString
    headObj["req_trans_no"] = req_trans_no; // 假设是 QString

    QString mw_str = QString("system_name%1req_time%2req_trans_no%3%4")
                         .arg(m_systemname)
                         .arg(req_time)
                         .arg(req_trans_no)
                         .arg(privatekey);
    QByteArray hash = QCryptographicHash::hash(mw_str.toUtf8(), QCryptographicHash::Md5);
    QString signature = hash.toHex();
    headObj["signature"] = signature; // 假设是 QString

    // 2. 构建 body
    QJsonObject commonVoObj;
    // 关键：Python的 None 对应 QJsonValue() 或 QJsonValue::Null
    commonVoObj["sceneId"] = QJsonValue();

    QJsonObject bodyObj;
    bodyObj["imgType"] = 1;
    bodyObj["img"] = imagebase64; // 假设返回 QString
    bodyObj["commonVo"] = commonVoObj;
    bodyObj["analysisType"] = analysisType; // 假设是 QString 或 int

    // 3. 组合成顶层 data 对象
    QJsonObject requestJson;
    requestJson["head"] = headObj;
    requestJson["body"] = bodyObj;
/*    QJsonObject requestJson;
    requestJson["file"] = imagebase64;
    requestJson["fileType"] = 1*/;

    QNetworkReply *reply = manager->post(request,QJsonDocument(requestJson).toJson());


    // 处理响应
    auto s = std::chrono::system_clock::now();
    connect(reply, &QNetworkReply::finished, [=]() {
        if(reply->error() == QNetworkReply::NoError) {
            if(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200){
                QByteArray response = reply->readAll();
                QJsonObject jsonDoc = QJsonDocument::fromJson(response).object();
                QJsonObject resultHead = jsonDoc["head"].toObject();
                QJsonObject resultBody = jsonDoc["body"].toObject();
                if (resultHead["error_code"] != "0"){
                    qDebug() << "远端出错";
                    qDebug() << resultHead["error_msg"];
                }


                // emit ocrResReady(text);
                RecognitionResult tmp = extractInfoFromTexts(resultBody);
                qDebug() << "filepath: " << filepath;
                tmp.setImgPath(filepath);
                emit ocrResReady(tmp);
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

RecognitionResult xydOcrClient::extractInfoFromTexts(const QJsonObject& rec_texts)
{
    // // 将QJsonArray转换为QStringList以便处理
    // QStringList texts;
    // for (const QJsonValue& val : rec_texts) {
    //     texts.append(val.toString());
    // }

    // QString joinedText = texts.join(" ");
    // QJsonObject result;

    // // 初始化返回结果
    // result["timestamp"] = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    // result["zip_code"] = QJsonValue::Null;
    // result["barcode"] = QJsonValue::Null;
    // result["address"] = QJsonValue::Null;
    // result["receiver"] = QJsonValue::Null;
    // result["raw_texts"] = rec_texts;
    RecognitionResult result;
    if (analysisType == 1) {
        result = jiexi_tongyongshibie(rec_texts);
    }else {
        result = jiexi_zhinengjiexi(rec_texts);
    }

    InfoComplete::evaluate_ocr_result(result);
    return result;
}

// --- [新] jiexi_zhinengjiexi 方法实现 ---
RecognitionResult xydOcrClient::jiexi_zhinengjiexi(const QJsonObject& result) {
    RecognitionResult info; // 默认为空
    info.setTimeStamp(std::chrono::system_clock::now());
    QString g_text = result["generalResult"].toString("  ");

    if (!result.contains("wordsResult") || !result["wordsResult"].isObject()) {
        return info;
    }
    QJsonObject wordsResult = result["wordsResult"].toObject();

    // 收件人
    std::regex re_suffix(R"((先生收|女士收))");
    QString receiver = wordsResult["userName"].toString();
    info.setReceiver(QString::fromStdString(std::regex_replace(receiver.toStdString(), re_suffix, "")));

    // 邮编 (查找第一个匹配)
    std::regex re_zip(R"(\b\d{6}\b)");
    info.setZipCode(find_first_match(g_text, re_zip));

    // 条形码 (查找第一个匹配及捕获组)
    std::string re_barcode_str = R"(\b信?([a-zA-Z0-9]{)" + std::to_string(m_barcode_len) + R"(})\b)";
    std::regex re_barcode(re_barcode_str);
    std::smatch match;
    std::string g_text_std = g_text.toStdString();
    QString barcode;
    if (std::regex_search(g_text_std, match, re_barcode)) {
        if (match.size() > 1 && match[1].length() > 0) {
            barcode = QString::fromStdString(match.str(1));
        } else {
            QString full_match = QString::fromStdString(match.str(0));
            barcode = full_match.startsWith("信") ? full_match.mid(1) : full_match;
        }
        // 确保长度
        if (barcode.length() > m_barcode_len) {
            info.setBarCode(barcode.left(m_barcode_len));
        }
    }

    // 地址
    QString address = wordsResult["userProvince"].toString() +
                   wordsResult["userCity"].toString() +
                   wordsResult["userArea"].toString() +
                   wordsResult["userAddress"].toString();

    info.setAddress(delete_bar_code(QString::fromStdString(std::regex_replace(address.toStdString(), re_suffix, "")))); // 使用辅助函数

    // 补充收件人
    if (info.getReceiver().isEmpty()) {
        std::regex re_recipient_fallback(R"((\S{2,5})[先生女士]{2}收)");
        if (std::regex_search(g_text_std, match, re_recipient_fallback) && match.size() > 1) {
            info.setReceiver(QString::fromStdString(match.str(1)));
        }
    }
    info.setText(g_text);

    return info;
}

// // --- [新] jiexi_tongyongshibie 方法实现 ---
RecognitionResult xydOcrClient::jiexi_tongyongshibie(const QJsonObject& result) {
    RecognitionResult info;
    info.setTimeStamp(std::chrono::system_clock::now());
    QString text;

    // 对应 @catch_exceptions
    try {
        if (!result.contains("wordsResult") || !result["wordsResult"].isArray()) {
            return info;
        }
        QJsonArray res_list = result["wordsResult"].toArray();
        QString address;

        // 辅助 lambda，用于计算匹配次数
        auto count_matches = [](const std::string& s, const std::regex& r) {
            auto it_begin = std::sregex_iterator(s.begin(), s.end(), r);
            auto it_end = std::sregex_iterator();
            return std::distance(it_begin, it_end);
        };

        // 预编译正则表达式
        std::regex re_zip(R"(\b\d{6}\b)");
        std::string re_barcode_str = R"(\b信?([a-zA-Z0-9]{)" + std::to_string(m_barcode_len) + R"(})\b)";
        std::regex re_barcode(re_barcode_str);
        std::regex re_recipient(R"((\S{2,5})[先生女士]{2}收)");
        std::smatch match;

        for (const QJsonValue& res_val : res_list) {
            if (!res_val.isObject()) continue;
            QJsonObject res = res_val.toObject();
            QString words = res["words"].toString();
            std::string words_std = words.toStdString();

            if (info.getZipCode().isNull() && words.length() == 6 && count_matches(words_std, re_zip) == 1) {
                info.setZipCode(words);
            } else if (info.getBarCode().isNull() &&
                       (words.length() == m_barcode_len || (words.startsWith("信") && words.length() == m_barcode_len + 1)) &&
                       count_matches(words_std, re_barcode) == 1) {

                // 对应 Python: words.strip('信')
                info.setBarCode(words.startsWith("信") ? words.mid(1) : words);

            } else if (info.getReceiver().isNull() && (words.contains("先生收") || words.contains("女士收"))) {
                if (std::regex_search(words_std, match, re_recipient) && match.size() > 1) {
                    info.setReceiver(QString::fromStdString(match.str(1)));
                }
            } else {
                address += words;
            }
        }
        info.setAddress(address);
        text.append(QString::number(info.getTimeStamp()));
        text.append("\n");
        text.append(info.getZipCode());
        text.append("\n");
        text.append(info.getBarCode());
        text.append("\n");
        text.append(info.getAddress());
        text.append("\n");
        text.append(info.getReceiver());
        info.setText(text);

    } catch (const std::exception& e) {
        qWarning() << "Exception in jiexi_tongyongshibie:" << e.what();
        return RecognitionResult{}; // 返回空 struct
    } catch (...) {
        qWarning() << "Unknown exception in jiexi_tongyongshibie.";
        return RecognitionResult{}; // 返回空 struct
    }

    return info;
}
