#ifndef XYDOCRCLIENT_H
#define XYDOCRCLIENT_H

#include <QObject>
#include "common.hpp"
#include "iocrservice.hpp"
#include <QNetworkAccessManager>

class xydOcrClient : public IOcrService
{
    Q_OBJECT
public:
    explicit xydOcrClient(QObject *parent = nullptr, RemoteConfig config = RemoteConfig());
    ~xydOcrClient() override;
public slots:
    void sendOCRRequest(const QImage& image, const QString &filepath) override;

private:

    RecognitionResult extractInfoFromTexts(const QJsonObject& rec_texts);
    RecognitionResult jiexi_tongyongshibie(const QJsonObject& result);
    RecognitionResult jiexi_zhinengjiexi(const QJsonObject& result);
    QString find_first_match(const QString& text, const std::regex& re) {
        // 关键转换点：std::regex 需要 std::string (UTF-8)
        std::string std_text = text.toStdString();
        std::smatch match;
        if (std::regex_search(std_text, match, re)) {
            // 关键转换点：从 std::string (UTF-8) 转回 QString
            return QString::fromStdString(match.str(0));
        }
        return QString(); // 返回 null QString
    }
    QString delete_bar_code(QString address) {
        std::string re_barcode_str = R"(\b信?([a-zA-Z0-9]{)" + std::to_string(m_barcode_len) + R"(})\b)";
        std::regex re_barcode(re_barcode_str);
        // 替换为空字符串
        return QString::fromStdString(std::regex_replace(address.toStdString(), re_barcode, ""));
    }
    QNetworkAccessManager *manager = nullptr;
    QNetworkRequest request;

    size_t format;
    QString m_systemname, privatekey;
    int analysisType, m_barcode_len;

};

#endif // XYDOCRCLIENT_H
