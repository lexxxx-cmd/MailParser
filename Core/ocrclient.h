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
    void sendOCRRequest(const QImage& image, const QString &filepath) override;

private:
    QJsonObject extractInfoFromTexts(const QJsonArray& rec_texts);
    QString cleanReceiverName(const QString& raw_name) {
        // 模拟实现：移除称谓
        // 关键转换点：在 C++ 中，正则表达式字符串需要转义 `\`
        std::regex re(R"((先生|女士|小姐|收|电话).*)");

        // 性能优化：直接在 QString 上使用 QRegularExpression 效率更高
        // 但为保持与Python re的兼容性，我们继续使用 std::regex
        std::string std_name = raw_name.toStdString();
        std::string cleaned_name = std::regex_replace(std_name, re, "");
        return QString::fromStdString(cleaned_name);
    }
    QNetworkAccessManager *manager = nullptr;
    QNetworkRequest request;

    size_t format;
};

#endif // OCRCLIENT_H
