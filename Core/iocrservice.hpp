#ifndef IOCRSERVICE_HPP
#define IOCRSERVICE_HPP

#include <QObject>
#include <opencv2/opencv.hpp>
class IOcrService : public QObject
{
    Q_OBJECT
public:
    explicit IOcrService(QObject* parent = nullptr) : QObject(parent) {};
    virtual ~IOcrService() = default;
public slots:
    virtual void sendOCRRequest(const QImage& image) = 0;

signals:
    void ocrResReady(const QString& ocr);
    void errorOccur(const QString& error);

private:

};
#endif // IOCRSERVICE_HPP
