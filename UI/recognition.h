#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <QWidget>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QMutexLocker>
#include <map>
#include <memory>
#include "../Core/icameraservice.hpp"
#include "../Core/mvgigecamera.h"
#include "../Core/opencvcamera.h"
#include "../Core/iocrservice.hpp"
#include "../Core/ocrclient.h"


namespace Ui {
class Recognition;
}



class Recognition : public QWidget
{
    Q_OBJECT

public:
    explicit Recognition(QWidget *parent = nullptr);
    void onErrorShow(const QString& error);
    void onCameraStateChanged(CameraState state);
    ~Recognition();
public slots:
    void sendROIRequest(const QImage& ROIimg, const QString &filepath) {
        emit sendOcrRequest(ROIimg, filepath);
    }
signals:
    void sendOcrRequest(const QImage& ROIimg, const QString &filepath);
private:
    Ui::Recognition *ui;
    // 使用QMap来存储相机池，键是相机类型，值是相机服务实例
    // 使用 std::unique_ptr 来自动管理内存
    ICameraService* m_currentCamera = nullptr;
    QTimer* m_switchTimer; // 防抖动
    QThread m_camThread;

    IOcrService* mp_OcrClient = nullptr;
    IOcrService* mp_XydClient = nullptr;
    QThread m_OcrClientThread,m_XydClientThread;


    void initializeCameras(CameraType curCamera); // 新增一个私有函数用于初始化所有相机
    void connectCameraSignals(ICameraService* camera);
    void disconnectCameraSignals(ICameraService* camera);
    void stopCurrentCamera();
    void stopCurrentOcr();
    void cleanupResources();

    void initializeOcrs(OcrType curOcr); // 新增一个私有函数用于初始化所有ocr
    void connectOcrSignals(IOcrService* ocr);
    void disconnectOcrSignals(IOcrService* ocr);
};

#endif // RECOGNITION_H
