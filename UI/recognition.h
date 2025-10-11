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
    void sendROIRequest(const QImage& ROIimg) {
        emit sendOcrRequest(ROIimg);
    }
signals:
    void sendOcrRequest(const QImage& ROIimg);
private:
    Ui::Recognition *ui;
    // 使用QMap来存储相机池，键是相机类型，值是相机服务实例
    // 使用 std::unique_ptr 来自动管理内存
    std::map<int, std::unique_ptr<ICameraService>> m_cameraPool;
    ICameraService* m_currentCamera = nullptr;
    QTimer* m_switchTimer; // 防抖动
    MVGigECamera* mvCam = nullptr;
    OpenCVCamera* cvCam = nullptr;
    QThread m_mvThread,m_cvThread;

    OcrClient* mp_OcrClient = nullptr;
    QThread m_OcrClientThread,m_XydClientThread;


    void initializeCameras(); // 新增一个私有函数用于初始化所有相机
    void connectCameraSignals(ICameraService* camera);
    void disconnectCameraSignals(ICameraService* camera);
    void stopCurrentCamera();
    void cleanupResources();

    void initializeOcrs(); // 新增一个私有函数用于初始化所有ocr
    void connectOcrSignals(IOcrService* ocr);
    void disconnectOcrSignals(IOcrService* ocr);
};

#endif // RECOGNITION_H
