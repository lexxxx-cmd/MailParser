#include "recognition.h"
#include "ui_recognition.h"

#include <QMessageBox>

Recognition::Recognition(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Recognition)
    , m_switchTimer(new QTimer(this))
{
    ui->setupUi(this);
    m_switchTimer->setSingleShot(true);
    m_switchTimer->setInterval(100); // 100ms防抖动

    connect(ui->rightConfigPanelWidget, &Control::captureRequested,this, [=](CameraType curCamera){
        initializeCameras(curCamera);
        m_currentCamera->startGrabbing();
    });
    connect(ui->rightConfigPanelWidget, &Control::camStopRequested,this, [=](){
        stopCurrentCamera();
    });
    initializeOcrs();// 初始化ocr
    //连接保存目录的按钮信号
    //ROI
    QObject::connect(ui->rightConfigPanelWidget,&Control::DragROIRequested,ui->leftPanelWidget,&Preview::setRoiSelectionEnabled);
    QObject::connect(ui->leftPanelWidget, &Preview::roiSelected,ui->rightConfigPanelWidget,&Control::setROI);

}

void Recognition::onErrorShow(const QString& error)
{
    QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning: ",error,QMessageBox::Yes);
    if (t_Re == QMessageBox::Yes) return;
}

void Recognition::initializeCameras(CameraType curCamera)
{
    qInfo() << "Initializing cameras...";
    if(curCamera == CameraType::MvGigeCamera) {
        m_currentCamera = new MVGigECamera();
    }else {
        m_currentCamera = new OpenCVCamera();
    }
    // 初始化MVGigE相机
    try {
        if (m_currentCamera) {
            m_currentCamera->moveToThread(&m_camThread);
            QObject::connect(&m_camThread, &QThread::finished,
                    m_currentCamera, &QObject::deleteLater);
            m_camThread.start();
            qInfo() << (int)curCamera << " Camera initialized successfully";

            connectCameraSignals(m_currentCamera);
        } else {
            delete m_currentCamera;
            m_currentCamera = nullptr;
            qWarning() << (int)curCamera << "Camera initialization failed";
        }
    } catch (const std::exception& e) {
        qCritical() << (int)curCamera << "Camera initialization exception:" << e.what();
        if (m_currentCamera) {
            delete m_currentCamera;
            m_currentCamera = nullptr;
        }
    }
}

void Recognition::initializeOcrs()
{
    qInfo() << "Initializing ocrs...";

    try {
        mp_OcrClient = new OcrClient();
        if (mp_OcrClient) {
            mp_OcrClient->moveToThread(&m_OcrClientThread);
            QObject::connect(&m_OcrClientThread, &QThread::finished,
                             mp_OcrClient, &QObject::deleteLater);
            m_OcrClientThread.start();
            qInfo() << "local ocr initialized successfully";

            connectOcrSignals(mp_OcrClient);
        } else {
            delete mp_OcrClient;
            mp_OcrClient = nullptr;
            qWarning() << "local ocr initialization failed";
        }
    } catch (const std::exception& e) {
        qCritical() << "local ocr initialization exception:" << e.what();
        if (mp_OcrClient) {
            delete mp_OcrClient;
            mp_OcrClient = nullptr;
        }
    }
}

void Recognition::stopCurrentCamera()
{
    if (m_currentCamera) {
        qDebug() << "Stopping current camera...";

        // 断开信号连接
        disconnectCameraSignals(m_currentCamera);

        // 停止采集（使用Qt::QueuedConnection确保线程安全）
        if (m_currentCamera->isGrabbing()) {
            QMetaObject::invokeMethod(m_currentCamera, "stopGrabbing",
                                      Qt::QueuedConnection);
        }

        // 关闭相机
        if (m_currentCamera->getState() != CameraState::Idle) {
            QMetaObject::invokeMethod(m_currentCamera, "release",
                                      Qt::QueuedConnection);
        }

        m_currentCamera = nullptr;
    }
}

void Recognition::connectCameraSignals(ICameraService* camera)
{
    if (!camera) return;

    // 连接图像信号
    connect(camera, &ICameraService::imageReady,
            ui->leftPanelWidget, &Preview::updateImage,
            Qt::QueuedConnection);

    // 连接错误信号
    connect(camera, &ICameraService::errorOccur,
            this, &Recognition::onErrorShow,
            Qt::QueuedConnection);

    // 连接状态变化信号

    // 连接控制信号

    connect(ui->rightConfigPanelWidget,&Control::ApplyROIRequested,camera,&ICameraService::setROI);

    connect(ui->rightConfigPanelWidget,&Control::ClearROIRequested,camera,&ICameraService::resetROI);

    connect(ui->rightConfigPanelWidget,&Control::ShowROIRequested,camera,&ICameraService::showROI);

    connect(camera,&ICameraService::roiImageReady,this,&Recognition::sendROIRequest);
    connect(this,&Recognition::sendOcrRequest,ui->bottomResultPanelWidget,&Result::updateImage);

    qDebug() << "Camera signals connected";
}

void Recognition::disconnectCameraSignals(ICameraService* camera)
{
    if (!camera) return;

    // 断开所有信号连接
    disconnect(camera, nullptr, nullptr, nullptr);
    disconnect(ui->rightConfigPanelWidget, nullptr, camera, nullptr);

    qDebug() << "Camera signals disconnected";
}

void Recognition::connectOcrSignals(IOcrService* ocr)
{
    if (!ocr) return;

    connect(ocr,&IOcrService::ocrResReady,ui->bottomResultPanelWidget,&Result::onOcrshow);
    connect(ocr,&IOcrService::errorOccur,this, &Recognition::onErrorShow);
    connect(this,&Recognition::sendOcrRequest,ocr,&IOcrService::sendOCRRequest);

    qDebug() << "ocr signals connected";
}

void Recognition::disconnectOcrSignals(IOcrService* ocr)
{
    if (!ocr) return;

    // 断开所有信号连接
    disconnect(ocr, nullptr, nullptr, nullptr);
    // disconnect(ui->rightConfigPanelWidget, nullptr, ocr, nullptr);

    qDebug() << "ocr signals disconnected";
}

Recognition::~Recognition()
{
    m_camThread.quit();
    m_camThread.wait();

    m_OcrClientThread.quit();
    m_OcrClientThread.wait();

    delete ui;
}
