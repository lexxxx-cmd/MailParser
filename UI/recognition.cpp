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
    initializeCameras(); // 初始化相机池
    initializeOcrs();// 初始化ocr
    //连接保存目录的按钮信号

    // 连接UI面板的切换信号
    //connect(ui->rightConfigPanelWidget, &Control::changeCamRequested, this, &Recognition::onSwitchCamera);
    //ROI
    QObject::connect(ui->rightConfigPanelWidget,&Control::DragROIRequested,ui->leftPanelWidget,&Preview::setRoiSelectionEnabled);
    QObject::connect(ui->leftPanelWidget, &Preview::roiSelected,ui->rightConfigPanelWidget,&Control::setROI);

}

void Recognition::onErrorShow(const QString& error)
{
    QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning: ",error,QMessageBox::Yes);
    if (t_Re == QMessageBox::Yes) return;
}

void Recognition::initializeCameras()
{
    qInfo() << "Initializing cameras...";

    // 初始化MVGigE相机
    // try {
    //     mvCam = new MVGigECamera();
    //     if (mvCam) {
    //         mvCam->moveToThread(&m_mvThread);
    //         QObject::connect(&m_mvThread, &QThread::finished,
    //                 mvCam, &QObject::deleteLater);
    //         m_mvThread.start();
    //         qInfo() << "MVGigE Camera initialized successfully";

    //         connectCameraSignals(mvCam);
    //     } else {
    //         delete mvCam;
    //         mvCam = nullptr;
    //         qWarning() << "MVGigE Camera initialization failed";
    //     }
    // } catch (const std::exception& e) {
    //     qCritical() << "MVGigE Camera initialization exception:" << e.what();
    //     if (mvCam) {
    //         delete mvCam;
    //         mvCam = nullptr;
    //     }
    // }

    try {
        cvCam = new OpenCVCamera();
        if (cvCam) {
            cvCam->moveToThread(&m_cvThread);
            QObject::connect(&m_cvThread, &QThread::finished,
                             cvCam, &QObject::deleteLater);
            m_cvThread.start();
            qInfo() << "opencv Camera initialized successfully";

            connectCameraSignals(cvCam);
        } else {
            delete cvCam;
            cvCam = nullptr;
            qWarning() << "opencv Camera initialization failed";
        }
    } catch (const std::exception& e) {
        qCritical() << "opencv Camera initialization exception:" << e.what();
        if (cvCam) {
            delete cvCam;
            cvCam = nullptr;
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
            QMetaObject::invokeMethod(m_currentCamera, "close",
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
    connect(ui->rightConfigPanelWidget, &Control::captureRequested,
            camera,&ICameraService::startGrabbing);

    connect(ui->rightConfigPanelWidget, &Control::camStopRequested,
            camera, &ICameraService::stopGrabbing);

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
    m_mvThread.quit();
    m_mvThread.wait();

    m_cvThread.quit();
    m_cvThread.wait();

    m_OcrClientThread.quit();
    m_OcrClientThread.wait();

    delete ui;
}
