#include "OpenCVCamera.h"
#include <QDebug>
#include <QThread>
#include <QDateTime>
#include <QDir>
#include <QMutexLocker>
#include <qtimer.h>

OpenCVCamera::OpenCVCamera(int cameraIndex, QObject *parent)
    : ICameraService(parent)
{
    m_cameraIndex = cameraIndex;
    m_type = CameraType::OpenCvCamera;
    m_state = CameraState::Uninitialized;
    initialize();
    qDebug() << "OpenCVCamera构造 - 索引:" << m_cameraIndex
             << "线程:" << QThread::currentThreadId();
}

OpenCVCamera::~OpenCVCamera()
{
    qDebug() << "OpenCVCamera析构 - 线程:" << QThread::currentThreadId();
    release();
}

bool OpenCVCamera::initialize()
{
    qDebug() << "OpenCVCamera::initialize - 线程:" << QThread::currentThreadId();

    if (m_state != CameraState::Uninitialized)
    {
        qWarning() << "相机已初始化，当前状态:" << static_cast<int>(m_state);
        return m_state != CameraState::Error;
    }

    m_capture.open(m_cameraIndex);

    if (!m_capture.isOpened())
    {
        qCritical() << "无法打开相机索引:" << m_cameraIndex;
        setState(CameraState::Error);
        emit errorOccur(QString("无法打开相机设备%1").arg(m_cameraIndex));
        return false;
    }

    // 设置相机参数
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    m_capture.set(cv::CAP_PROP_FPS, 30);

    setState(CameraState::Idle);
    qInfo() << "相机初始化成功 - 索引:" << m_cameraIndex;

    return true;
}

void OpenCVCamera::release()
{
    qDebug() << "OpenCVCamera::release - 线程:" << QThread::currentThreadId();

    if (m_state == CameraState::Grabbing)
    {
        stopGrabbing();
    }

    if (m_capture.isOpened())
    {
        m_capture.release();
        qInfo() << "相机资源已释放";
    }

    m_lastImage = QImage();
    setState(CameraState::Uninitialized);
}

bool OpenCVCamera::startGrabbing()
{
    qDebug() << "OpenCVCamera::startGrabbing - 线程:" << QThread::currentThreadId();

    if (!m_capture.isOpened())
    {
        qWarning() << "相机未初始化";
        emit errorOccur("相机未初始化，无法开始采集");
        return false;
    }

    if (m_state == CameraState::Grabbing)
    {
        qWarning() << "相机已在采集中";
        return true;
    }

    setState(CameraState::Grabbing);

    // 启动连续采集（使用定时器或循环）
    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this, timer]() {
        if (m_state != CameraState::Grabbing)
        {
            timer->stop();
            timer->deleteLater();
            return;
        }

        cv::Mat frame;
        if (!m_capture.read(frame) || frame.empty())
        {
            qWarning() << "读取图像帧失败";
            return;
        }

        m_lastImage = matToQImage(frame);
        if (!m_lastImage.isNull())
        {
            emit imageReady(m_lastImage);
        }
    });

    timer->start(33); // 约30fps

    qInfo() << "相机开始连续采集";
    return true;
}

bool OpenCVCamera::stopGrabbing()
{
    qDebug() << "OpenCVCamera::stopGrabbing - 线程:" << QThread::currentThreadId();

    if (m_state != CameraState::Grabbing)
    {
        qDebug() << "相机未在采集状态";
        return true;
    }

    setState(CameraState::Idle);
    qInfo() << "相机停止采集";

    return true;
}

bool OpenCVCamera::grabOnce()
{
    qDebug() << "OpenCVCamera::grabOnce - 线程:" << QThread::currentThreadId();

    if (!m_capture.isOpened())
    {
        qWarning() << "相机未初始化";
        emit errorOccur("相机未初始化");
        return false;
    }

    if (m_state == CameraState::Grabbing)
    {
        qWarning() << "相机正在连续采集中，无法执行单次抓取";
        return false;
    }

    cv::Mat frame;
    if (!m_capture.read(frame))
    {
        qCritical() << "单次抓取图像失败";
        emit errorOccur("单次抓取图像失败");
        return false;
    }

    if (frame.empty())
    {
        qWarning() << "获取到空图像";
        emit errorOccur("获取到空图像");
        return false;
    }

    m_lastImage = matToQImage(frame);

    if (m_lastImage.isNull())
    {
        qCritical() << "图像转换失败";
        emit errorOccur("图像转换失败");
        return false;
    }

    emit imageReady(m_lastImage);
    qInfo() << "单次抓取成功";

    return true;
}

bool OpenCVCamera::saveImage(const QString& filepath)
{
    qDebug() << "OpenCVCamera::saveImage - 线程:" << QThread::currentThreadId();

    if (m_lastImage.isNull())
    {
        qWarning() << "没有可保存的图像";
        emit errorOccur("没有可保存的图像");
        return false;
    }

    // 确保目录存在
    QFileInfo fileInfo(filepath);
    QDir dir = fileInfo.absoluteDir();
    if (!dir.exists())
    {
        if (!dir.mkpath("."))
        {
            qCritical() << "无法创建目录:" << dir.absolutePath();
            emit errorOccur("无法创建保存目录");
            return false;
        }
    }

    if (!m_lastImage.save(filepath))
    {
        qCritical() << "保存图像失败:" << filepath;
        emit errorOccur(QString("保存图像失败: %1").arg(filepath));
        return false;
    }

    qInfo() << "图像保存成功:" << filepath;
    return true;
}

QImage OpenCVCamera::matToQImage(const cv::Mat& mat)
{
    if (mat.empty())
    {
        return QImage();
    }

    switch (mat.type())
    {
    case CV_8UC1: // 灰度图
        return QImage(mat.data, mat.cols, mat.rows,
                      static_cast<int>(mat.step),
                      QImage::Format_Grayscale8).copy();

    case CV_8UC3: // BGR彩色图
    {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows,
                      static_cast<int>(rgb.step),
                      QImage::Format_RGB888).copy();
    }

    case CV_8UC4: // BGRA
    {
        cv::Mat rgba;
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows,
                      static_cast<int>(rgba.step),
                      QImage::Format_RGBA8888).copy();
    }

    default:
        qWarning() << "不支持的图像格式:" << mat.type();
        return QImage();
    }
}
