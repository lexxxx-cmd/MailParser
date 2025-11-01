#ifndef OPENCVCAMERASERVICE_H
#define OPENCVCAMERASERVICE_H

#include "icameraservice.hpp"
#include <opencv2/opencv.hpp>
#include <QThread> // 包含QThread头文件，可能会用到msleep

/**
 * @class OpenCVCamera
 * @brief 使用OpenCV库实现ICamera接口的工作者类。
 *
 * 这个类被设计为运行在一个单独的线程中(通过moveToThread)。
 * 它的所有耗时操作（如打开、关闭、抓取循环）都在它所在的线程中执行，
 * 不会阻塞主UI线程。
 */
class OpenCVCamera : public ICameraService
{
    Q_OBJECT

public:
    explicit OpenCVCamera(int cameraIndex = 0, QObject *parent = nullptr);
    ~OpenCVCamera() override;

    // --- ICameraService接口的实现，现在作为槽函数 ---
public slots:
    // 实现基类接口
    bool initialize() override;
    void release() override;
    bool startGrabbing() override;
    bool stopGrabbing() override;
    bool grabOnce() override;
    bool saveImage() override;

private:
    /**
     * @brief 将cv::Mat转换为QImage
     */
    QImage matToQImage(const cv::Mat& mat);

private:
    cv::VideoCapture m_capture;      // OpenCV视频捕获对象
    int m_cameraIndex;               // 相机设备索引

    // m_bRun标志现在用于控制抓取循环的启停
};

#endif // OPENCVCAMERASERVICE_H
