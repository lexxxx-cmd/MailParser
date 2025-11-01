#ifndef MVGIGECAMERA_H
#define MVGIGECAMERA_H

#include "icameraservice.hpp"
#include <MVGigE.h>
#include <MVImageC.h>
#include <MVCamProptySheet.h>

/**
 * @brief MVGigE相机的具体实现
 *
 * 继承自ICameraService，并实现了其定义的纯虚函数。
 * 所有与MVGigE SDK相关的头文件、句柄和API调用都封装在此类内部。
 */
class MVGigECamera : public ICameraService
{
    Q_OBJECT
public:
    explicit MVGigECamera(int cameraIndex = 0, QObject *parent = nullptr);
    ~MVGigECamera() override;
public slots:
    // --- 实现ICameraService的接口 ---
    bool initialize() override;
    void release() override;
    bool startGrabbing() override;
    bool stopGrabbing() override;
    bool grabOnce() override;
    bool saveImage() override;

    // MV特有功能
    void showPropertyDialog();
    // bool setTriggerMode(bool enable);
    int convert2QImage(MV_IMAGE_INFO* pInfo);

private:
    int m_nCam;
    HANDLE m_hCam;
    HANDLE m_hPropDlg;
    HANDLE m_hImg;
    QImage convertMVImageToQImage(HANDLE hImg);

    static int __stdcall streamCallback(MV_IMAGE_INFO* pInfo, ULONG_PTR nUserVal);
};

#endif // MVGIGECAMERA_H

