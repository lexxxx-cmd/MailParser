#include "mvgigecamera.h"


MVGigECamera::MVGigECamera(int cameraIndex, QObject *parent)
    : ICameraService{parent},m_nCam(cameraIndex),m_hCam(NULL),m_hPropDlg(NULL),m_hImg(NULL)
{
    m_type = CameraType::MvGigeCamera;
}

MVGigECamera::~MVGigECamera()
{
    release();
}

bool MVGigECamera::initialize() {
    if (m_state != CameraState::Uninitialized) {
        release();
    }
    MVInitLib();

    int nCams = 0;
    MVGetNumOfCameras(&nCams);
    if (nCams == 0 || m_cameraIndex >= nCams) {
        emit errorOccur("未找到MV相机");
        setState(CameraState::Error);
        return false;
    }
    MVSTATUS_CODES r = MVOpenCamByIndex(m_cameraIndex, &m_hCam);
    if (r != MVST_SUCCESS || m_hCam == nullptr) {
        emit errorOccur("打开MV相机失败");
        setState(CameraState::Error);
        return false;
    }
    // 设置为连续非触发模式
    MVSetTriggerMode(m_hCam, TriggerMode_Off);
    MVSetHeartbeatTimeout(m_hCam, 5000);
    // 创建图像缓冲区
    int w, h;
    MV_PixelFormatEnums pixelFormat;
    MVGetWidth(m_hCam, &w);
    MVGetHeight(m_hCam, &h);
    MVGetPixelFormat(m_hCam, &pixelFormat);
    int bpp = (pixelFormat == PixelFormat_Mono8) ? 8 : 24;
    m_hImg = MVImageCreate(w, h, bpp);

    if (m_hImg == nullptr) {
        emit errorOccur("创建图像缓冲区失败");
        MVCloseCam(m_hCam);
        m_hCam = nullptr;
        setState(CameraState::Error);
        return false;
    }
    setState(CameraState::Idle);
    return true;
}

void MVGigECamera::release() {

    if (m_state == CameraState::Grabbing) {
        MVStopGrab(m_hCam);
    }
    if (m_hPropDlg != nullptr) {
        MVCamProptySheetDestroy(m_hPropDlg);
        m_hPropDlg = nullptr;
    }
    if (m_hImg != nullptr) {
        MVImageDestroy(m_hImg);
        m_hImg = nullptr;
    }
    if (m_hCam != nullptr) {
        MVCloseCam(m_hCam);
        m_hCam = nullptr;
    }
    MVTerminateLib();
    setState(CameraState::Uninitialized);
}

bool MVGigECamera::startGrabbing()
{

    if (m_state != CameraState::Idle) {
        emit errorOccur("相机未就绪");
        return false;
    }
    MVSTATUS_CODES r = MVStartGrab(m_hCam, streamCallback, (ULONG_PTR)this);
    if (r != MVST_SUCCESS) {
        emit errorOccur("启动采集失败");
        return false;
    }
    if (m_hPropDlg != nullptr) {
        MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_ON);
    }
    setState(CameraState::Grabbing);
    return true;
}
bool MVGigECamera::stopGrabbing()
{

    if (m_state != CameraState::Grabbing) {
        return true;
    }
    MVSTATUS_CODES r = MVStopGrab(m_hCam);
    if (r != MVST_SUCCESS) {
        emit errorOccur("停止采集失败");
        return false;
    }
    if (m_hPropDlg != nullptr) {
        MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_OFF);
    }
    setState(CameraState::Idle);
    return true;
}

bool MVGigECamera::grabOnce()
{

    if (m_state == CameraState::Uninitialized) {
        emit errorOccur("相机未初始化");
        return false;
    }
    MVSTATUS_CODES r = MVSingleGrab(m_hCam, m_hImg, 500);
    if (r != MVST_SUCCESS) {
        emit errorOccur("单次采集失败");
        return false;
    }
    QImage img = convertMVImageToQImage(m_hImg);
    if (!img.isNull()) {
        m_lastImage = img;
        emit imageReady(img);
        return true;
    }

    return false;
}

bool MVGigECamera::saveImage(const QString& filepath)
{

    if (m_hImg == nullptr) {
        emit errorOccur("无图像可保存");
        return false;
    }
    HANDLE tempImg = MVImageCreate(
        MVImageGetWidth(m_hImg),
        MVImageGetHeight(m_hImg),
        MVImageGetBPP(m_hImg)
        );

    if (tempImg == nullptr) {
        emit errorOccur("创建临时图像失败");
        return false;
    }
    memcpy(
        MVImageGetBits(tempImg),
        MVImageGetBits(m_hImg),
        MVImageGetPitch(tempImg) * MVImageGetHeight(tempImg)
        );
    try{
        MVImageSave(tempImg, filepath.toStdString().c_str());
    }catch(std::exception &e){
        emit errorOccur(e.what());
        return false;
    }
    MVImageDestroy(tempImg);
    return true;
}

void MVGigECamera::showPropertyDialog()
{

    if (m_hCam == nullptr) {
        emit errorOccur("相机未初始化");
        return;
    }
    if (m_hPropDlg == nullptr) {
        const char title[] = "Camera Property";
        MVCamProptySheetCreateEx(&m_hPropDlg, m_hCam, 0, (LPCTSTR)title, 0xffff);
    }
    if (m_hPropDlg != nullptr) {
        MVCamProptySheetShow(m_hPropDlg, SW_SHOW);
    }
}

QImage MVGigECamera::convertMVImageToQImage(HANDLE hImg)
{
    if (hImg == nullptr) {
        return QImage();
    }
    int w = MVImageGetWidth(hImg);
    int h = MVImageGetHeight(hImg);
    int bpp = MVImageGetBPP(hImg);
    int pitch = MVImageGetPitch(hImg);
    unsigned char* pData = (unsigned char*)MVImageGetBits(hImg);
    if (bpp == 8) {
        QImage image(pData, w, h, pitch, QImage::Format_Indexed8);
        image.setColorCount(256);
        for (int i = 0; i < 256; i++) {
            image.setColor(i, qRgb(i, i, i));
        }
        return image.copy(); // 深拷贝
    }
    else if (bpp == 24) {
        QImage image(pData, w, h, pitch, QImage::Format_RGB888);
        return image.rgbSwapped().copy();
    }
    return QImage();
}
int MVGigECamera::convert2QImage(MV_IMAGE_INFO* pInfo)
{
    if (pInfo == nullptr) {
        return -1;
    }
    MVInfo2Image(m_hCam, pInfo, (MVImage*)m_hImg);
    QImage img = convertMVImageToQImage(m_hImg);

    if (!img.isNull()) {
        m_lastImage = img;
        emit imageReady(img);
        return 0;
    }

    return -1;
}

int __stdcall MVGigECamera::streamCallback(MV_IMAGE_INFO* pInfo, ULONG_PTR nUserVal)
{
    MVGigECamera* pWorker = (MVGigECamera*)nUserVal;
    return pWorker->convert2QImage(pInfo);
}
