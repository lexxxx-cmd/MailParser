#include "mvcamera.h"
#include <QImage>
#include <QFileDialog>

MVCamera::MVCamera(QObject *parent):QObject(parent),m_nCam(0),m_hCam(NULL),m_hPropDlg(NULL),m_hImg(NULL) {
    initial();
}
MVCamera::~MVCamera() {
    release();
}
void MVCamera::initial() {
    MVInitLib();
    int nCams = 0;
    //MVGigE.h
    MVGetNumOfCameras(&nCams);
    //TODO nCam == 0 ?
    //打开相机
    MVSTATUS_CODES r = MVOpenCamByIndex(0, &m_hCam);
    //TODO m_hCam == NULL

    TriggerModeEnums enumMode;
    MVGetTriggerMode(m_hCam, &enumMode);
    //触发模式
    if (enumMode != TriggerMode_Off)
    {
        //设置为连续非触发模式
        MVSetTriggerMode(m_hCam, TriggerMode_Off);
    }
    // 显示图像
    int w, h;
    //3 GigECamera_Types.h
    MV_PixelFormatEnums PixelFormat;
    MVGetWidth(m_hCam, &w);
    MVGetHeight(m_hCam, &h);
    MVGetPixelFormat(m_hCam, &PixelFormat);

    if (PixelFormat == PixelFormat_Mono8) {
        m_hImg = MVImageCreate(w, h, 8);
    }else {
        m_hImg = MVImageCreate(w, h, 24);
    }
    //设置属性页
    if (m_hPropDlg == NULL)
    {
        //创建及初始化属性页对话框
        const char t_Title[] = "Camera Property";
        LPCTSTR strCaption = (LPCTSTR)t_Title;
        //2 MVCamProptySheet.h
        MVCamProptySheetCreateEx(&m_hPropDlg, m_hCam,0,strCaption,0xffff);
        //TODO 创建失败 if (m_hPropDlg == NULL)
    }
}
void MVCamera::release() {
    if (m_hCam != NULL)
    {
        //1 MVGigE.h
        MVCloseCam(m_hCam);
        m_hCam = NULL;
    }
    if (m_hPropDlg != NULL)
    {
        //销毁属性页对话框
        //2 MVCamProptySheet.h
        MVCamProptySheetDestroy(m_hPropDlg);
        m_hPropDlg = NULL;
    }

    MVTerminateLib();
}

void MVCamera::showProperty() {
    if (m_hPropDlg != NULL)
    {
        //2 MVCamProptySheet.h
        MVCamProptySheetShow(m_hPropDlg, SW_SHOW);
    }
}

QImage img2QImage(HANDLE hImg)
{
    int w = MVImageGetWidth(hImg); // 1280
    int h = MVImageGetHeight(hImg); // 960
    int bpp = MVImageGetBPP(hImg);
    int pitch = MVImageGetPitch(hImg);
    unsigned char *pImgData = (unsigned char *)MVImageGetBits(hImg);

    if (bpp == 8)
    {
        uchar *pSrc = pImgData;
        QImage image(pSrc, w,h, pitch, QImage::Format_Indexed8);
        image.setColorCount(256);
        for (int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        return image;
    }
    else if (bpp == 24)
    {
        const uchar *pSrc = (const uchar*)pImgData;
        QImage image(pSrc, w,h, pitch, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else
    {
        return QImage();
    }
}

int MVCamera::convert2Qimg(MV_IMAGE_INFO* pInfo)
{
    MVInfo2Image(m_hCam,pInfo,(MVImage *)m_hImg);
    QImage t_Image = img2QImage(m_hImg);
    emit imageReady(t_Image);
    return 0;
}
int __stdcall StreamCB(MV_IMAGE_INFO* pInfo, ULONG_PTR nUserVal)
{
    MVCamera* pCam = (MVCamera*)nUserVal;
    return (pCam->convert2Qimg(pInfo));
}

void MVCamera::grabStrat() {
    MVStartGrab(m_hCam, StreamCB, (ULONG_PTR)this);
    m_bRun = TRUE;
    if (m_hPropDlg != NULL)
    {
        //2 MVCamProptySheet.h
        // 如果相机在采集模式，属性页的某些属性不能改变
        MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_ON);
    }
}

void MVCamera::stopGrabbing() {
    MVSTATUS_CODES r = MVStopGrab(m_hCam);
    if (r == MVST_SUCCESS){
        m_bRun = FALSE;
        if (m_hPropDlg != NULL)
        {
            //2 MVCamProptySheet.h
            MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_OFF);
            //qDebug("关闭相机成功");
        }
    }else{
        emit errorOccur("停止相机失败！");
    }
}

void MVCamera::grabOnce() {
    MVSTATUS_CODES r = MVSingleGrab(m_hCam,m_hImg,500); // TODO 通过配置改变曝光时间
    if (r == MVST_SUCCESS) {
        QImage t_Image = img2QImage(m_hImg);
        emit imageReady(t_Image);
    }else {
        //QImage t_Image = QImage();
        emit errorOccur("单次采集失败！");
    }
}

void MVCamera::saveCurImg(LPCSTR file) {

    HANDLE t_Img = MVImageCreate(MVImageGetWidth(m_hImg), MVImageGetHeight(m_hImg), MVImageGetBPP(m_hImg));
    memcpy(MVImageGetBits(t_Img), MVImageGetBits(m_hImg), MVImageGetPitch(t_Img) * MVImageGetHeight(t_Img));

    MVImageSave(t_Img,file);

    MVImageDestroy(t_Img);

}

bool MVCamera::isRun() const {
    return m_bRun;
}
