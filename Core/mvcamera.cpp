#include "mvcamera.h"

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
