#ifndef MVCAMERA_H
#define MVCAMERA_H

#include <QObject>
#include <MVGigE.h>
#include <MVImageC.h>
#include <MVCamProptySheet.h>

class MVCamera :public QObject
{
    Q_OBJECT
public:
    explicit MVCamera(QObject *parent = nullptr);
    ~MVCamera();
    // 主界面接口：属性页
    void showProperty();
    // 主界面接口：单次采集
    void grabOnce();
    // 主界面接口：连续采集
    void grabStrat();
    // 主界面接口：停止采集
    void detectCurOnce();
    // 主界面接口：识别
    void saveCurImg();
    // 主界面接口：保存图像
    int convert2Qimg(MV_IMAGE_INFO* pInfo);
signals:
    void imageReady(const QImage& image);
    void errorOccur(const QString& error);
private:
    void initial();
    void release();

    // 相机数目
    int m_nCam;
    // 当前相机句柄
    HANDLE m_hCam;
    // 房钱相机属性
    HANDLE m_hPropDlg;
    // 当前相机图像数据
    HANDLE m_hImg;
    // 当前相机是否运行flag
    bool m_bRun;
};

#endif // MVCAMERA_H
