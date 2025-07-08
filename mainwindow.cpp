#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QScreen>
#include <QDebug>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_hCam(NULL)
    , m_hImg(NULL)
    , m_hPropDlg(NULL)
    , m_bRun(false)
{
    ui->setupUi(this);
    initialCamera();
    //设置label最大宽高为Preview的最大值

    ui->label->setMaximumSize(640,480);
}

MainWindow::~MainWindow()
{
    delete ui;
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
    //1 MVGigE.h
    MVTerminateLib();
}

void MainWindow::initialCamera()
{
    // 设置鼠标等待效果
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    // 初始化相机库
    MVInitLib();
    // 打开相机
    int nCams = 0;
    //MVGigE.h
    MVGetNumOfCameras(&nCams);
#ifdef __SINGLEGRAB__
    qDebug("found %d cameras\n",nCams);
#endif
    //无相机
    if (nCams == 0)
    {
        //恢复鼠标效果
        QApplication::restoreOverrideCursor();
        QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning","找不到相机！",QMessageBox::Yes);
        if(t_Re == QMessageBox::Yes)
        {
            return ;
        }
    }
    //打开连接到电脑上的第一个相机
    //TODO 默认第一，可选相机
    MVSTATUS_CODES r = MVOpenCamByIndex(0, &m_hCam);
    if (m_hCam == NULL)
    {
        if (r == MVST_ACCESS_DENIED)
        {
            QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning","相机无法访问，可能正被别的程序访问！",QMessageBox::Yes);
            if(t_Re == QMessageBox::Yes)
            {
                return ;
            }
        }
        else
        {
            QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning","相机打开失败，其他原因！",QMessageBox::Yes);
            if(t_Re == QMessageBox::Yes)
            {
                return ;
            }
        }
        return;
    }
    //析构时需要释放相机
    TriggerModeEnums enumMode;
    MVGetTriggerMode(m_hCam, &enumMode);
    //TODO
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
    //根据相机的宽、高、像素格式创建图像
    MVGetPixelFormat(m_hCam, &PixelFormat);
    //根据相机的宽、高、像素格式创建图像
    switch (PixelFormat) {
    case PixelFormat_Mono8:
        m_hImg = MVImageCreate(w, h, 8);
#ifdef __SINGLEGRAB__
        qDebug() << "图像创建：" << 8;
#endif
        break;
    case PixelFormat_Mono16:
        m_hImg = MVImageCreate(w, h, 24);  // 16位灰度
#ifdef __SINGLEGRAB__
        qDebug() << "图像创建：" << 16;
#endif
        break;
    case PixelFormat_BayerBG8:
    case PixelFormat_BayerRG8:
    case PixelFormat_BayerGB8:
    case PixelFormat_BayerGR8:
        m_hImg = MVImageCreate(w, h, 24);  // 转24位BGR
#ifdef __SINGLEGRAB__
        qDebug() << "图像创建：" << 24;
#endif
        break;
    case PixelFormat_BayerBG16:
    case PixelFormat_BayerRG16:
    case PixelFormat_BayerGB16:
    case PixelFormat_BayerGR16:
        m_hImg = MVImageCreate(w, h, 24);  // 转48位BGR（需库支持）
#ifdef __SINGLEGRAB__
        qDebug() << "图像创建：" << 48;
#endif
        break;
    default:
        // 异常处理
        QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning","像素格式未知！",QMessageBox::Yes);
        if(t_Re == QMessageBox::Yes)
        {
            return ;
        }
    }
    //设置相机属性页
    if (m_hPropDlg == NULL)
    {
        //创建及初始化属性页对话框
        const char t_Title[] = "相机属性";
        LPCTSTR strCaption = (LPCTSTR)t_Title;
        //2 MVCamProptySheet.h
        MVCamProptySheetCreateEx(&m_hPropDlg, m_hCam,0,strCaption,0xffff);
        if (m_hPropDlg == NULL)
        {
            QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Waring","创建相机属性页失败！",QMessageBox::Yes);
            if(t_Re == QMessageBox::Yes)
            {
                return ;
            }
        }
    }
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonShot->setEnabled(true);
    //恢复鼠标效果
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_ButtonShot_clicked()
{
    MVSTATUS_CODES r = MVSingleGrab(m_hCam, m_hImg, 500);
    if (r == MVST_SUCCESS)
    {
        drawImage();
    }
    else
    {
        QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"warning","图像采集失败",QMessageBox::Yes);
        if(t_Re == QMessageBox::Yes)
        {
            return ;
        }
    }
    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(true);
}

QImage img2QImage(HANDLE hImg)
{
    int w = MVImageGetWidth(hImg);
    int h = MVImageGetHeight(hImg);
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

void MainWindow::drawImage()
{
    QImage t_Image = img2QImage(m_hImg);

    ui->label->setPixmap(QPixmap::fromImage(t_Image));
    //m_ShowImage->setScaledContents(true);
#ifdef __SINGLEGRAB__
    qDebug() << "width " << w << " height " << h << " bpp " << b;
#endif
}

int MainWindow::showStreamOnLabel(MV_IMAGE_INFO* pInfo)
{
    MVInfo2Image(m_hCam,pInfo,m_hImg);
    drawImage();
    return 0;
}

int __stdcall StreamCB(MV_IMAGE_INFO* pInfo, ULONG_PTR nUserVal)
{
    MainWindow* pWd = (MainWindow*)nUserVal;
    return (pWd->showStreamOnLabel(pInfo));
}

void MainWindow::on_ButtonOpen_clicked()
{
    MVStartGrab(m_hCam, StreamCB, (ULONG_PTR)this);
    m_bRun = TRUE;
    if (m_hPropDlg != NULL)
    {
        //2 MVCamProptySheet.h
        // 如果相机在采集模式，属性页的某些属性不能改变
        MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_ON);
    }
    ui->ButtonShot->setEnabled(false);
    ui->ButtonOpen->setEnabled(false);
    ui->ButtonStop->setEnabled(true);
    ui->ButtonSave->setEnabled(true);
}




void MainWindow::on_ButtonStop_clicked()
{
    MVStopGrab(m_hCam);
    m_bRun = FALSE;
    if (m_hPropDlg != NULL)
    {
        //2 MVCamProptySheet.h
        MVCamProptySheetCameraRun(m_hPropDlg, MVCameraRun_OFF);
    }
    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(false);

}






void MainWindow::on_Property_clicked()
{
    if (m_hPropDlg != NULL)
    {
        //2 MVCamProptySheet.h
        MVCamProptySheetShow(m_hPropDlg, SW_SHOW);
    }
}


void MainWindow::on_ButtonSave_clicked()
{
    bool t_Run = false;
    if (m_bRun)
    {
        t_Run = true;
        on_ButtonStop_clicked();
    }
    HANDLE t_Img = MVImageCreate(MVImageGetWidth(m_hImg), MVImageGetHeight(m_hImg), MVImageGetBPP(m_hImg));
    memcpy(MVImageGetBits(t_Img), MVImageGetBits(m_hImg), MVImageGetPitch(t_Img) * MVImageGetHeight(t_Img));
    QString t_FileName = QFileDialog::getSaveFileName(this, tr("保存图像"),
                                                      "untitled.bmp",
                                                      tr("Images (*.png *.xpm *.jpg *.bmp *.tif)"));
    char t_File[100];
    sprintf(t_File,"%s",t_FileName.toStdString().c_str());
    if (!t_FileName.isEmpty())
    {
        MVImageSave(t_Img,t_File);
    }
    MVImageDestroy(t_Img);
    if (t_Run)
    {
        on_ButtonOpen_clicked();
    }
}

