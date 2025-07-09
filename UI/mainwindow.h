#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <MVGigE.h>
#include <MVImageC.h>
#include <MVCamProptySheet.h>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    // 显示函数
    int showStreamOnLabel(MV_IMAGE_INFO* pInfo);

private slots:
    void on_ButtonShot_clicked();

    void on_ButtonOpen_clicked();





    void on_ButtonStop_clicked();


    void on_Property_clicked();

    void on_ButtonSave_clicked();

    void on_ButtonDetect_clicked();

private:
    Ui::MainWindow *ui;
    // 相机初始化 构造
    void initialCamera();
    void drawImage();
    HANDLE m_hCam;
    HANDLE m_hImg;
    HANDLE m_hPropDlg;

    bool m_bRun;
};
#endif // MAINWINDOW_H
