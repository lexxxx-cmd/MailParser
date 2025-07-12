#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>


#include "../Core/mvcamera.h"

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
    //int showStreamOnLabel(MV_IMAGE_INFO* pInfo);

private slots:
    void on_ButtonShot_clicked();
    void on_ButtonOpen_clicked();
    void on_ButtonStop_clicked();
    void on_Property_clicked();
    void on_ButtonSave_clicked();
    void on_ButtonDetect_clicked();

    // 从相机类收到信号（显示或错误）
    void onImageShow(const QImage& image);
    void onErrorShow(const QString& error);

private:
    Ui::MainWindow *ui;
    // 维视相机指针
    std::unique_ptr<MVCamera> m_Camera;
};
#endif // MAINWINDOW_H
