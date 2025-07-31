#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <qlabel.h>


#include "../Core/mvcamera.h"
#include "../Core/yolo.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QThread YOLOThread;
public slots:
    // 收到通知显示结果
    void showYOLORes(const QImage& image);
signals:
    // 收到相机QIMAGE通知YOLO检测
    void operateYOLO(const QImage& image);
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
    void setScaledPixmap(QLabel* label, const QPixmap& pixmap);
    void onImageShow(const QImage& image);
    void onErrorShow(const QString& error);

    void on_radioButton_toggled(bool checked);

    void on_radioButton_2_toggled(bool checked);

private:
    Ui::MainWindow *ui;
    // 维视相机指针
    std::unique_ptr<MVCamera> m_Camera;
    // YOLO 指针
    YOLO* mp_Yolo = nullptr;
};
#endif // MAINWINDOW_H
