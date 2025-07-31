#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QObject>
#include <QMessageBox>
#include <QScreen>
#include <QDebug>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    cudaSetDevice(0);
    ui->setupUi(this);
    // 创建一个相机对象
    m_Camera = std::make_unique<MVCamera>(parent);
    // 设置label最大宽高为Preview的最大值
    ui->label->setMaximumSize(640,480);
    // 链接信号与槽
    QObject::connect(m_Camera.get(),&MVCamera::imageReady,this,&MainWindow::onImageShow);
    QObject::connect(m_Camera.get(),&MVCamera::errorOccur,this,&MainWindow::onErrorShow);
    // 显示默认界面，不打开相机
    //m_Camera->openCamera(m_Camera->getCameras().first());
    // 创建一个yolo对象
    mp_Yolo = new YOLO("E:/Qt/repos/MailParser/model/mailmodelfp16.engine");// TODO
    mp_Yolo->make_pipe(true);
    mp_Yolo->moveToThread(&YOLOThread);
    // 链接信号与槽
    connect(&YOLOThread,&QThread::finished,mp_Yolo,&QObject::deleteLater);
    connect(this,&MainWindow::operateYOLO,mp_Yolo,&YOLO::pipeline);
    connect(mp_Yolo,&YOLO::resReady,this,&MainWindow::showYOLORes);
    YOLOThread.start();
    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(false);
    ui->ButtonDetect->setEnabled(false);

}

MainWindow::~MainWindow()
{
    YOLOThread.quit();
    YOLOThread.wait();
    delete ui;
}

void MainWindow::on_ButtonShot_clicked()
{
    m_Camera->grabOnce();
    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(true);
    ui->ButtonDetect->setEnabled(true);
}


void MainWindow::on_ButtonOpen_clicked()
{
    m_Camera->grabStrat();
    ui->ButtonShot->setEnabled(false);
    ui->ButtonOpen->setEnabled(false);
    ui->ButtonStop->setEnabled(true);
    ui->ButtonSave->setEnabled(true);
    ui->ButtonDetect->setEnabled(true);
}




void MainWindow::on_ButtonStop_clicked()
{
    m_Camera->stopGrabbing();

    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(false);
    ui->ButtonDetect->setEnabled(false);

}

void MainWindow::on_Property_clicked()
{
    m_Camera->showProperty();
}


void MainWindow::on_ButtonSave_clicked()
{
    bool t_Run = false;
    if(m_Camera->isRun()) {
        t_Run = true;
        on_ButtonStop_clicked();
    }
    QString t_FileName = QFileDialog::getSaveFileName(this, tr("保存图像"),
                                                      "untitled.bmp",
                                                      tr("Images (*.png *.xpm *.jpg *.bmp *.tif)"));
    if (!t_FileName.isEmpty()) {
        char t_File[100];
        sprintf(t_File,"%s",t_FileName.toStdString().c_str());
        m_Camera->saveCurImg(t_File);
    }
    if(t_Run) {
        on_ButtonOpen_clicked();
    }

}


void MainWindow::on_ButtonDetect_clicked()
{

}
void MainWindow::setScaledPixmap(QLabel* label, const QPixmap& pixmap)
{
    // 获取QLabel的可用尺寸
    QSize labelSize = label->size();

    // 计算保持宽高比的缩放尺寸
    QSize scaledSize = pixmap.size();
    scaledSize.scale(labelSize, Qt::KeepAspectRatio);

    // 缩放图片
    QPixmap scaledPixmap = pixmap.scaled(
        scaledSize,
        Qt::IgnoreAspectRatio,
        Qt::SmoothTransformation
        );

    // 设置图片
    label->setPixmap(scaledPixmap);
}
void MainWindow::onImageShow(const QImage& image)
{
    setScaledPixmap(ui->label,QPixmap::fromImage(image));
    //ui->label->setPixmap(QPixmap::fromImage(image));
    // 同时发送给YOLO线程
    emit operateYOLO(image);
}

void MainWindow::onErrorShow(const QString& error)
{
    QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning: ",error,QMessageBox::Yes);
    if (t_Re == QMessageBox::Yes) return;
}

void MainWindow::showYOLORes(const QImage& image)
{
    //qDebug() << "showRes...";
    setScaledPixmap(ui->labelRes,QPixmap::fromImage(image));
    //ui->labelRes->setPixmap(QPixmap::fromImage(image));
}



void MainWindow::on_radioButton_toggled(bool checked)
{
    ui->labelRes->setVisible(false);
    ui->label->setVisible(true);
}


void MainWindow::on_radioButton_2_toggled(bool checked)
{
    ui->label->setVisible(false);
    ui->labelRes->setVisible(true);
}

