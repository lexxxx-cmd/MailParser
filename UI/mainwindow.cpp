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
    ui->setupUi(this);
    //创建一个相机对象
    m_Camera = std::make_unique<MVCamera>(parent);
    //设置label最大宽高为Preview的最大值
    ui->label->setMaximumSize(640,480);
    //链接信号与槽
    QObject::connect(m_Camera.get(),&MVCamera::imageReady,this,&MainWindow::onImageShow);
    QObject::connect(m_Camera.get(),&MVCamera::errorOccur,this,&MainWindow::onErrorShow);
    //显示默认界面，不打开相机
    //m_Camera->openCamera(m_Camera->getCameras().first());

    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(false);
    ui->ButtonDetect->setEnabled(false);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_ButtonShot_clicked()
{   
    ui->ButtonShot->setEnabled(true);
    ui->ButtonOpen->setEnabled(true);
    ui->ButtonStop->setEnabled(false);
    ui->ButtonSave->setEnabled(true);
    ui->ButtonDetect->setEnabled(true);
}


void MainWindow::on_ButtonOpen_clicked()
{  
    ui->ButtonShot->setEnabled(false);
    ui->ButtonOpen->setEnabled(false);
    ui->ButtonStop->setEnabled(true);
    ui->ButtonSave->setEnabled(true);
    ui->ButtonDetect->setEnabled(true);
}




void MainWindow::on_ButtonStop_clicked()
{
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
}


void MainWindow::on_ButtonDetect_clicked()
{

}

void MainWindow::onImageShow(const QImage& image)
{
    ui->label->setPixmap(QPixmap::fromImage(image));
}

void MainWindow::onErrorShow(const QString& error)
{
    QMessageBox::StandardButton t_Re = QMessageBox::warning(this,"Warning: ",error,QMessageBox::Yes);
    if (t_Re == QMessageBox::Yes) return;
}
