#include "control.h"
#include "ui_control.h"

Control::Control(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Control)
{
    ui->setupUi(this);

    //开启摄像头信号
    connect(ui->btnStartCamera, &QPushButton::clicked, this, &Control::captureRequested);
    //关闭摄像头信号
    connect(ui->btnStopCamera, &QPushButton::clicked, this, &Control::camStopRequested);
    //获取并检测信号
    connect(ui->btnDetectCurrentFrame, &QPushButton::clicked, this, &Control::ShowROIRequested);
    //拖拽ROI信号
    connect(ui->chkEnableRoiDrag, &QCheckBox::toggled, this, &Control::DragROIRequested);
    //应用ROI信号
    connect(ui->btnApplyRoi, &QCheckBox::clicked, this, [=](){
        emit ApplyROIRequested(getROI());
    });
    //清除ROI
    connect(ui->btnClearRoi, &QCheckBox::clicked, this,&Control::ClearROIRequested);
    //切换mv相机信号
    connect(ui->rbCameraMvGige, &QRadioButton::toggled, this, [=](){
        emit changeCamRequested(0);
    });
    //切换opencv相机信号
    connect(ui->rbCameraOpencv, &QRadioButton::toggled, this, [=](){
        emit changeCamRequested(1);
    });
}

Control::~Control()
{
    delete ui;
}

void Control::setROI(const QRectF& roi) {
    ui->lblRoiX->setText(QString::number(roi.x()));
    ui->lblRoiY->setText(QString::number(roi.y()));
    ui->lblRoiW->setText(QString::number(roi.width()));
    ui->lblRoiH->setText(QString::number(roi.height()));
}
QRectF Control::getROI() const {
    return QRectF(
        ui->lblRoiX->text().toDouble(),
        ui->lblRoiY->text().toDouble(),
        ui->lblRoiW->text().toDouble(),
        ui->lblRoiH->text().toDouble()
        );
}
