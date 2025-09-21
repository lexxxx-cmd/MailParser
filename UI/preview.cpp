#include "preview.h"
#include "./ui_preview.h"

Preview::Preview(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Preview)
{
    ui->setupUi(this);
    ui->label->setCursor(Qt::CrossCursor);

    connect(ui->label,&ROISelectorLabel::roiSelected,
            this, &Preview::onRoiSelected);
}

Preview::~Preview()
{
    delete ui;
}

void Preview::on_checkBox_toggled(bool checked)
{
    ui->label->setRoiSelectionEnabled(checked);
}

void Preview::onRoiSelected(const QRect &roi)
{
    // 在这里处理选中的ROI，例如打印坐标、裁剪图像等
    qDebug() << "ROI Selected:" << roi;
}

