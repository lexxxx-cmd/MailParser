#include "preview.h"
#include "./ui_preview.h"

Preview::Preview(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Preview)
{
    ui->setupUi(this);
    ui->camView->setText("Waiting for image...");
    ui->camView->setCursor(Qt::CrossCursor);

    connect(ui->camView,&ROISelectorLabel::roiSelected,
            this, &Preview::onRoiSelected);
}

Preview::~Preview()
{
    delete ui;
}

void Preview::setRoiSelectionEnabled(bool checked)
{
    ui->camView->setRoiSelectionEnabled(checked);
}

void Preview::onRoiSelected(const QRectF &roi)
{
    // 在这里处理选中的ROI，例如打印坐标、裁剪图像等
    emit roiSelected(roi);
    qDebug() << "ROI Selected:" << roi;
}

void Preview::setScaledPixmap(ROISelectorLabel* label, const QPixmap& pixmap)
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

void Preview::updateImage(const QImage &image)
{
    if (image.isNull()) {
        qDebug() << "Received a null image.";
        return;
    }
    qDebug() << "DisplayWidget received image, updating UI.";
    // 将QImage转换为QPixmap并显示在QLabel上
    setScaledPixmap(ui->camView,QPixmap::fromImage(image));
}

