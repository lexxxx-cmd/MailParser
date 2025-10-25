#include "result.h"
#include "ui_result.h"
#include <QString>
#include "../Core/databasemanager.h"

Result::Result(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Result)
{
    ui->setupUi(this);
    DatabaseManager* dbManager = DatabaseManager::instance();

    connect(this,&Result::requestInsertResult, dbManager, &DatabaseManager::insertResult);

}

Result::~Result()
{
    delete ui;
}

void Result::setScaledPixmap(QLabel* label, const QPixmap& pixmap)
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

void Result::updateImage(const QImage &image)
{
    if (image.isNull()) {
        qDebug() << "Received a null image.";
        return;
    }
    qDebug() << "DisplayWidget received image, updating UI.";
    // 将QImage转换为QPixmap并显示在QLabel上
    setScaledPixmap(ui->lblRecognitionImageView,QPixmap::fromImage(image));
}

// void Result::onOcrshow(const QString& ocr)
// {
//     ui->lblRecognitionResultText->append(ocr);
// }

void Result::onOcrshow(const RecognitionResult& res)
{
    ui->lblRecognitionResultText->append(QString::number(res.getTimeStamp()));
    emit requestInsertResult(res);
}
