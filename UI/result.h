#ifndef RESULT_H
#define RESULT_H

#include <QWidget>
#include <qlabel.h>
#include "common.hpp"

namespace Ui {
class Result;
}

class Result : public QWidget
{
    Q_OBJECT

public:
    explicit Result(QWidget *parent = nullptr);
    ~Result();
public slots:
    // 定义一个公共槽函数，用于接收并显示图片
    void updateImage(const QImage &image);
    //void onOcrshow(const QString& ocr);
    void onOcrshow(const RecognitionResult& res);
signals:
    void requestInsertResult(const RecognitionResult& result);

private:
    Ui::Result *ui;

    void setScaledPixmap(QLabel* label, const QPixmap& pixmap);
};

#endif // RESULT_H
