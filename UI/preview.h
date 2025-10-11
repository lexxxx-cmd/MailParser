#ifndef PREVIEW_H
#define PREVIEW_H

#include <QWidget>
#include "../Core/roiselectorlabel.h"

namespace Ui {
class Preview;
}

class Preview : public QWidget
{
    Q_OBJECT

public:
    explicit Preview(QWidget *parent = nullptr);
    ~Preview();
public slots:
    // 定义一个公共槽函数，用于接收并显示图片
    void updateImage(const QImage &image);
    // 调节ROI拖拽功能
    void setRoiSelectionEnabled(bool checked);
private slots:
    void onRoiSelected(const QRectF &roi);

signals:
    void roiSelected(const QRectF &roi);

private:
    Ui::Preview *ui;

    void setScaledPixmap(ROISelectorLabel* label, const QPixmap& pixmap);
};

#endif // PREVIEW_H
