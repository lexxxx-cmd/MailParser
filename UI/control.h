#ifndef CONTROL_H
#define CONTROL_H

#include "icameraservice.hpp"
#include "iocrservice.hpp"
#include <QWidget>

namespace Ui {
class Control;
}

class Control : public QWidget
{
    Q_OBJECT

public:
    explicit Control(QWidget *parent = nullptr);
    ~Control();

public slots:
    void setROI(const QRectF& roi);

signals:
    void imgSaveDirSet(const QString& dirPath);
    void captureRequested(CameraType& camType);
    void loadOcrRequested(OcrType& ocrType);
    void camStopRequested();
    void ocrStopRequested();
    void DragROIRequested(bool checked);
    void ApplyROIRequested(const QRectF& roi);
    void ClearROIRequested();
    void ShowROIRequested();

    void changeCamRequested(int index);//TODO 改为枚举类
    void changeOcrRequested(int index);//TODO 改为枚举类

private:
    Ui::Control *ui;
    QRectF getROI() const;
};

#endif // CONTROL_H
