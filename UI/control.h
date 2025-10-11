#ifndef CONTROL_H
#define CONTROL_H

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
    void captureRequested();
    void camStopRequested();
    void DragROIRequested(bool checked);
    void changeCamRequested(int index);//TODO 改为枚举类

private:
    Ui::Control *ui;
};

#endif // CONTROL_H
