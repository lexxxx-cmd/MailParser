#ifndef ROISELECTORLABEL_H
#define ROISELECTORLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QRect>
#include <QRubberBand>

class ROISelectorLabel : public QLabel
{
    Q_OBJECT

public:
    explicit ROISelectorLabel(QWidget *parent = nullptr);
    QRectF mapWidgetRectToOriginal(const QRect &widgetRect);

public slots:
    void setRoiSelectionEnabled(bool enabled);

signals:
    void roiSelected(const QRectF &roi);

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    bool m_roiSelectionEnabled = false;
    QPoint m_originPoint;
    QRubberBand *m_rubberBand = nullptr;
    QRect   m_originalRoi;      // 存储在原始图片坐标系下的ROI
    QRect   m_scaledRoi;        // 存储计算出的、在当前显示坐标系下的ROI

};

#endif // ROISELECTORLABEL_H

