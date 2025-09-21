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

public slots:
    void setRoiSelectionEnabled(bool enabled);

signals:
    void roiSelected(const QRect &roi);

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    bool m_roiSelectionEnabled = false;
    QPoint m_originPoint;
    QRubberBand *m_rubberBand = nullptr;
};

#endif // ROISELECTORLABEL_H

