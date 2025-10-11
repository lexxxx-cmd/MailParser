#include "roiselectorlabel.h"

ROISelectorLabel::ROISelectorLabel(QWidget *parent) : QLabel(parent)
{
    // 初始化QRubberBand，它是用于显示选择框的理想工具
    m_rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
}

QRectF ROISelectorLabel::mapWidgetRectToOriginal(const QRect &widgetRect)
{
    QSize labelSize = this->size();
    // 1. 计算缩放后图片在控件中的偏移量
    float RatioW = static_cast<float>(widgetRect.width()) / labelSize.width();
    float RatioH = static_cast<float>(widgetRect.height()) / labelSize.height();
    float RatioX = static_cast<float>(widgetRect.x()) / labelSize.width();
    float RatioY = static_cast<float>(widgetRect.y()) / labelSize.height();
    return QRectF(RatioX, RatioY, RatioW, RatioH);
}

// 这个公共槽函数是连接复选框的关键
void ROISelectorLabel::setRoiSelectionEnabled(bool enabled)
{
    m_roiSelectionEnabled = enabled;
}

void ROISelectorLabel::mousePressEvent(QMouseEvent *event)
{
    // 如果功能未启用或不是鼠标左键，则不处理
    if (!m_roiSelectionEnabled || event->button() != Qt::LeftButton) {
        return;
    }

    m_originPoint = event->pos();
    m_rubberBand->setGeometry(QRect(m_originPoint, QSize()));
    m_rubberBand->show();
}

void ROISelectorLabel::mouseMoveEvent(QMouseEvent *event)
{
    // 如果功能未启用或rubberband不可见（说明没有按下左键），则不处理
    if (!m_roiSelectionEnabled || !m_rubberBand->isVisible()) {
        return;
    }

    // 更新rubberband的几何形状
    m_rubberBand->setGeometry(QRect(m_originPoint, event->pos()).normalized());
}

void ROISelectorLabel::mouseReleaseEvent(QMouseEvent *event)
{
    if (!m_roiSelectionEnabled || event->button() != Qt::LeftButton || !m_rubberBand->isVisible()) {
        return;
    }

    m_rubberBand->hide();
    QRect selectedRect = m_rubberBand->geometry();

    // 如果选区有效（大小不为0）
    if (selectedRect.width() > 0 && selectedRect.height() > 0) {
        // 发出信号，通知外部世界ROI已经选定
        emit roiSelected(mapWidgetRectToOriginal(selectedRect));
    }
}
