#ifndef ICAMERASERVICE_H
#define ICAMERASERVICE_H
#include <QObject>
#include <QImage>
#include <QMutex>
#include <qdebug.h>
#include <QDateTime>

// 我们将使用class而不是struct，因为它有方法。
// 并且为了在智能指针中使用，我们进行前向声明。
enum class CameraType {
    MvGigeCamera,
    OpenCvCamera
    // ... 可以继续添加其他相机类型
};

enum class CameraState {
    Uninitialized,
    Idle,
    Grabbing,
    Error
};
/**
 * @brief 相机服务的抽象接口
 *
 * 定义了所有相机实现类必须遵循的通用契约。
 * 它对任何特定的相机SDK（如MVGigE, OpenCV等）都一无所知。
 */
class ICameraService : public QObject
{
    Q_OBJECT
public:
    explicit ICameraService(QObject *parent = nullptr) : QObject(parent), m_state(CameraState::Uninitialized) {}
    virtual ~ICameraService() = default;

    // --- 纯虚函数：定义标准相机控制接口 ---

    virtual bool initialize() = 0;
    virtual void release() = 0;
    virtual bool startGrabbing() = 0;
    virtual bool stopGrabbing() = 0;
    virtual bool grabOnce() = 0;
    virtual bool saveImage() = 0;


    // --- 纯虚函数：定义标准相机状态查询接口 ---

    CameraState getState() const { return m_state; }
    CameraType getType() const { return m_type; }
    bool isGrabbing() const { return m_state == CameraState::Grabbing; }
public slots:
    void setImgSaveDir(QString filepath) {
        m_filepath = filepath;
    }
    void setROI(const QRectF& roi) {
        ROI = roi;
    }
    void showROI() {
        sendROIImg();
    }
    void resetROI() {
        ROI = QRectF(0.0, 0.0, 1.0, 1.0);
    }

signals:
    /** @brief 当一帧新的图像准备好时发出此信号 */
    void imageReady(const QImage& frame);

    /** @brief 当发生错误时发出此信号 */
    void errorOccur(const QString& errorMessage);

    void stateChanged(CameraState state);

    void roiImageReady(const QImage &roiImage, const QString &filepath);

protected:
    int m_cameraIndex;
    CameraState m_state;
    CameraType m_type;
    QString m_filepath;
    QImage m_lastImage; // 缓存最后一帧

    QRectF ROI{0.0,0.0,1.0,1.0};



    void setState(CameraState state) {
        if (m_state != state) {
            m_state = state;
            emit stateChanged(state);
        }
    };
    void sendROIImg() {
        if (m_lastImage.isNull()) {
            qWarning() << "Last image is null, cannot send ROI image";
            return;
        }

        if (!ROI.isValid() || ROI.isEmpty()) {
            qWarning() << "ROI is invalid or empty";
            return;
        }

        // 将比例值转换为实际像素坐标
        int imgWidth = m_lastImage.width();
        int imgHeight = m_lastImage.height();

        QRect pixelROI(
            qRound(ROI.x() * imgWidth),
            qRound(ROI.y() * imgHeight),
            qRound(ROI.width() * imgWidth),
            qRound(ROI.height() * imgHeight)
            );

        // 确保ROI在有效范围内
        pixelROI = pixelROI.intersected(QRect(0, 0, imgWidth, imgHeight));

        if (pixelROI.isEmpty()) {
            qWarning() << "Calculated ROI is empty or outside bounds";
            return;
        }

        // 提取ROI区域
        QImage roiImage = m_lastImage.copy(pixelROI);

        if (roiImage.isNull()) {
            qWarning() << "Failed to copy ROI from image";
            return;
        }
        //saveImage();
        // 发射信号

        QString filepath = m_filepath + "/" + QString::number(QDateTime::currentMSecsSinceEpoch()) +  ".jpg";
        emit roiImageReady(roiImage, filepath);
        roiImage.save(filepath);
    }
};

#endif // ICAMERASERVICE_H
