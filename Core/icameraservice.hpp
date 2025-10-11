#ifndef ICAMERASERVICE_H
#define ICAMERASERVICE_H
#include <QObject>
#include <QImage>
#include <QMutex>

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
    virtual bool saveImage(const QString& filepath) = 0;

    // --- 纯虚函数：定义标准相机状态查询接口 ---

    CameraState getState() const { return m_state; }
    CameraType getType() const { return m_type; }
    bool isGrabbing() const { return m_state == CameraState::Grabbing; }


signals:
    /** @brief 当一帧新的图像准备好时发出此信号 */
    void imageReady(const QImage& frame);

    /** @brief 当发生错误时发出此信号 */
    void errorOccur(const QString& errorMessage);

    void stateChanged(CameraState state);

protected:
    int m_cameraIndex;
    CameraState m_state;
    CameraType m_type;
    QImage m_lastImage; // 缓存最后一帧

    void setState(CameraState state) {
        if (m_state != state) {
            m_state = state;
            emit stateChanged(state);
        }
    };
};

#endif // ICAMERASERVICE_H
