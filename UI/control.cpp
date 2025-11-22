#include "control.h"
#include "ui_control.h"
#include <QFileDialog>
#include <QDir> // 为了使用 QDir::homePath()
#include <QMessageBox>

enum class ROIType {
    defualt,
    typeONe
};

Control::Control(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Control)
{
    ui->setupUi(this);
    //保存路径信号
    connect(ui->btnSelectSaveDir, &QPushButton::clicked, this, [=](){
        // 弹出对话框，让用户选择目录
        QString dirPath = QFileDialog::getExistingDirectory(
            this,                                 // 父窗口
            tr("选择保存目录"),                    // 对话框标题
            QDir::homePath(),                     // 默认打开的目录
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks | QFileDialog::DontUseNativeDialog
            );

        // 如果用户选择了目录（而不是取消），则路径不为空
        if (!dirPath.isEmpty()) {
            dirPath = QDir::fromNativeSeparators(dirPath);
            ui->lblSaveDirPath->setText(dirPath);
            emit imgSaveDirSet(dirPath);
        }else {
            QMessageBox msg;
            msg.setIcon(QMessageBox::Warning);
            msg.setText("必须选择裁剪图片保存目录！！！");
            msg.addButton(QMessageBox::Ok);
            msg.exec();
        }

    });
    //开启摄像头信号
    connect(ui->btnStartCamera, &QPushButton::clicked, this, [=](){
        CameraType curCamera = ui->rbCameraMvGige->isChecked() ? CameraType::MvGigeCamera : CameraType::OpenCvCamera;
        emit captureRequested(curCamera);
    });
    //关闭摄像头信号
    connect(ui->btnStopCamera, &QPushButton::clicked, this, &Control::camStopRequested);

    //开启ocr信号
    connect(ui->btnLoadOcrEngine, &QPushButton::clicked, this, [=](){
        ui->stackedWidget->setCurrentIndex(0);
        OcrType curOcr = ui->rbOcrLocal->isChecked() ? OcrType::LocalOcr : OcrType::XydOcr;
        emit loadOcrRequested(curOcr);
        RemoteConfig curConfig;
        getConfig(curConfig);
        emit changeRemoteConfig(curConfig);
    });
    //关闭ocr信号
    connect(ui->btnStopOcrEngine, &QPushButton::clicked, this, [=](){
        ui->stackedWidget->setCurrentIndex(1);
        emit ocrStopRequested();

    });

    //获取并检测信号
    connect(ui->btnDetectCurrentFrame, &QPushButton::clicked, this, [=](){
        emit imgSaveDirSet(ui->lblSaveDirPath->text());
        emit ShowROIRequested();
    });
    //拖拽ROI信号
    connect(ui->chkEnableRoiDrag, &QCheckBox::toggled, this, &Control::DragROIRequested);
    //应用ROI信号
    connect(ui->btnApplyRoi, &QCheckBox::clicked, this, [=](){
        emit ApplyROIRequested(getROI());
    });
    //清除ROI
    connect(ui->btnClearRoi, &QCheckBox::clicked, this,&Control::ClearROIRequested);
    //切换mv相机信号
    connect(ui->rbCameraMvGige, &QRadioButton::toggled, this, [=](){
        emit changeCamRequested(0);
    });
    //切换opencv相机信号
    connect(ui->rbCameraOpencv, &QRadioButton::toggled, this, [=](){
        emit changeCamRequested(1);
    });

    //切换本地ocr信号
    connect(ui->rbOcrLocal, &QRadioButton::toggled, this, [=](){
        ui->widget_ocr->hide();
        emit changeOcrRequested(0);
    });
    //切换远端ocr信号
    connect(ui->rbOcrRemote, &QRadioButton::toggled, this, [=](){
        ui->widget_ocr->show();
        emit changeOcrRequested(1);
    });

    ui->stackedWidget->setCurrentIndex(1);
    ui->widget_ocr->hide();
    QValidator *validator = new QIntValidator(0, 99, this);

    // 2. 将验证器设置给 QLineEdit
    ui->lblBarcodeLength->setValidator(validator);

    // 假设 ui->comboBox 是你的控件
    // 定义一个Lambda来封装初始化逻辑，避免污染外部变量作用域
    auto initRoiCombo = [](QComboBox* combo) {
        // 定义一个临时结构体，仅用于此处的数据组织
        struct RoiConfig { QString name; QRectF roi; };

        // --- 核心配置区：像写配置文件一样简洁直观 ---
        const QList<RoiConfig> configs = {
            { "全屏模式 (Full)",    QRectF(0, 0, 1, 1) },
            { "左半屏 (Split L)",   QRectF(0, 0, 0.5, 1)  },
            { "右半屏 (Split R)",   QRectF(0.5, 0, 0.5, 1)},
            { "特写镜头 (Detail)",  QRectF(0, 0.1, 0.7, 0.5) }
        };
        // ---------------------------------------

        combo->clear();
        // 批量填充：自动将 QRect 存入 UserData
        for (const auto& item : configs) {
            combo->addItem(item.name, item.roi);
        }

    };

    // 执行 Lambda
    initRoiCombo(ui->cmbROISet);
    // 假设 ui->comboBox 已经按之前的方案二填充了 QRect 数据

    // 显式指定连接 activated(int) 重载版本
    connect(ui->cmbROISet, QOverload<int>::of(&QComboBox::activated),
            this, [=](int index) {
                // 1. 关键点：使用 itemData(index) 而不是 currentData()
                // 这样能保证取到的就是用户刚刚点击的那一项，无论当前选中状态如何
                QVariant data = ui->cmbROISet->itemData(index);

                // 2. 转换为 QRect
                QRectF roi = data.toRectF();

                // 3. 校验有效性（这是一个好习惯）
                if (roi.isValid() || !roi.isNull()) {
                    qDebug() << "User activated item at index" << index
                             << "with ROI:" << roi;

                    // 在这里执行你的业务逻辑，例如：
                    setROI(roi);
                } else {
                    qWarning() << "Activated item has no valid ROI data";
                }
            });
}

Control::~Control()
{
    delete ui;
}

void Control::setROI(const QRectF& roi) {
    ui->lblRoiX->setText(QString::number(roi.x()));
    ui->lblRoiY->setText(QString::number(roi.y()));
    ui->lblRoiW->setText(QString::number(roi.width()));
    ui->lblRoiH->setText(QString::number(roi.height()));
}
QRectF Control::getROI() const {
    return QRectF(
        ui->lblRoiX->text().toDouble(),
        ui->lblRoiY->text().toDouble(),
        ui->lblRoiW->text().toDouble(),
        ui->lblRoiH->text().toDouble()
        );
}

void Control::getConfig(RemoteConfig& config) {
    config.address = ui->lblRemoteAddress->text();
    config.key = ui->lblRemoteKey->text();
    config.barcodeLength = ui->lblBarcodeLength->text().toInt();
    config.analysisType = 1-ui->cmbAnalysisType->currentIndex();
}
