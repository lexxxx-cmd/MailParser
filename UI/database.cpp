#include "database.h"
#include "ui_database.h"
#include "../Core/databasemanager.h"
#include <qdir.h>

Database::Database(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Database)
{
    ui->setupUi(this);
    // 获取单例实例
    DatabaseManager* dbManager = DatabaseManager::instance();


    connect(ui->btnEdit, &QPushButton::clicked, this, [=](){
        QModelIndex currentIndex = ui->tblMailData->currentIndex();
        if (!currentIndex.isValid()) {
            return; // 没有选中行
        }

        // 从模型中获取主键'id'的值
        // 假设 'id' 是第 0 列
        int primaryKeyColumn = 0;
        QModelIndex idIndex = m_model->index(currentIndex.row(), primaryKeyColumn);
        int recordId = m_model->data(idIndex).toInt();

        // 发射信号，将主键传递给后台处理
        DatabaseManager::instance()->editResult(recordId);
    });
    connect(ui->btnExport, &QPushButton::clicked, dbManager, &DatabaseManager::exportAllResults);
    connect(ui->btnDelete, &QPushButton::clicked, this, [=](){
        QModelIndex currentIndex = ui->tblMailData->currentIndex();
        if (!currentIndex.isValid()) {
            return; // 没有选中行
        }

        // 从模型中获取主键'id'的值
        // 假设 'id' 是第 0 列
        int primaryKeyColumn = 0;
        QModelIndex idIndex = m_model->index(currentIndex.row(), primaryKeyColumn);
        int recordId = m_model->data(idIndex).toInt();

        // 发射信号，将主键传递给后台处理
        DatabaseManager::instance()->deleteResult(recordId);

    });
    connect(ui->btnDeleteAll, &QPushButton::clicked, dbManager, &DatabaseManager::deleteResultAll);

    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE"); // 默认连接
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dir(dataPath);
    if (!dir.exists()) {
        dir.mkpath("."); // 如果目录不存在，则创建它
    }
    const QString m_dbPath = dataPath + "/recognition_data.db";
    db.setDatabaseName(m_dbPath);
    if (!db.open()) {
        qCritical() << "UI数据库连接失败!";
    }

    m_model = new QSqlTableModel;
    m_model->setTable("recognition_results");
    m_model->select();

    ui->tblMailData->setModel(m_model);

    // 连接来自工作线程的信号，当数据变化时刷新模型
    connect(DatabaseManager::instance(), &DatabaseManager::dataChanged,
            m_model, &QSqlTableModel::select);
    connect(ui->btnRefresh, &QPushButton::clicked,
            m_model, &QSqlTableModel::select);
    ui->tblMailData->hideColumn(0);
    ui->tblMailData->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tblMailData->show();
    connect(ui->tblMailData, &QTableView::clicked, this, &Database::onRowClicked);
}

Database::~Database()
{
    delete ui;
}

void Database::onRowClicked() {
    QModelIndex currentIndex = ui->tblMailData->currentIndex();
    if (!currentIndex.isValid()) {
        return; // 没有选中行
    }

    // 从模型中获取主键'id'的值
    // 假设 'id' 是第 0 列
    int primaryKeyColumn = 0;
    QModelIndex idIndex = m_model->index(currentIndex.row(), primaryKeyColumn);
    int recordId = m_model->data(idIndex).toInt();

    // 读取数据库图片到label上
    QString imagePath;
    QSqlQuery query; // 会使用默认数据库连接
    query.prepare("SELECT img_path FROM recognition_results WHERE id = ?");
    query.addBindValue(recordId);

    if (query.exec() && query.next()) {
        imagePath = query.value(0).toString();
    } else {
        qWarning() << "无法找到ID为" << recordId << "的记录的图片路径。";
        ui->lblDetectPic->clear(); // 清空标签
        return;
    }

    // 2. 使用QPixmap加载图片
    QPixmap pixmap(imagePath);

    // 3. 检查图片是否加载成功
    if (pixmap.isNull()) {
        qWarning() << "加载图片失败，路径可能无效:" << imagePath;
        // 可以设置一张默认的“加载失败”图片
        ui->lblDetectPic->setText("图片加载失败");
        return;
    }

    // 4. 在QLabel上显示图片，并进行缩放以适应Label大小
    ui->lblDetectPic->setPixmap(pixmap.scaled(
        ui->lblDetectPic->size(),       // 缩放到Label的尺寸
        Qt::KeepAspectRatio,          // 保持宽高比
        Qt::SmoothTransformation      // 平滑缩放
        ));
}
