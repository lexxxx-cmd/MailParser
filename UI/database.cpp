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

    connect(ui->btnRefresh, &QPushButton::clicked, dbManager, &DatabaseManager::getAllResults);
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
    ui->tblMailData->hideColumn(0);
    ui->tblMailData->show();
}

Database::~Database()
{
    delete ui;
}
