#include "databasemanager.h"
#include <qdir.h>
#include <QSqlError>
#include <QSqlQuery>

DatabaseManager::DatabaseManager(QObject *parent)
    : QObject{parent}
{
    // 1. 确定数据库路径
    // 推荐将数据库放在应用程序数据目录，而不是和可执行文件放一起
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dir(dataPath);
    if (!dir.exists()) {
        dir.mkpath("."); // 如果目录不存在，则创建它
    }
    m_dbPath = dataPath + "/recognition_data.db";
    qDebug() << "数据库路径设置为:" << m_dbPath;

    // 2. 执行初始化检查和创建
    initDatabase();

    m_worker = new DatabaseWorker(m_dbPath);
    m_worker->moveToThread(&m_dbThread);


    // 连接请求信号：Manager -> Worker
    connect(this, &DatabaseManager::requestInitDatabase, m_worker, &DatabaseWorker::doInitDatabase);
    connect(this, &DatabaseManager::requestInsertResult, m_worker, &DatabaseWorker::doInsertResult);
    connect(this, &DatabaseManager::requestDeleteResult, m_worker, &DatabaseWorker::doDeleteResult);
    connect(this, &DatabaseManager::requestDeleteAllResult, m_worker, &DatabaseWorker::doDeleteAllResult);
    // ... 连接所有请求

    // 连接结果信号：Worker -> Manager
    connect(m_worker, &DatabaseWorker::dataChanged, this, [=](){
        emit dataChanged();
    });
    // connect(m_worker, &DatabaseWorker::allResultsFetched, this, &DatabaseManager::allResultsFetched);
    // ... 连接所有结果

    // 线程生命周期管理
    connect(&m_dbThread, &QThread::finished, m_worker, &QObject::deleteLater);

    emit requestInitDatabase();
    m_dbThread.start();

}

DatabaseManager::~DatabaseManager()
{
    m_dbThread.quit();
    m_dbThread.wait();
}

DatabaseManager* DatabaseManager::instance() {
    static DatabaseManager instance;
    return &instance;
}

void DatabaseManager::initDatabase()
{
    // 使用一个唯一的连接名，避免与工作线程的连接冲突
    const QString connectionName = "init_connection";

    // 添加数据库驱动
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connectionName);
    db.setDatabaseName(m_dbPath);

    // 尝试打开数据库，如果文件不存在，Qt会自动创建
    if (!db.open()) {
        qCritical() << "数据库初始化失败：无法打开数据库文件！错误:" << db.lastError().text();
        // 在实际应用中，这里可能需要抛出异常或退出程序
        return;
    }

    qDebug() << "数据库文件打开成功。";

    // 检查 'recognition_results' 表是否存在
    QSqlQuery checkQuery(db);
    checkQuery.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='recognition_results'");

    if (!checkQuery.exec()) {
        qCritical() << "检查表是否存在时出错:" << checkQuery.lastError().text();
        db.close();
        QSqlDatabase::removeDatabase(connectionName);
        return;
    }

    // 如果查询结果为空 (checkQuery.next() 返回 false)，说明表不存在
    if (!checkQuery.next()) {
        qDebug() << "'recognition_results' 表不存在，现在开始创建...";

        // 使用 C++11 的原始字符串字面量，可以方便地书写多行SQL
        // const QString createTableSql = R"(
        //     CREATE TABLE recognition_results (
        //         id INTEGER PRIMARY KEY AUTOINCREMENT,
        //         timestamp TEXT NOT NULL,
        //         zip_code TEXT,
        //         barcode TEXT,
        //         address TEXT,
        //         receiver TEXT,
        //         raw_texts_json TEXT,
        //         image_path TEXT NOT NULL,
        //         thumbnail_path TEXT,
        //         score REAL DEFAULT 0.0,
        //         grade TEXT DEFAULT '未知',
        //         created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        //         updated_time DATETIME DEFAULT CURRENT_TIMESTAMP
        //     );
        // )";
        const QString createTableSql = R"(
            CREATE TABLE recognition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                raw_texts TEXT,
                zip_code TEXT,
                barcode TEXT,
                address TEXT,
                receiver TEXT,
                score REAL,
                grade TEXT,
                img_path TEXT
            );
        )";

        QSqlQuery createQuery(db);
        if (!createQuery.exec(createTableSql)) {
            qCritical() << "创建 'recognition_results' 表失败！错误:" << createQuery.lastError().text();
        } else {
            qDebug() << "'recognition_results' 表创建成功。";
        }
    } else {
        qDebug() << "'recognition_results' 表已存在，无需创建。";
        // 未来可以在这里添加逻辑，检查表结构是否需要更新（ALTER TABLE）
    }

    // 关闭并移除这个临时的初始化连接
    db.close();
    QSqlDatabase::removeDatabase(connectionName);
}

void DatabaseManager::insertResult(const RecognitionResult& result){
    emit requestInsertResult(result);
    qDebug() << "发送增数据信号";
}
void DatabaseManager::deleteResult(const int& recordId){
    emit requestDeleteResult(recordId);
    qDebug() << "发送删数据信号";
}
void DatabaseManager::editResult(const int& recordId) {
    //TODO
    // 拿到id 和修改后的值
    // updateResult()
    qDebug() << "发送改数据信号";
}
void DatabaseManager::updateResult(){
    //emit requestUpdateResult(id, result);
    qDebug() << "发送改数据信号";
}
void DatabaseManager::getAllResults(){
    //emit requestGetAllResults(sortBy);
    qDebug() << "发送查询所有数据信号";
}
void DatabaseManager::getResultById(int id){
    //emit requestCheckResult(id);
    qDebug() << "发送查数据信号";
}

void DatabaseManager::deleteResultAll() {
    emit requestDeleteAllResult();
    qDebug() << "发送删所有数据信号";
}

void DatabaseManager::exportAllResults() {
    qDebug() << "发送导出数据信号";
}
