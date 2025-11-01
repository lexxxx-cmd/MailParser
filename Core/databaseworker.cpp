#include "databaseworker.h"
#include "common.hpp"
#include <QSqlError>
#include <QThread>
#include <qcoreapplication.h>

// 构造函数保存路径，但不立即打开数据库
DatabaseWorker::DatabaseWorker(const QString& dbPath, QObject *parent)
    : QObject(parent), m_dbPath(dbPath)
{
    // 注意：不要在这里创建或打开数据库连接！
    // 因为此时 DatabaseWorker 对象仍在主线程。
    // 数据库连接必须在它被移动到的目标线程（工作线程）中创建。
}

DatabaseWorker::~DatabaseWorker()
{
    if (m_database.isOpen()) {
        m_database.close();
    }
}

// 在第一个需要数据库的操作中打开连接，或者通过一个init槽来打开
bool DatabaseWorker::openDatabase()
{
    // 检查是否已经打开
    if (m_database.isOpen()) {
        return true;
    }

    // 确保我们在工作线程中
    Q_ASSERT(QThread::currentThread() != QCoreApplication::instance()->thread());

    // 为工作线程创建唯一的连接
    const QString connectionName = QString("worker_connection_%1").arg(quintptr(QThread::currentThreadId()));
    m_database = QSqlDatabase::addDatabase("QSQLITE", connectionName);
    m_database.setDatabaseName(m_dbPath);

    if (!m_database.open()) {
        qCritical() << "工作线程无法打开数据库:" << m_database.lastError().text();
        emit operationFailed("数据库连接失败。");
        return false;
    }

    qDebug() << "工作线程数据库连接成功。";


    return true;
}

void DatabaseWorker::doInitDatabase(){
    if(!openDatabase()) {
        qDebug() << "工作线程数据库连接失败。";
        emit operationFailed("数据库连接失败。");
    }
}
void DatabaseWorker::doInsertResult(const RecognitionResult& result) {
    // 假设 m_db 是此工作线程中已经打开的 QSqlDatabase 连接
    QSqlQuery query(m_database);

    // 使用预处理语句防止SQL注入
    // query.prepare(R"(
    //     INSERT INTO recognition_results (
    //         timestamp, zip_code, barcode, address, receiver,
    //         raw_texts_json, image_path, thumbnail_path, score, grade
    //     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    // )");
    query.prepare(R"(
        INSERT INTO recognition_results (
            timestamp, raw_texts, zip_code, barcode, address, receiver, score, grade, img_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    )");

    query.addBindValue(result.getTimeStamp());
    query.addBindValue(result.getText());
    query.addBindValue(result.getZipCode());
    query.addBindValue(result.getBarCode());
    query.addBindValue(result.getAddress());
    query.addBindValue(result.getReceiver());

    // query.addBindValue(result.thumbnail_path);
    query.addBindValue(result.getScore());
    query.addBindValue(result.getGrade());
    query.addBindValue(result.getImgPath());

    if (!query.exec()) {
        emit operationFailed("插入数据失败: " + query.lastError().text());
    } else {
        emit dataChanged(); // 通知数据已变更
    }
}

void DatabaseWorker::doDeleteResult(int id){
    QSqlQuery query(m_database);
    query.prepare("DELETE FROM recognition_results WHERE id = ?");
    query.addBindValue(id);

    if (!query.exec()) {
        emit operationFailed("删除记录失败: " + query.lastError().text());
    } else {
        // 检查是否真的有行被删除
        if (query.numRowsAffected() > 0) {
            emit dataChanged(); // 成功，通知UI刷新
        } else {
            emit operationFailed("删除记录失败: 无此ID或已被删除。");
        }
    }
}

void DatabaseWorker::doDeleteAllResult(){
    QSqlQuery query(m_database);

    m_database.transaction();
    if(!query.exec("DELETE FROM recognition_results")) {
        qCritical() << "DELETE 失败:" << query.lastError().text();
        m_database.rollback();
        emit operationFailed("删除表失败！");
        return;
    }

    if(!query.exec("DELETE FROM sqlite_sequence WHERE name='recognition_results'")) {
        if(query.lastError().isValid()) {
            qWarning() << "重置自增计数器失败:" << query.lastError().text();
            emit operationFailed("重置自增计数器失败");
            m_database.rollback();
        }
        return;
    }
    if (m_database.commit()) {
        emit dataChanged(); // 仅在成功提交后通知UI
    } else {
        qCritical() << "事务提交失败:" << m_database.lastError().text();
        m_database.rollback(); // 提交失败也需要回滚
        emit operationFailed("数据操作提交失败！");
    }
}
