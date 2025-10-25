#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H

#include "databaseworker.h"
#include <QObject>
#include <QStandardPaths>
#include <qthread.h>

class DatabaseManager : public QObject
{
    Q_OBJECT
private:
    explicit DatabaseManager(QObject *parent = nullptr);
    ~DatabaseManager();

    void initDatabase(); // 新增：私有的初始化方法
    void _deleteResult(int id);
    void _updateResult(int id, const RecognitionResult& result);
    void _getAllResults(const QString& sortBy = "");


    static DatabaseManager* m_instance;
    QThread m_dbThread; // 工作线程实例
    DatabaseWorker* m_worker; // 工作对象指针
    QString m_dbPath; // 新增：存储数据库文件路径

public:
    static DatabaseManager* instance();

    // 异步调用接口
    void insertResult(const RecognitionResult& result);
    void exportAllResults();
    void deleteResult(const int& recordId);
    void deleteResultAll();
    void editResult(const int& recordId);
    void updateResult();
    void getAllResults();
    void getResultById(int id);

    // 图片管理（现在只负责路径生成和逻辑，I/O委托给Worker）
    // QString generateImagePath();
    // QImage loadImage(const QString& path); // 这个可以保留在主线程，如果加载很快的话，或者也做成异步

signals:
    // 转发给Worker的信号
    void requestInitDatabase();
    void requestInsertResult(const RecognitionResult& result);
    void requestDeleteResult(int id);
    void requestDeleteAllResult();
    void requestCheckResult(int id);
    void requestUpdateResult(int id, const RecognitionResult& result);
    void requestGetAllResults(const QString& sortBy = "");
    // ... 其他请求信号

    // 从Worker接收并转发给UI的信号
    void dataChanged();
    void operationFailed(const QString& error);
    void allResultsFetched(const QList<RecognitionResult>& results);
    // ... 其他结果信号
};

#endif // DATABASEMANAGER_H
