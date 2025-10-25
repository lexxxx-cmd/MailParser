#ifndef DATABASEWORKER_H
#define DATABASEWORKER_H

#include <QObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include "common.hpp"

class DatabaseWorker : public QObject
{
    Q_OBJECT
public:
    explicit DatabaseWorker(const QString& dbPath, QObject *parent = nullptr);
    ~DatabaseWorker();
private:
    QSqlDatabase m_database;
    QString m_dbPath;

    bool openDatabase();
    // void closeDatabase();

//     // 辅助函数
//     QString saveImageToFile(const QImage& image);
//     bool deleteImageFile(const QString& path);
public slots:
//     // 初始化槽函数，由主线程信号触发
    void doInitDatabase();

//     // CRUD 槽函数
    void doInsertResult(const RecognitionResult& result);
    void doDeleteResult(int id);
    void doDeleteAllResult();
//     void doUpdateResult(int id, const RecognitionResult& result);
//     void doGetAllResults(const QString& sortBy);
//     void doGetResultById(int id);

//     // 批量/耗时操作槽函数
//     void doExportToExcel(const QString& path);
//     void doBackupDatabase(const QString& backupPath);
//     void doGenerateThumbnails(const QList<int>& ids);

signals:
//     // 向上层（主线程）报告结果的信号
//     void databaseInitialized(bool success, const QString& error);
//     void resultInserted(int newId, bool success);
//     void resultDeleted(bool success);
//     void resultUpdated(bool success);
//     void allResultsFetched(const QList<RecognitionResult>& results);
//     void resultFetchedById(const RecognitionResult& result);

//     // 耗时操作完成信号
//     void exportFinished(bool success, const QString& message);
//     void backupFinished(bool success, const QString& message);
//     void thumbnailsGenerated();

//     // 通用信号
    void dataChanged(); // 用于通知UI刷新
    void operationFailed(const QString& error);
};

#endif // DATABASEWORKER_H
