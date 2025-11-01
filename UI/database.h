#ifndef DATABASE_H
#define DATABASE_H

#include <QWidget>
#include <QSqlTableModel>
namespace Ui {
class Database;
}

class Database : public QWidget
{
    Q_OBJECT

public:
    explicit Database(QWidget *parent = nullptr);
    ~Database();
public slots:
    void onRowClicked();

private:
    Ui::Database *ui;
    QSqlTableModel *m_model;

};

#endif // DATABASE_H
