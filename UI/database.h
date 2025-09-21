#ifndef DATABASE_H
#define DATABASE_H

#include <QWidget>

namespace Ui {
class Database;
}

class Database : public QWidget
{
    Q_OBJECT

public:
    explicit Database(QWidget *parent = nullptr);
    ~Database();

private:
    Ui::Database *ui;
};

#endif // DATABASE_H
