#include "database.h"
#include "ui_database.h"

Database::Database(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Database)
{
    ui->setupUi(this);
}

Database::~Database()
{
    delete ui;
}
