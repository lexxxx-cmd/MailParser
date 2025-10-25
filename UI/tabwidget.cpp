#include "tabwidget.h"
#include "ui_tabwidget.h"

TabWidget::TabWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::TabWidget)
{
    ui->setupUi(this);

    // 获取单例实例
    DatabaseManager* dbManager = DatabaseManager::instance();

    // 1. 连接从UI到单例的触发（虽然通常直接调用函数更常见）
    // 比如一个按钮点击后刷新数据
    //connect(ui->tab_2->btnRefresh, &QPushButton::clicked, dbManager, &DatabaseManager::getAllResults);

    // 2. 连接从单例到UI的信号，用于接收数据或错误信息

}

TabWidget::~TabWidget()
{
    delete ui;
}
