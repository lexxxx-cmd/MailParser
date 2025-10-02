#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <QWidget>

namespace Ui {
class Recognition;
}

class Recognition : public QWidget
{
    Q_OBJECT

public:
    explicit Recognition(QWidget *parent = nullptr);
    ~Recognition();

private:
    Ui::Recognition *ui;
};

#endif // RECOGNITION_H
