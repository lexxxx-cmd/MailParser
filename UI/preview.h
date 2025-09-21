#ifndef PREVIEW_H
#define PREVIEW_H

#include <QWidget>
namespace Ui {
class Preview;
}

class Preview : public QWidget
{
    Q_OBJECT

public:
    explicit Preview(QWidget *parent = nullptr);
    ~Preview();

private slots:
    void onRoiSelected(const QRect &roi);

    void on_checkBox_toggled(bool checked);

private:
    Ui::Preview *ui;
};

#endif // PREVIEW_H
