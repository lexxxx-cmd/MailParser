#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H

#include <QString>
#include <QSize>
#include "yaml-cpp/yaml.h"


class ConfigManager
{
public:
    static ConfigManager* instance();
    bool load(const QString& filePath = "config.yaml");
    // 提供类型安全的getter方法
    QString getAppTitle() const;
    QString getAppGeometry() const;
    // ... 其他getter
private:
    ConfigManager(); // 私有构造函数
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    YAML::Node m_config;
    bool m_isLoaded = false;
};

#endif // CONFIGMANAGER_H
