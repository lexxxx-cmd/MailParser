#include "configmanager.h"
#include <QFile>
#include <QTextStream>
#include <stdexcept>

ConfigManager* ConfigManager::instance() {
    static ConfigManager inst;
    return &inst;
}

ConfigManager::ConfigManager() {}

bool ConfigManager::load(const QString& filePath) {
    try {
        m_config = YAML::LoadFile(filePath.toStdString());
        m_isLoaded = true;
        return true;
    } catch (const YAML::Exception& e) {
        // 使用qWarning或你的日志系统记录错误
        qWarning("Failed to load config.yaml: %s", e.what());
        m_isLoaded = false;
        return false;
    }
}

QString ConfigManager::getAppTitle() const {
    if (!m_isLoaded) return "Default Title";
    // 使用 .as<type>() 进行类型转换，并提供默认值
    return QString::fromStdString(m_config["app"]["title"].as<std::string>("Default Title"));
}

QString ConfigManager::getAppGeometry() const {
    if (!m_isLoaded) return "Default dpi";
    // 使用 .as<type>() 进行类型转换，并提供默认值
    return QString::fromStdString(m_config["app"]["geometry"].as<std::string>("Default dpi"));
}


