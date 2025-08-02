#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "cpp-httplib/httplib.h"
#include <Windows.h>
#include "nlohmann/json.hpp"
#include "base64.hpp"

int main() {
    httplib::Client client("localhost", 8080);
    const std::string filePath = "./demo.jpg";

    // 1. 打开文件并获取大小
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // 2. 读取文件内容到 buffer
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);  // 明确指定 vector 的元素类型为 char

    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    // 3. 将 buffer 转换为字符串并 Base64 编码
    std::string bufferStr(buffer.data(), static_cast<size_t>(size));  // 明确 static_cast 的目标类型
    std::string encodedFile = base64::to_base64(bufferStr);
    auto start = std::chrono::system_clock::now();
    // 4. 构造 JSON 请求并发送
    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/ocr", jsonObj.dump(), "application/json");

    // 5. 处理响应
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["ocrResults"].is_array()) {
            std::cerr << "Unexpected response structure." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["ocrResults"].size(); ++i) {
            auto ocrResult = result["ocrResults"][i];
            std::cout << ocrResult["prunedResult"] << std::endl;

            std::string ocrImgPath = "ocr_" + std::to_string(i) + ".jpg";
            std::string encodedImage = ocrResult["ocrImage"];
            std::string decodedImage = base64::from_base64(encodedImage);

            std::ofstream outputImage(ocrImgPath, std::ios::binary);
            if (outputImage.is_open()) {
                outputImage.write(decodedImage.c_str(), static_cast<std::streamsize>(decodedImage.size()));  // 修正 static_cast
                outputImage.close();
                std::cout << "Output image saved at " << ocrImgPath << std::endl;
            } else {
                std::cerr << "Unable to open file for writing: " << ocrImgPath << std::endl;
            }
        }
    } else {
        std::cerr << "Failed to send HTTP request." << std::endl;
        if (response) {
            std::cerr << "HTTP status code: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
        return 1;
    }
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    std::cout << "time: "<< tc << "/n";
    return 0;
}
