#include "yolo.h"
#include <filesystem>
#include <qimage.h>
#include <QDebug>
YOLO::YOLO(const std::string& engine_file_path)
{
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);
    // 从引擎文件构建
    std::filesystem::path p(engine_file_path);
    if(p.extension() == ".engine") {
        bool built = buildFromEngine(engine_file_path);
    }else if (p.extension() == ".onnx") {
        bool built = buildFromOnnx(engine_file_path);
    }else {
        assert(this->engine != nullptr);
    }

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    cudaStreamCreate(&this->stream);
    // 拿到输入输出信息
#ifdef TRT_10
    this->num_bindings = this->engine->getNbIOTensors();
#else
    this->num_bindings = this->num_bindings = this->engine->getNbBindings();
#endif

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;
#ifdef TRT_10
        std::string        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
#endif
        binding.name  = name;
        binding.dsize = type_to_size(dtype);
#ifdef TRT_10
        bool IsInput = this->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->num_inputs += 1;
#ifdef TRT_10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else {
#ifdef TRT_10
            dims = this->context->getTensorShape(name.c_str());
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLO::~YOLO()
{
#ifdef TRT_10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLO::needOcr()
{
    if (!mb_NeedOcr) mb_NeedOcr = true;
}
bool YOLO::buildFromOnnx(const std::string& onnx_path)
{
    // 按官方api来
    this->builder = nvinfer1::createInferBuilder(this->gLogger);
    if (!this->builder)
    {
        return false;
    }

    this->network = this->builder->createNetworkV2(0);
    if (!this->network)
    {
        return false;
    }

    this->config = this->builder->createBuilderConfig();
    if (!this->config)
    {
        return false;
    }
    // 设置内存限制
    this->config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    // 设置精度模式
    this->config->setFlag(nvinfer1::BuilderFlag::kFP16);

    this->parser
        = nvonnxparser::createParser(*this->network, this->gLogger);
    if (!this->parser)
    {
        return false;
    }
    auto parsed = this->parser->parseFromFile(onnx_path.c_str(),
                                        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        // 输出解析错误
        for (int32_t i = 0; i < this->parser->getNbErrors(); ++i) {
            std::cout << "Parser error: " << this->parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    // 构建序列化网络
    nvinfer1::IHostMemory* serializedModel = this->builder->buildSerializedNetwork(*this->network, *this->config);
    if (!serializedModel) return false;
    // 反序列化构建引擎
    this->engine = this->runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    if (this->engine)
    {
        // 保存
        std::ofstream file("E:/Qt/repos/MailParser/model/mailmodelfp16.engine",std::ios::binary);// TODO:改为变量
        file.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
        file.close();
    }

    // 清理资源 TODO:智能指针
    delete serializedModel;
    delete this->parser;
    delete this->config;
    delete this->network;
    delete this->builder;

    serializedModel = nullptr;

    return this->engine != nullptr;
}

bool YOLO::buildFromEngine(const std::string& engine_path)
{
    // 按yolov8_tensorrt文档来
    std::ifstream file(engine_path, std::ios::binary);
    assert(file.good());

    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    char* trtModelStream = new char[size];
    assert(trtModelStream);
    // 存入内存
    file.read(trtModelStream, size);
    file.close();
    // 反序列化并创建上下文
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    return this->engine != nullptr;
}

void YOLO::make_pipe(bool warmup)
{
    // 准备缓冲区，gpu2个（输入+输出），cpu1个（输出拿数据）
    // 为输入绑定分配 GPU 内存
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream)); // 4 * 1228800(1*3*640*640)
        this->device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str(); // "images"
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }
    // 为输出绑定分配 GPU 内存 + 主机固定内存
    for (auto& bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize; // 2142000(1 * 25200 * 85) * 4
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str(); // “output0”
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 5; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLO::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    // 目标size
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));
    // 添加灰边
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

// 预处理后图像拷贝至gpu
void YOLO::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    int      width      = in_binding.dims.d[3];
    int      height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
}

void YOLO::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    // 从 CPU 异步拷贝到 GPU
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
    // 设置动态输入形状
#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
#endif
}

void YOLO::infer()
{
#ifdef TRT_10
    this->context->enqueueV3(this->stream);
#else
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif
    // 将输出结果从 GPU 异步拷贝到 CPU
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}
/*
void nms(std::vector<Object>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Object) / sizeof(float);
    std::map<float, std::vector<Object>> m;
    for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Object det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        // Prevent code from cross-border access
        auto left = (std::max)(det.bbox[0] - det.bbox[2] / 2.f, 0.f);
        auto top = (std::max)(det.bbox[1] - det.bbox[3] / 2.f, 0.f);
        auto right = (std::min)(det.bbox[0] + det.bbox[2] / 2.f, kInputW - 1.f);
        auto bottom = (std::min)(det.bbox[1] + det.bbox[3] / 2.f, kInputH - 1.f);
        det.bbox[2] = right - left;
        det.bbox[3] = bottom - top;
        det.bbox[0] = left + det.bbox[2] / 2.f;
        det.bbox[1] = top + det.bbox[3] / 2.f;
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
*/
void YOLO::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    std::vector<cv::Rect_<float>> bboxes;
    std::vector<float> scores;
    std::vector<int> indices;
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    float* output = static_cast<float*>(this->host_ptrs[0]);
    int num_anchors = this->output_bindings[0].dims.d[1];
    int obj_dims = this->output_bindings[0].dims.d[2];

    for (int i = 0; i < num_anchors; i++) {
        // 每个框开头地址
        float* anchor = output + i * obj_dims;
        float conf = anchor[4];
        if (conf < 0.25) {
            continue;
        }

        // 找到最大类别概率
        float max_class_prob = 0.0;
        int max_class_id = 0;

        for (int j = 5; j < obj_dims; j++) {
            if (anchor[j] > max_class_prob) {
                max_class_prob = anchor[j];
                max_class_id = j - 5;
            }
        }
        // 解析边界框坐标 (cx, cy, w, h)
        float cx = anchor[0];
        float cy = anchor[1];
        float w = anchor[2];
        float h = anchor[3];

        float final_score = conf * max_class_prob;

        if (final_score < 0.25) {
            continue;
        }
        cx -= dw;
        cy -= dh;

        // 2. 缩放到原始图像尺寸
        cx *= ratio;
        cy *= ratio;
        w *= ratio;
        h *= ratio;

        // 3. 转换为左上角坐标(x,y)
        float x = cx - w/2;
        float y = cy - h/2;

        // 4. 边界保护
        x = std::clamp(x, 0.0f, width);
        y = std::clamp(y, 0.0f, height);
        w = std::clamp(w, 0.0f, width - x);
        h = std::clamp(h, 0.0f, height - y);
        // 存储候选框
        bboxes.push_back(cv::Rect_<float>(x, y, w, h));
        scores.push_back(final_score);
        indices.push_back(max_class_id);

    }

    // NMS

    std::vector<int> keep_indices = FastNMS::nms(bboxes, scores, 0.4, 0.5);

    for (auto idx : keep_indices) {
        Object obj;
        obj.rect = cv::Rect_<float>(bboxes[idx]);
        obj.label = indices[idx];
        obj.prob = scores[idx];
        objs.push_back(obj);
    }
}


void YOLO::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(0x27, 0xC1, 0x36);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", "mail", obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

void YOLO::pipeline(const QImage& image)
{
    // ui线程通知拿到图像数据转为mat格式
    cv::Mat res;
    cv::Mat mat = Converter::QImage2Mat(image);
    cv::Size            size = cv::Size{640, 640};
    objs.clear();
    this->copy_from_Mat(mat, size);
    auto start = std::chrono::system_clock::now();
    this->infer();
    this->postprocess(objs);
    auto end = std::chrono::system_clock::now();
    this->draw_objects(mat, res, objs);
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    qDebug() << "time: " << tc;
    emit resReady(Converter::cvMatToQImage(res));

    // 打包发送图像与rect数据？或者用一个bool变量来控制是否发送至ROI的ui以及ocr线程？
    if (mb_NeedOcr && !objs.empty()) {
        cv::Mat roi = mat(objs[0].rect);
        // 发送信号至ui及ocr线程
        emit roiReady(roi);
        mb_NeedOcr = false;
    }
}
