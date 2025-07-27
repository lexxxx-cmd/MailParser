#include "yolo.h"
#include <qimage.h>
YOLO::YOLO(const std::string& engine_file_path)
{
    // 读取引擎
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    // 存入内存
    file.read(trtModelStream, size);
    file.close();
    // 初始化插件、运行时
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);
    // 反序列化并创建上下文
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
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
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
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
void YOLO::make_pipe(bool warmup)
{
    // 为输入绑定分配 GPU 内存
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }
    // 为输出绑定分配 GPU 内存 + 主机固定内存
    for (auto& bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
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
void YOLO::letterbox(const QImage& image, cv::Mat& out, cv::Size& size)
{
    // 目标size
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.height();
    float       width  = image.width();

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

void YOLO::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int*  num_dets = static_cast<int*>(this->host_ptrs[0]);
    auto* boxes    = static_cast<float*>(this->host_ptrs[1]);
    auto* scores   = static_cast<float*>(this->host_ptrs[2]);
    int*  labels   = static_cast<int*>(this->host_ptrs[3]);
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Object obj;
        obj.rect.x      = x0;
        obj.rect.y      = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob        = *(scores + i);
        obj.label       = *(labels + i);
        objs.push_back(obj);
    }
    // TODO：发送信号至UI？
}

void YOLO::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

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
    QImage res;
    objs.clear();
    this->copy_from_Mat(image, size);
    emit resReady(res);

}
