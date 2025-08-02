conda create -n paddleocr python=3.10
conda activate paddleocr
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddlex
pip install "paddlex[ocr]"
paddlex --install serving
# 启动服务
# paddlex --serve --pipeline OCR
