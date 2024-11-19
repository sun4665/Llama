from flask import Flask, Response, request
import os, sys, json
import base64
import torch
from transformers import AutoTokenizer, AutoModel
from io import BytesIO
from PIL import Image

# 设置文件的基础路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 确保 'examples' 文件夹存在
examples_dir = os.path.join(BASE_DIR, 'examples')
os.makedirs(examples_dir, exist_ok=True)

# 创建 Flask 应用实例
app = Flask(__name__)
model_path = os.path.join(BASE_DIR, 'checkpoint', 'visualglm-6b')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

# 路由处理函数
@app.route('/muti_round_chart', methods=['POST'])
def generate_text_stream():
    if request.method != 'POST':
        return Response('request method must be post!', status=405)
    
    # 获取并解析请求数据
    data = request.get_data().decode('utf-8')
    data = json.loads(data)
    image = data['image']
    prompt = data['prompt']
    
    def generate_output():
        torch.cuda.empty_cache()
        
        # 将 base64 字符串转换为图像文件
        def base64_to_image_file(base64_str: str, image_path):
            base64_data = base64_str.split(',')[-1]
            image_data = base64.b64decode(base64_data)
            with open(image_path, 'wb') as f:
                f.write(image_data)
        
        # 设置图像路径并保存图像
        image_path = os.path.join(BASE_DIR, 'examples/xx.png')
        base64_to_image_file(image, image_path)
        
        # 生成流式响应
        for reply, history in model.stream_chat(
            tokenizer,
            image_path,
            prompt,
            history=[],
            max_length=9000,
            top_p=0.4,
            top_k=45,
            temperature=0.4
        ):
            query, response = history[-1]
            yield f'data:{json.dumps(response, ensure_ascii=False)}\n\n'
    
    # 返回流式响应
    return Response(generate_output(), mimetype='text/event-stream')

# 启动应用
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)