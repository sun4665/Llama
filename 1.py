import argparse
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--quant", type=int, choices=[4, 8], help="Quantization level (4 or 8)")
args = parser.parse_args()

def predict(input, image_path, chatbot, max_length, top_p,
            temperature, history):
    if image_path is None:
        return [(input, "图片不能为空。请重新上传图片。")], []
    chatbot.append((input, ""))
    with torch.no_grad():
        for response, history in model.stream_chat(
                tokenizer,
                image_path,
                input,
                history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            chatbot[-1] = (input, response)
            yield chatbot, history


def predict_new_image(image_path, chatbot, max_length,
                      top_p, temperature):
    input, history = "描述这张图片。", []
    chatbot.append((input, ""))
    with torch.no_grad():
        for response, history in model.stream_chat(
                tokenizer,
                image_path,
                input,
                history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature):
            chatbot[-1] = (input, response)
            yield chatbot, history


def reset_user_input():
    return gr.update(value='')

def reset_state():
    return None, [], []


global model, tokenizer
# 使用本地路径加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/zzp/smh/pyworks/lora/checkpoint/visualglm-6b", trust_remote_code=True
)

# 使用本地路径加载 model
if args.quant in [4, 8]:
    model = AutoModel.from_pretrained(
        "/zzp/smh/pyworks/lora/checkpoint/visualglm-6b", trust_remote_code=True
    ).quantize(args.quant).half().cuda()
else:
    model = AutoModel.from_pretrained(
        "/zzp/smh/pyworks/lora/checkpoint/visualglm-6b", trust_remote_code=True
    ).half().cuda()

model = model.eval()


def main():
    with gr.Blocks(css="style.css") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                image_path = gr.Image(
                    type="filepath",
                    label="Image Prompt",
                    value=None,
                    height=504  # 将高度直接作为参数传入
                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=480  # 将高度直接作为参数传入
                )
        with gr.Row():
            with gr.Column(scale=2, min_width=100):
                max_length = gr.State(1024)
                top_p = gr.Slider(0, 1, value=0.4, step=0.01,
                                  label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.8, 
                                        step=0.01, label="Temperature", 
                                        interactive=True)
            with gr.Column(scale=4):
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input = gr.Textbox(show_label=False, 
                                                    placeholder="Input...",
                                                    lines=4).style(container=False)
                        with gr.Column(scale=1, min_width=64):
                            submitBtn = gr.Button("Generate")
                            emptyBtn = gr.Button("Clear")
        history = gr.State([])

        submitBtn.click(predict, [user_input, image_path, chatbot,
                                  max_length, top_p, temperature, history],
                        [chatbot, history], show_progress=True)
        
        image_path.upload(predict_new_image, [image_path, chatbot, 
                                              max_length, top_p, temperature],
                          [chatbot, history], show_progress=True)

        image_path.clear(reset_state, outputs=[image_path, chatbot, history], 
                         show_progress=True)
        
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[image_path, chatbot, history], 
                       show_progress=True)
        
        demo.queue().launch(inbrowser=True, server_name='0.0.0.0', server_port=8080)


if __name__ == '__main__':