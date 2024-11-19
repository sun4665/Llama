import base64

with open("/zzp/smh/pyworks/lora/examples/loss_plot0.png", "rb") as image_file:
    base64_image_string = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')

# 将编码字符串保存到文件
with open("/zzp/smh/pyworks/lora/base64_image.txt", "w") as output_file:
    output_file.write(base64_image_string)

print("Base64 编码已保存到 /zzp/smh/pyworks/lora/base64_image.txt")