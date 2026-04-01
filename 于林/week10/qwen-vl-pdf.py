import os
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
from openai import OpenAI

# 初始化 OpenAI 客户端（使用百炼 API）
client = OpenAI(
    api_key='sk-9bf45d961ac64f75a3b6a64c7fd08817',  # 请确保已设置环境变量 DASHSCOPE_API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def image_to_base64(image_path):
    """将图像文件转换为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def pdf_page_to_image(pdf_path, page_num, dpi=200):
    """将 PDF 的某一页转换为 PIL Image 对象"""
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)  # 设置缩放矩阵
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    pdf_doc.close()
    return img

def extract_images_from_pdf(pdf_path, output_dir):
    """提取 PDF 中所有嵌入的图片并保存到 output_dir，返回图片文件列表"""
    pdf_doc = fitz.open(pdf_path)
    img_list = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(pdf_doc, xref)
            if pix.n - pix.alpha < 4:  # 灰度或RGB图片
                img_data = pix.tobytes("png")
                img_filename = f"page{page_num+1}_img{img_index+1}.png"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_data)
                img_list.append(img_path)
            else:  # CMYK 或其他格式，先转换
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.tobytes("png")
                img_filename = f"page{page_num+1}_img{img_index+1}.png"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_data)
                img_list.append(img_path)
            pix = None
    pdf_doc.close()
    return img_list

def parse_pdf_with_qwen_vl(pdf_path, output_md_path, images_output_dir):
    """
    使用 qwen-vl-max 模型解析 PDF 文件
    - pdf_path: PDF 文件路径
    - output_md_path: 输出的 Markdown 文件路径
    - images_output_dir: 提取的图片保存目录
    """
    # 创建图片保存目录
    os.makedirs(images_output_dir, exist_ok=True)

    # 1. 提取 PDF 中的原始图片（如果有）
    extracted_images = extract_images_from_pdf(pdf_path, images_output_dir)
    print(f"已提取 {len(extracted_images)} 张图片到 {images_output_dir}")

    # 2. 将 PDF 每一页转换为图像并调用模型解析
    pdf_doc = fitz.open(pdf_path)
    total_pages = len(pdf_doc)
    pdf_doc.close()

    markdown_content = ""

    for page_num in range(total_pages):
        print(f"正在处理第 {page_num+1}/{total_pages} 页...")
        # 将当前页转换为 PIL Image
        img = pdf_page_to_image(pdf_path, page_num, dpi=150)
        # 将图像转为 base64 字符串
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # 构建请求消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请将这张图片中的内容以 Markdown 格式输出，包括文本、表格、公式等。如果图片中有嵌入的图片，请用占位符描述。"
                    }
                ]
            }
        ]

        # 调用模型（非流式）
        completion = client.chat.completions.create(
            model="qwen-vl-max",  # 可根据需要更换模型，如 qwen-vl-plus
            messages=messages,
            stream=False
        )

        page_content = completion.choices[0].message.content
        # 在 Markdown 中添加页码标题
        markdown_content += f"## 第 {page_num+1} 页\n\n{page_content}\n\n---\n\n"

    # 3. 保存 Markdown 文件
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Markdown 文件已保存至 {output_md_path}")
    print(f"图片已保存至 {images_output_dir}")

if __name__ == "__main__":
    # 示例使用
    pdf_file = "Week04-Transfomer和BERT、GPT模型.pdf"            # 替换为你的 PDF 文件路径
    md_output = "output.md"             # 输出的 Markdown 文件
    img_dir = "extracted_images"        # 提取图片的保存目录

    parse_pdf_with_qwen_vl(pdf_file, md_output, img_dir)