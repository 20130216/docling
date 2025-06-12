import logging
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
from datetime import datetime  # 用于生成时间戳


def ollama_vlm_options(model: str, prompt: str):
    options = ApiVlmOptions(
        url="http://localhost:11434/v1/chat/completions",
        params=dict(model=model),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def watsonx_vlm_options(model: str, prompt: str):
    load_dotenv(dotenv_path=Path("/Users/wingzheng/Desktop/github/ParseDoc/docling/.env"))
    api_key = os.environ.get("WX_API_KEY")
    project_id = os.environ.get("WX_PROJECT_ID")

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        api_out = res.json()
        print(f"{api_out=}")
        return api_out["access_token"]

    options = ApiVlmOptions(
        url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-07",
        params=dict(
            model_id=model,
            project_id=project_id,
            parameters=dict(max_new_tokens=400),
        ),
        headers={"Authorization": "Bearer " + _get_iam_access_token(api_key=api_key)},
        prompt=prompt,
        timeout=60,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def blendapi_vlm_options(model: str, prompt: str, image_path: Path = None):
    # 优化部分1：加载 .env 文件，保留绝对路径
    load_dotenv(
        dotenv_path=Path("/Users/wingzheng/Desktop/github/ParseDoc/docling/.env"),
        override=True,
    )
    
    # 优化部分2：从环境变量中获取 BlendAPI 配置
    api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT", "https://api.blendapi.com/v1/chat/completions")
    api_key = os.getenv("BLENDAPI_API_KEY")

    # 优化部分3：验证环境变量
    if not api_endpoint or not api_endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Invalid BlendAPI endpoint: {api_endpoint}")
    if not api_key:
        raise ValueError("BlendAPI API key is missing")

    # 优化部分4：调试信息输出
    print(f"DEBUG--BLENDAPI_API_ENDPOINT: {api_endpoint}")
    print(f"DEBUG--BLENDAPI_API_KEY: {api_key}")

    # 优化部分5：处理图像输入
    content = [{"type": "text", "text": prompt}]
    if image_path and image_path.exists():
        try:
            # 假设 image_path 是 JPEG 或 PNG
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    }
                )
                print(f"DEBUG--Image {image_path} encoded to base64, length: {len(img_base64)}")
        except Exception as e:
            print(f"WARNING--Failed to process image {image_path}: {e}")

    # 优化部分6：适配多模态输入
    options = ApiVlmOptions(
        url=api_endpoint,
        params=dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=400,
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        prompt=prompt,
        timeout=60,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def pdf_to_jpeg(pdf_path: Path, output_jpeg: Path, page_num: int = 0):
    """将 PDF 的指定页面转换为 JPEG。"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 放大 2 倍
        pix.save(output_jpeg, "JPEG")
        print(f"DEBUG--Converted PDF page {page_num} to {output_jpeg}")
        doc.close()
        return True
    except Exception as e:
        print(f"WARNING--Failed to convert PDF to JPEG: {e}")
        return False


def main():
    # logging.basicConfig(level=logging.INFO)   # DEBUG ；INFO、WARNING、ERROR、CRITICAL
    logging.basicConfig(level=logging.DEBUG)
    input_doc_path = Path("tests/data/pdf/2305.03393v1-pg9.pdf")
    # 优化部分7：预处理 PDF 为 JPEG
    temp_jpeg_path = Path("./temp_page.jpg")
    if not pdf_to_jpeg(input_doc_path, temp_jpeg_path):
        raise ValueError(f"Failed to convert {input_doc_path} to JPEG")

    pipeline_options = VlmPipelineOptions(enable_remote_services=True)
    # 优化部分8：传递 JPEG 图像路径
    pipeline_options.vlm_options = blendapi_vlm_options(
        model="gpt-4.1",
        prompt="Please perform OCR on the provided page and convert it to markdown format.",
        image_path=temp_jpeg_path,  # 使用转换后的 JPEG
    )
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )
    # 优化部分9：增强错误处理并保存到文件
    try:
        result = doc_converter.convert(input_doc_path)
        markdown_content = result.document.export_to_markdown()
        
        # 优化部分10：生成文件名（解析的文本名 + 当前日期到分钟）
        pdf_name = input_doc_path.stem  # 提取文件名不含扩展名，例如 "2305.03393v1-pg9"
        current_time = datetime.now().strftime("%Y%m%d_%H%M")  # 例如 "20250607_2319"
        output_file = Path(f"{pdf_name}_{current_time}.md")
        
        # 优化部分11：保存到文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"INFO--Markdown content saved to {output_file}")
        
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error: {req_err}")
        if hasattr(req_err, "response") and req_err.response is not None:
            print(f"Response Content: {req_err.response.text}")
        raise
    except Exception as e:
        print(f"Conversion Error: {e}")
        raise
    finally:
        # 优化部分12：清理临时文件
        if temp_jpeg_path.exists():
            temp_jpeg_path.unlink()


if __name__ == "__main__":
    main()