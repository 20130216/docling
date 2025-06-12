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
from datetime import datetime
import time
import argparse
import pymupdf as fitz
from PIL import Image
import io
import base64
import pandas as pd
import aiohttp
import asyncio
import pytesseract
import json
import re
import unicodedata
from typing import Optional
from asyncio import Semaphore

_log = logging.getLogger(__name__)

# 配置 Tesseract 路径
def get_tesseract_path() -> Optional[str]:
    try:
        import shutil
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            _log.debug(f"Tesseract 找到：{tesseract_path}")
            return tesseract_path
        else:
            _log.warning("Tesseract 未找到，请确保 Tesseract 已安装并在系统 PATH 中")
            return None
    except Exception as e:
        _log.warning(f"检测 Tesseract 失败：{e}，请确保 Tesseract 已安装")
        return None

pytesseract.pytesseract.tesseract_cmd = get_tesseract_path() or '/opt/homebrew/bin/tesseract'
if not pytesseract.pytesseract.tesseract_cmd:
    _log.warning("Tesseract 路径未设置，图像文本提取可能受限")

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log")
    ],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 优化：限制 max_tokens 上限为 32,768
def estimate_max_tokens(input_path: Path) -> int:
    try:
        if input_path.suffix.lower() in [".pdf"]:
            doc = fitz.open(input_path)
            total_tokens = 0
            for page in doc:
                text = page.get_text()
                images = page.get_images()
                text_tokens = len(text) // 3
                image_tokens = len(images) * 500
                total_tokens += text_tokens + image_tokens
            doc.close()
            buffer = 1.2
            return min(int(total_tokens * buffer), 32768)  # 限制上限
        elif input_path.suffix.lower() in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(input_path)
            total_tokens = 1000
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                total_tokens += len(df.to_markdown()) // 3
            return min(total_tokens, 32768)
        else:
            with Image.open(input_path) as img:
                width, height = img.size
                pixels = width * height
                return min(10000 + (pixels // 10000) * 100, 32768)
    except Exception as e:
        _log.warning(f"无法估算 token 数：{e}，使用默认 16000")
        return 16000

# 优化：检测复杂页面，降低 zoom
def pdf_to_jpeg(pdf_path: Path, output_jpeg: Path, page_num: int = 0) -> bool:
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        has_complex_content = len(page.get_images()) > 0 or any(
            annot for annot in page.annots() if annot.type[0] in [8, 9]
        ) or "Math" in page.get_text()  # 检测公式
        zoom = 1.0 if has_complex_content else 1.2  # 复杂页面降低 zoom
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        quality = 80 if has_complex_content else 70
        text_blocks = page.get_text("blocks")
        if text_blocks:
            x0, y0, x1, y1 = text_blocks[0][0], text_blocks[0][1], text_blocks[-1][2], text_blocks[-1][3]
            margin = 50
            crop_box = (
                max(0, int(x0 - margin)),
                max(0, int(y0 - margin)),
                min(img.width, int(x1 + margin)),
                min(img.height, int(y1 + margin))
            )
            crop_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
            orig_area = img.width * img.height
            if crop_area < 0.1 * orig_area:
                _log.warning(f"页面 {page_num} 裁剪面积过小 ({crop_area}/{orig_area})，回退到无裁剪")
                crop_box = (0, 0, img.width, img.height)
            else:
                _log.debug(f"页面 {page_num} 裁剪坐标：{crop_box}")
                img = img.crop(crop_box)
        img.save(output_jpeg, "JPEG", quality=quality, optimize=True)
        _log.debug(f"PDF 页面 {page_num} 转换为 {output_jpeg}，zoom={zoom}，质量={quality}%")
        doc.close()
        return True
    except Exception as e:
        _log.error(f"无法将 PDF 转换为 JPEG：{e}")
        return False

def convert_to_jpeg(input_path: Path, output_jpeg: Path) -> bool:
    try:
        if input_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                quality = 80
                img.save(output_jpeg, "JPEG", quality=quality, optimize=True)
                _log.debug(f"已将图像 {input_path} 转换为 {output_jpeg}，质量={quality}%")
                return True
        elif input_path.suffix.lower() == ".pdf":
            return pdf_to_jpeg(input_path, output_jpeg)
        else:
            raise ValueError(f"不支持的文件类型：{input_path.suffix}")
    except Exception as e:
        _log.error(f"无法将 {input_path} 转换为 JPEG：{e}")
        return False

def excel_to_text(excel_path: Path) -> str:
    try:
        xls = pd.ExcelFile(excel_path)
        markdown = ""
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            markdown += f"## Sheet: {sheet_name}\n\n{df.to_markdown(index=False)}\n\n"
        _log.debug(f"Excel 内容大小：{len(markdown)} 字符")
        return markdown
    except Exception as e:
        _log.error(f"无法提取 Excel 内容：{e}")
        raise ValueError(f"无法提取 Excel 内容：{e}")

def normalize_markdown_headers(markdown: str) -> str:
    if not markdown:
        _log.warning("Markdown 内容为空，返回空字符串")
        return ""
    lines = markdown.split("\n")
    normalized_lines = []
    current_level = 1
    for line in lines:
        if line.startswith("#"):
            hashes = len(line.split()[0])
            if hashes > 0:
                line = f"{'#' * min(hashes, current_level)} {line.lstrip('# ').strip()}"
                current_level = min(hashes + 1, 6)
        # 过滤掉包含提示的行
        if any(keyword in line for keyword in ["以下是", "注：", "表格内容", "图像内容", "如需后续"]):
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()

def extract_image_text(img: Image.Image) -> Optional[str]:
    try:
        text = pytesseract.image_to_string(img, lang="chi_sim", config="--psm 6 --oem 1")
        _log.debug(f"OCR 提取文本：{text[:100]}...")
        return text.strip() if text.strip() else None
    except Exception as e:
        _log.warning(f"OCR 提取失败：{e}")
        return None

# 优化：记录完整请求参数，降级处理 400 错误
async def async_post(session: aiohttp.ClientSession, options: ApiVlmOptions, cache_path: Path, retries: int = 5, backoff_factor: float = 2.0) -> Optional[str]:
    url_str = str(options.url)
    clean_url = re.sub(r'[^\w\-]', '_', url_str)
    cache_key = f"{clean_url}_{options.params['model']}_{hash(str(options.params['messages']))}"
    cache_file = cache_path / f"{cache_key}.json"
    
    cache_path.mkdir(exist_ok=True)
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                markdown_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not markdown_content:
                    _log.warning(f"缓存文件 {cache_file} 包含空内容，将重新请求")
                else:
                    _log.debug(f"从缓存加载响应：{cache_file}")
                    # 过滤缓存中的提示
                    markdown_content = "\n".join(
                        line for line in markdown_content.split("\n")
                        if not any(keyword in line for keyword in ["以下是", "注：", "表格内容", "图像内容", "如需后续"])
                    ).strip()
                    return markdown_content
        except Exception as e:
            _log.error(f"无法读取缓存文件 {cache_file}：{e}，将重新请求")

    _log.debug(f"准备发起 API 请求，URL：{url_str}")
    params_copy = options.params.copy()
    if "messages" in params_copy and params_copy["messages"]:
        for msg in params_copy["messages"]:
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image_url" and item.get("image_url", {}).get("url"):
                        item["image_url"]["url"] = item["image_url"]["url"][:50] + "... (截断)"
    _log.debug(f"API 请求参数：{json.dumps(params_copy, ensure_ascii=False, indent=2)}")
    
    for attempt in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=max(options.timeout, 120), connect=15, sock_connect=15)
            async with session.post(url_str, json=options.params, headers=options.headers, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                _log.debug(f"API 响应状态码：{response.status}，内容长度：{len(str(data))} 字节")
                markdown_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not markdown_content:
                    _log.error(f"API 返回空内容，响应数据：{data}")
                    return None
                # 过滤 API 响应中的提示
                markdown_content = "\n".join(
                    line for line in markdown_content.split("\n")
                    if not any(keyword in line for keyword in line["以下", "是", "注：", "表格内容", "图像内容", "如需续"])
                ).strip()
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    _log.debug(f"响应已缓存至：{cache_file}")
                except Exception as e:
                    _log.warning(f"无法写入缓存文件 {cache_file}：{e}")
                return markdown_content
        except aiohttp.ClientResponseError as e:
            _log.error(f"异步请求错误（尝试 {attempt + 1}/{retries}）：{e.status}, message='{e.message}', url='{e.request_info.url}'")
            try:
                error_text = await response.text()
                _log.error(f"API 错误详情：{error_text}")
                if response.headers:
                    _log.debug(f"API 响应头：{response.headers}")
            except Exception as err:
                _log.error(f"无法读取错误详情：{err}")
            if e.status == 400 and attempt < retries - 1:
                _log.debug(f"HTTP 400 错误，降低 max_tokens 后重试")
                options.params["max_tokens"] = max(options.params["max_tokens"] // 2, 4096)
                await asyncio.sleep(backoff_factor ** attempt)
            elif attempt < retries - 1:
                sleep_time = backoff_factor ** attempt
                _log.debug(f"等待 {sleep_time} 秒后重试")
                await asyncio.sleep(sleep_time)
            else:
                _log.error(f"API 请求失败 {retries} 次，放弃重试")
                return None
        except aiohttp.ClientError as e:
            _log.error(f"异步请求错误（尝试 {attempt + 1}/{retries}）：{e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor ** attempt
                _log.debug(f"等待 {sleep_time} 秒后重试")
                await asyncio.sleep(sleep_time)
            else:
                _log.error(f"API 请求失败 {retries} 次，放弃重试")
                return None
        except Exception as e:
            _log.error(f"API 请求失败（尝试 {attempt + 1}/{retries}）：{e}")
            return None

def get_adaptive_prompt(input_path: Path) -> str:
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        return "将 Excel 转换为 Markdown，保留表格结构，输出纯 Markdown 内容，无额外说明。"
    elif input_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        return "将图像转换为 Markdown，提取文本和手写签名（若存在，格式为 <签名: XXX>），输出纯 Markdown 内容，无额外说明。"
    else:  # PDF
        doc = fitz.open(input_path)
        page_count = len(doc)
        has_images = any(len(page.get_images()) > 0 for page in doc)
        doc.close()
        if page_count > 50:
            return "将 PDF 转换为 Markdown，保留标题、表格和正文，分段清晰，忽略页眉页脚，输出纯 Markdown 内容，无额外说明。"
        elif has_images:
            return "将 PDF 转换为 Markdown，保留文本、表格和图像内容，手写签名格式为 <签名: XXX>，输出纯 Markdown 内容，无额外说明。"
        return "将 PDF 转换为 Markdown，保留所有文本、表格和结构，输出纯 Markdown 内容，无额外说明。"

# 优化：验证 messages 格式
def blendapi_vlm_options(model: str, prompt: str, input_path: Path = None, api_key: str = None, max_tokens: int = 16000):
    api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT", "https://api.blendapi.com/v1/chat/completions")
    api_key = api_key or os.getenv("BLENDAPI_API_KEY")
    max_file_size = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))
    max_base64_size = 15 * 1024 * 1024
    allowed_file_types = [".pdf", ".jpg", ".jpeg", ".png", ".xlsx", ".xls"]

    if not api_endpoint or not isinstance(api_endpoint, str):
        raise ValueError(f"无效的 BlendAPI 端点：{api_endpoint}")
    if not api_key:
        raise ValueError("缺少 BlendAPI API 密钥")

    if not hasattr(blendapi_vlm_options, 'logged'):
        masked_key = api_key[:4] + "****" + api_key[-4:] if api_key else "None"
        _log.debug(f"BLENDAPI_API_ENDPOINT: {api_endpoint}")
        _log.debug(f"BLENDAPI_API_KEY: {masked_key}")
        blendapi_vlm_options.logged = True

    content = [{"type": "text", "text": prompt}]
    if input_path and input_path.exists():
        if input_path.suffix.lower() not in allowed_file_types:
            raise ValueError(f"不支持的文件类型：{input_path.suffix}")
        if input_path.stat().st_size > max_file_size:
            raise ValueError(f"文件 {input_path} 超过最大大小：{max_file_size} 字节")

        if input_path.suffix.lower() in [".xlsx", ".xls"]:
            excel_content = excel_to_text(input_path)
            content = [{"type": "text", "text": f"{prompt}\n\n以下是 Excel 内容：\n{excel_content}"}]
        else:
            temp_jpeg_path = Path("./temp_page.jpg")
            if not convert_to_jpeg(input_path, temp_jpeg_path):
                raise ValueError(f"无法将 {input_path} 转换为 JPEG")
            try:
                with Image.open(temp_jpeg_path) as img:
                    img = img.convert("RGB")
                    buffered = io.BytesIO()
                    quality = 80
                    img.save(buffered, format="JPEG", quality=quality, optimize=True)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    if len(img_base64) > max_base64_size:
                        raise ValueError(f"Base64 图像大小 ({len(img_base64)} 字节) 超过 API 限制 ({max_base64_size} 字节)")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "high"
                        }
                    })
                    _log.debug(f"图像 {temp_jpeg_path} 已编码为 Base64，大小：{len(img_base64)} 字节")
            finally:
                if temp_jpeg_path.exists():
                    temp_jpeg_path.unlink()

    base_timeout = 120
    image_timeout = 60 if input_path and input_path.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"] else 0
    timeout_value = min(base_timeout + image_timeout + max_tokens // 500, 1200)
    _log.debug(f"计算的 max_tokens：{max_tokens}，超时时间：{timeout_value}秒")
    options = ApiVlmOptions(
        url=str(api_endpoint),
        params=dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        ),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        prompt=prompt,
        timeout=timeout_value,
        response_format=ResponseFormat.MARKDOWN,
    )
    if not options.params["messages"] or not all("content" in msg for msg in options.params["messages"]):
        _log.error(f"无效的 messages 结构：{options.params['messages']}")
        raise ValueError("API 请求参数中 messages 格式无效")
    return options, None

async def process_pdf_page(session: aiohttp.ClientSession, input_path: Path, page_num: int, options: ApiVlmOptions, cache_path: Path) -> Optional[str]:
    temp_jpeg_path = Path(f"./temp_page_{page_num}.jpg")
    try:
        if not pdf_to_jpeg(input_path, temp_jpeg_path, page_num):
            _log.error(f"无法将页面 {page_num} 转换为 JPEG")
            return None
        with Image.open(temp_jpeg_path) as img:
            img = img.convert("RGB")
            ocr_text = extract_image_text(img)
            buffered = io.BytesIO()
            quality = 80
            img.save(buffered, format="JPEG", quality=quality, optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            if len(img_base64) < 5000:
                _log.warning(f"页面 {page_num} Base64 大小 ({len(img_base64)} 字节) 过小，尝试无裁剪重试")
                doc = fitz.open(input_path)
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=70, optimize=True)
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                doc.close()
            with open(f"debug_page_{page_num}.jpg", "wb") as f:
                f.write(base64.b64decode(img_base64))
            _log.debug(f"调试图像已保存至 debug_page_{page_num}.jpg")
            if len(img_base64) < 1000:
                _log.error(f"页面 {page_num} 的 Base64 图像大小 ({len(img_base64)} 字节) 过小")
                return None
            options.params["messages"][0]["content"] = [
                {"type": "text", "text": options.prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                }
            ]
        markdown_content = await async_post(session, options, cache_path)
        if markdown_content:
            # 只在检测到有效 OCR 文本时替换签名
            if ocr_text:
                markdown_content = re.sub(r"!\[签名\d+\]", f"<签名: {ocr_text}>", markdown_content)
            # 过滤多余提示
            markdown_content = "\n".join(
                line for line in markdown_content.split("\n")
                if not any(keyword in line for keyword in ["以下是", "注：", "表格内容", "图像内容", "如需后续"])
            ).strip()
        return normalize_markdown_headers(markdown_content)
    except Exception as e:
        _log.error(f"处理页面 {page_num} 失败：{e}")
        return None
    finally:
        if temp_jpeg_path.exists():
            temp_jpeg_path.unlink()

async def process_pdf_async(input_path: Path, pipeline_options: VlmPipelineOptions, cache_path: Path) -> str:
    doc = fitz.open(input_path)
    tasks = []
    semaphore = Semaphore(5)
    async with aiohttp.ClientSession() as session:
        for page_num in range(len(doc)):
            async def process_with_semaphore(page_num: int):
                async with semaphore:
                    options, _ = blendapi_vlm_options(
                        model=pipeline_options.vlm_options.params["model"],
                        prompt=pipeline_options.vlm_options.prompt,
                        api_key=pipeline_options.vlm_options.headers["Authorization"].split("Bearer ")[1],
                        max_tokens=estimate_page_tokens(doc[page_num]),
                    )
                    result = await process_pdf_page(session, input_path, page_num, options, cache_path)
                    if result:
                        with open(cache_path / f"page_{page_num}.md", "w", encoding="utf-8") as f:
                            f.write(result)
                    return result
            tasks.append(process_with_semaphore(page_num))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    doc.close()
    markdown_content = []
    errors = []
    for page_num, result in enumerate(results):
        if isinstance(result, str) and result:
            markdown_content.append(result)
        elif isinstance(result, Exception):
            _log.error(f"页面 {page_num} 处理失败：{result}")
            errors.append(f"页面 {page_num}: {result}")
        else:
            _log.error(f"页面 {page_num} 处理结果无效：{result}")
            errors.append(f"页面 {page_num}: 无效结果")
    
    if not markdown_content:
        _log.error(f"所有页面处理失败：{'; '.join(errors)}")
        raise ValueError(f"所有页面处理失败，错误详情：{'; '.join(errors)}")
    
    return normalize_markdown_headers("\n\n".join(markdown_content))

# 优化：限制 token 上限为 32,768
def estimate_page_tokens(page: fitz.Page) -> int:
    text = page.get_text()
    images = page.get_images()
    return min((len(text) // 3) + (len(images) * 500), 32768)  # 限制上限

async def process_non_pdf_async(input_path: Path, options: ApiVlmOptions, cache_path: Path) -> str:
    async with aiohttp.ClientSession() as session:
        markdown_content = await async_post(session, options, cache_path)
    return normalize_markdown_headers(markdown_content)

def main():
    parser = argparse.ArgumentParser(description="使用 Docling 和 BlendAPI 解析文档为 Markdown。")
    parser.add_argument("input_path", type=str, help="输入文件路径")
    args = parser.parse_args()

    input_doc_path = Path(unicodedata.normalize('NFC', args.input_path))
    if not input_doc_path.exists():
        _log.error(f"输入文件 {input_doc_path} 不存在")
        raise FileNotFoundError(f"输入文件 {input_doc_path} 不存在")
    allowed_file_types = [".pdf", ".jpg", ".jpeg", ".png", ".xlsx", ".xls"]
    if input_doc_path.suffix.lower() not in allowed_file_types:
        _log.error(f"输入文件 {input_doc_path} 必须是 PDF、JPEG/PNG 或 Excel 文件")
        raise ValueError(f"输入文件 {input_doc_path} 必须是 PDF、JPEG/PNG 或 Excel 文件")

    doc_name = input_doc_path.stem
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    cache_path = Path("cache")
    cache_path.mkdir(exist_ok=True)

    load_dotenv(dotenv_path=Path(".env"), override=True)
    api_key = os.environ.get("BLENDAPI_API_KEY")
    if not api_key:
        _log.error("缺少 BLENDAPI_API_KEY，请在 .env 文件中配置")
        raise ValueError("缺少 BLENDAPI_API_KEY")

    prompt = get_adaptive_prompt(input_doc_path)
    model = "gpt-4.1"
    max_tokens = estimate_max_tokens(input_doc_path)

    if input_doc_path.suffix.lower() in [".xlsx", ".xls", ".jpg", ".jpeg", ".png"]:
        options, _ = blendapi_vlm_options(
            model=model,
            prompt=prompt,
            input_path=input_doc_path,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        try:
            start_time = time.time()
            markdown_content = asyncio.run(process_non_pdf_async(input_doc_path, options, cache_path))
            output_file = Path(f"{doc_name}_{current_time}.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            _log.info(f"Markdown 内容已保存至 {output_file}，耗时：{time.time() - start_time} 秒")
        except aiohttp.ClientError as e:
            _log.error(f"请求错误：{e}")
            raise
        return

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,
        force_backend_text=False,
    )
    options, _ = blendapi_vlm_options(
        model=model,
        prompt=prompt,
        input_path=input_doc_path,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    pipeline_options.vlm_options = options
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, pipeline_cls=VlmPipeline)
        }
    )

    max_retries = 3
    retry_delay = 5
    retries = 0
    current_delay = retry_delay

    try:
        while retries < max_retries:
            try:
                start_time = time.time()
                result = doc_converter.convert(input_doc_path)
                markdown_content = result.document.export_to_markdown()
                output_file = Path(f"{doc_name}_{current_time}.md")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                _log.info(f"Markdown 内容已保存至 {output_file}，耗时：{time.time() - start_time} 秒")
                break
            except aiohttp.ClientError as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                    _log.warning(f"429 请求过多，将在 {current_delay} 秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1
                else:
                    _log.error(f"请求错误：{e}")
                    raise
            except Exception as e:
                _log.error(f"转换错误：{e}")
                raise
        else:
            raise Exception(f"重试 {max_retries} 次后仍无法转换文档")
    finally:
        pass

if __name__ == "__main__":
    main()