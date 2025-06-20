import json
import logging
import time
from pathlib import Path

import torch

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def main():
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("debug.log")
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # input_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    # input_doc_path = Path("./tests/data/pdf/2206.01062.pdf") 
    # input_doc_path = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")   
    input_doc_path = Path("/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/签名/（脱敏版）测试1财务表格6签名页面：税友招股书_（去除打印的名字）.pdf")  
  

    ###########################################################################

    # The following sections contain a combination of PipelineOptions
    # and PDF Backends for various configurations.
    # Uncomment one section at the time to see the differences in the output.

    # PyPdfium without EasyOCR
    # --------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = False

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # PyPdfium with EasyOCR
    # -----------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # Docling Parse without EasyOCR
    # -------------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with EasyOCR
    # ----------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )
    
    # 配置 EasyOCR 作为 OCR 引擎，支持多语言且灵活
    # 通过 docs/examples/full_page_ocr.py得到的配置启示
    # 验证1:EasyOcrOptions的验证结果：文字清晰，但签名都以图片代替；
    # pipeline_options.ocr_options = EasyOcrOptions(   
    #     force_full_page_ocr=True,  # 强制全页 OCR，适合签名页面
    #     lang=["en", "ch_sim"],  # 支持英语和简体中文，覆盖常见签名页语言
    #     use_gpu=True,  # 启用 GPU 加速（若可用）
    #     confidence_threshold=0.3,  # 降低阈值以捕获更多签名内容
    #     download_enabled=True  # 允许下载模型
    # )
    # 验证2:TesseractCliOcrOptions的验证结果：还不如EasyOcrOptions，文字居然解析成了乱码
    # 虽然grok评价：根据当前的公共数据和社区评估，TesseractCliOcrOptions 似乎对一般 OCR 任务具有最佳效果。
    pipeline_options.ocr_options = TesseractCliOcrOptions(
    force_full_page_ocr=True,
    lang=["eng", "chi_sim"],
    tesseract_cmd="tesseract",
    path=None
    ) # TesseractCliOcrOptions（根据 pipeline_options.py）不支持 GPU 加速，也不包含 use_gpu 参数
    
    # 配置加速器选项，优化性能
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,  # 增加线程数，最大化 CPU 利用率
        device=AcceleratorDevice.AUTO  # 自动选择 CPU/GPU
    )
 
    # Log configured device
    _log.debug(f"Accelerator device configured: {pipeline_options.accelerator_options.device}")

    # Log actual device used by OCR engine
    if isinstance(pipeline_options.ocr_options, EasyOcrOptions) and pipeline_options.ocr_options.use_gpu and torch.cuda.is_available():
        _log.debug(f"OCR device selected: GPU (CUDA, {torch.cuda.get_device_name(0)})")
    else:
        _log.debug(f"OCR device selected: CPU (using {pipeline_options.ocr_options.__class__.__name__})")

    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Docling Parse with EasyOCR (CPU only)
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.ocr_options.use_gpu = False  # <-- set this.
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with Tesseract CLI
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractCliOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # Docling Parse with ocrmac(Mac only)
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = OcrMacOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    ###########################################################################

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())


if __name__ == "__main__":
    main()
