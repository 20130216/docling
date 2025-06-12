from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from datetime import datetime
import logging

def main():
    
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("debug.log")
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.debug("开始设置输入文档路径")    
    # input_doc = Path("./tests/data/pdf/2206.01062.pdf") #  已测成功
    # input_doc = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")   # 已测成功
    input_doc = Path("/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/签名/（脱敏版）测试1财务表格6签名页面：税友招股书_（去除打印的名字）.pdf")  # 效果很差 完全是接近于乱码的文字；签名部分是<!-- image -->
    logging.debug("输入文档路径设置完成: %s", input_doc)    

    logging.debug("开始配置管道选项")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    logging.debug("管道选项配置完成")

    logging.debug("开始设置OCR选项")
    # Any of the OCR options can be used:EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions(Mac only), RapidOcrOptions
    ocr_options = EasyOcrOptions(force_full_page_ocr=True) # 签名页 比OcrMacOptions还差 不忍卒看 直接被我放弃比较，接近0
    # ocr_options = TesseractOcrOptions(force_full_page_ocr=True) # 一直安装不了，放弃！
    # ocr_options = OcrMacOptions(force_full_page_ocr=True)  # 签名页 比TesseractCliOcrOptions效果还差 评测28分
    # ocr_options = RapidOcrOptions(force_full_page_ocr=True)  # 效果页很差 但评分却评测为77分；实际目测50分左右
    # ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True) # 签名页，只评测为49.5分
    pipeline_options.ocr_options = ocr_options
    logging.debug("OCR选项设置完成")

    logging.debug("开始初始化文档转换器")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    logging.debug("文档转换器初始化完成")

    logging.debug("开始转换文档")
    doc = converter.convert(input_doc).document
    logging.debug("文档转换完成")
    
    logging.debug("开始导出为Markdown")
    md = doc.export_to_markdown()
    # print(md)
    
    # 生成输出文件名，格式：文件名_YYYYMMDD_HHMM.md
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = input_doc.stem + f"_{timestamp}.md"
    # output_path = input_doc.parent / output_filename #输出到文件输入的位置
    output_path = Path.cwd() / output_filename

    # 写入Markdown到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)    
    print(f"输出文件已生成: {output_path}")

if __name__ == "__main__":
    main()
