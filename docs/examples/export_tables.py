import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("debug.log")
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Original input path (commented out)
    # input_doc_path = Path(r"/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/税友财务表格+exel表格生成pdf---测连续性/测试1财务表格1--7合并.pdf")
    
    # New input path
    input_doc_path = Path(r"/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/税友财务表格+exel表格生成pdf---测连续性/九月加班核对.pdf")
    if not input_doc_path.exists():
        _log.error(f"File not found: {input_doc_path}")
        raise FileNotFoundError(f"File not found: {input_doc_path}")

    doc_converter = DocumentConverter()

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    # Output directory: create 'output' folder in current working directory
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    # Generate timestamp for output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Generate output filenames with timestamp
        base_filename = f"{doc_filename}_table-{table_ix + 1}_{timestamp}"
        
        # Save the table as CSV
        element_csv_filename = output_dir / f"{base_filename}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)

        # Save the table as HTML
        element_html_filename = output_dir / f"{base_filename}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

    end_time = time.time() - start_time
    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()