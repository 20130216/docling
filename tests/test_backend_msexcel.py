import logging
import os
from pathlib import Path
import pytest
import sys

from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter
from test_data_gen_flag import GEN_TEST_DATA
from verify_utils import verify_document, verify_export

from datetime import datetime

_log = logging.getLogger(__name__)
GENERATE = GEN_TEST_DATA

def get_xlsx_paths():
    directory = Path("/Users/wingzheng/Downloads/解析结果评测/测试集收集/dify-rag-test-excel/岗位职责")
    pdf_files = sorted(directory.rglob("*.xlsx"))  # 变量名笔误，实际为 XLSX 文件
    return pdf_files

def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.XLSX])
    return converter

@pytest.fixture(scope="module")
def documents() -> list[tuple[Path, DoclingDocument]]:
    documents: list[tuple[Path, DoclingDocument]] = []
    xlsx_paths = get_xlsx_paths()
    converter = get_converter()

    for xlsx_path in xlsx_paths:
        _log.debug(f"converting {xlsx_path}")
        gt_path = xlsx_path.parent.parent / "groundtruth" / "docling_v2" / xlsx_path.name
        conv_result: ConversionResult = converter.convert(xlsx_path)
        doc: DoclingDocument = conv_result.document
        assert doc, f"Failed to convert document from file {gt_path}"
        documents.append((gt_path, doc))
    return documents

def test_e2e_xlsx_conversions(documents) -> None:
    for gt_path, doc in documents:
        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"
        pred_itxt: str = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt"), "export to indented-text"
        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "document document"

def test_pages(documents) -> None:
    path = next(item for item in get_xlsx_paths() if item.stem == "test-01")
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.XLSX,
        filename=path.stem,
        backend=MsExcelDocumentBackend,
    )
    backend = MsExcelDocumentBackend(in_doc=in_doc, path_or_stream=path)
    assert backend.page_count() == 3
    doc = next(item for path, item in documents if path.stem == "test-01")
    assert len(doc.pages) == 3
    assert doc.pages.get(1).size.as_tuple() == (3.0, 7.0)
    assert doc.pages.get(2).size.as_tuple() == (9.0, 18.0)
    assert doc.pages.get(3).size.as_tuple() == (13.0, 36.0)

def pytest_addoption(parser):
    parser.addoption(
        "--input-path",
        action="store",
        default=None,
        help="Path to XLSX file or directory containing XLSX files"
    )

def convert_xlsx_to_markdown(input_path: str, base_dir: str = None) -> None:
    """
    将单个 XLSX 文件或包含 XLSX 文件的文件夹（包括子文件夹）转换为 Markdown 文件。
    
    Args:
        input_path (str): 单个 XLSX 文件名或文件夹名（相对或绝对路径）。
        base_dir (str, optional): 基础目录，用于解析相对路径。如果为 None，使用当前工作目录。
    
    Notes:
        - 单个 XLSX 文件：生成 <文件名>_YYYYMMDDHHMM.md，保存在同一目录。
        - 文件夹：生成 <文件夹名>_md_YYYYMMDDHHMM 目录，保留子文件夹结构。
        - 支持递归解析子文件夹中的 XLSX 文件。
    """
    _log.info(f"Starting conversion for input: {input_path}")
    base_dir = Path(base_dir or os.getcwd())
    input_path = Path(input_path)
    
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    
    _log.info(f"Resolved input path: {input_path}")
    if not input_path.exists():
        _log.error(f"Input path does not exist: {input_path}")
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    converter = DocumentConverter(allowed_formats=[InputFormat.XLSX])
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    
    if input_path.is_file() and input_path.suffix.lower() == ".xlsx":
        _log.info(f"Converting single file: {input_path}")
        try:
            conv_result = converter.convert(input_path)
            doc = conv_result.document
            if not doc:
                _log.error(f"Failed to convert {input_path}")
                return
            _log.debug(f"Document converted: {doc}")
            markdown_content = doc.export_to_markdown()
            output_file = input_path.with_stem(f"{input_path.stem}_{timestamp}").with_suffix(".md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            _log.info(f"Markdown text parsed and saved to {output_file}")  # 修改日志
        except Exception as e:
            _log.error(f"Error converting {input_path}: {str(e)}")
            raise
        return
    
    if input_path.is_dir():
        _log.info(f"Converting directory: {input_path}")
        output_dir = input_path.parent / f"{input_path.name}_md_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Output directory created: {output_dir}")
        
        xlsx_files = sorted(input_path.rglob("*.xlsx"))
        xlsx_files = [f for f in xlsx_files if not f.name.startswith(".~")]  # 新增过滤
        _log.info(f"Found {len(xlsx_files)} valid XLSX files in {input_path} (including subdirectories)")
        if not xlsx_files:
            _log.warning(f"No valid XLSX files found in {input_path}")
            return
        
        for xlsx_path in xlsx_files:
            _log.info(f"Converting {xlsx_path}")
            try:
                conv_result = converter.convert(xlsx_path)
                doc = conv_result.document
                if not doc:
                    _log.error(f"Failed to convert {xlsx_path}")
                    continue
                markdown_content = doc.export_to_markdown()
                relative_path = xlsx_path.relative_to(input_path)
                output_file = output_dir / relative_path.with_stem(f"{relative_path.stem}_{timestamp}").with_suffix(".md")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                _log.info(f"Markdown text parsed and saved to {output_file}")
            except Exception as e:
                _log.error(f"Error converting {xlsx_path}: {str(e)}")
                continue
        return
    
    _log.error(f"Invalid input path: {input_path}")
    raise ValueError(f"Input path must be an XLSX file or directory: {input_path}")

@pytest.mark.parametrize("input_path", [pytest.param(None, id="convert-xlsx")])
def test_convert_xlsx_to_markdown(request, input_path):
    input_path = request.config.getoption("--input-path")
    if not input_path:
        pytest.skip("No input path provided. Use --input-path <xlsx_file_or_directory>")
    _log.info(f"Running test_convert_xlsx_to_markdown with input: {input_path}")
    convert_xlsx_to_markdown(input_path)

if __name__ == "__main__":
    print("sys.argv:", sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python tests/test_backend_msexcel.py <xlsx_file_or_directory>")
        sys.exit(1)
    
    input_path = ' '.join(sys.argv[1:])
    print("Input path:", input_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")  # 新增时间戳
    output_path = (Path(input_path).with_stem(f"{Path(input_path).stem}_{timestamp}").with_suffix(".md") 
                   if Path(input_path).suffix.lower() == ".xlsx" 
                   else Path(input_path).parent / f"{Path(input_path).name}_md_{timestamp}")  # 新增输出路径
    print("Output path:", output_path)  # 新增日志
    _log.info(f"Invoking convert_xlsx_to_markdown with path: {input_path}")
    try:
        convert_xlsx_to_markdown(input_path)
    except Exception as e:
        _log.error(f"Failed to convert: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)