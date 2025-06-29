import json
import logging
import os
from pathlib import Path

import yaml

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

def main():
    input_paths = [Path("README.md")]
    # input_doc = Path("./tests/data/pdf/2206.01062.pdf") 
    # input_doc = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")   
    # input_paths = [Path("/Users/wingzheng/Downloads/解析结果评测/测试集收集/olmocr 测试文档集2- 精心收集：连续性+签名+多列等/签名/（脱敏版）测试1财务表格6签名页面：税友招股书_（去除打印的名字）.pdf")] 

    # 定义输出目录
    out_path = Path("scratch")
    # 确保输出目录存在
    out_path.mkdir(exist_ok=True)

    for path in input_paths:
        in_doc = InputDocument(
            path_or_stream=path,
            # format=InputFormat.PDF,     # 错误！❌，并不支持PDF格式的输入
            format=InputFormat.MD,  # 仅支持md格式的输入
            backend=MarkdownDocumentBackend,
        )
        mdb = MarkdownDocumentBackend(in_doc=in_doc, path_or_stream=path)
        document = mdb.convert()

        print(f"Document {path} converted.\nSaved markdown output to: {out_path!s}")

        # 获取文件名（不含扩展名）以避免重复 .md
        fn = os.path.splitext(os.path.basename(path))[0]

        # 三种输出格式：md JSON YAML
        # 写入 Markdown 文件
        with (out_path / f"{fn}.md").open("w") as fp:
            fp.write(document.export_to_markdown())

        # 写入 JSON 文件
        with (out_path / f"{fn}.json").open("w") as fp:
            fp.write(json.dumps(document.export_to_dict()))

        # 写入 YAML 文件
        with (out_path / f"{fn}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(document.export_to_dict()))

if __name__ == "__main__":
    main()