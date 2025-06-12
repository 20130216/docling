import os
from pydantic import TypeAdapter
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

# 加载 .env 文件
env_path = Path("/Users/wingzheng/Desktop/github/ParseDoc/docling/.env")
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")
load_dotenv(dotenv_path=env_path, override=True)
print("Loaded .env:", dotenv_values(env_path))

# 获取 BlendAPI 配置
api_endpoint = os.getenv("BLENDAPI_API_ENDPOINT", "https://api.blendapi.com/v1/chat/completions")
api_key = os.getenv("BLENDAPI_API_KEY")
DOCLING_GEN_TEST_DATA = os.getenv("DOCLING_GEN_TEST_DATA")

# 调试输出
print(f"DEBUG--DOCLING_GEN_TEST_DATA: {DOCLING_GEN_TEST_DATA}")
GEN_TEST_DATA = TypeAdapter(bool).validate_python(os.getenv("DOCLING_GEN_TEST_DATA", 0))
print(f"DEBUG--GEN_TEST_DATA: {GEN_TEST_DATA}, type: {type(GEN_TEST_DATA)}")
print(f"DEBUG--BLENDAPI_API_ENDPOINT: {api_endpoint}")
print(f"DEBUG--BLENDAPI_API_KEY: {api_key}")

# 验证环境变量
if not api_endpoint or not api_endpoint.startswith(("http://", "https://")):
    raise ValueError(f"Invalid BlendAPI endpoint: {api_endpoint}")
if not api_key:
    raise ValueError("BlendAPI API key is missing")

def test_gen_test_data_flag():
    print(f"DEBUG--GEN_TEST_DATA in test: {GEN_TEST_DATA}")
    assert not GEN_TEST_DATA  # 根据需求调整
    
if __name__ == "__main__":
    test_gen_test_data_flag()    