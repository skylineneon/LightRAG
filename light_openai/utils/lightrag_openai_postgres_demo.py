import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
import PyPDF2
from typing import List
# from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens"
# 加载env文件
from dotenv import load_dotenv

# 指定.env文件的路径
dotenv_path = "/DATA/LLM/gaojiale/workspace/LightRAG/light_openai/.env"
load_dotenv(dotenv_path)


def configure_logging():
    """Configure logging for the application"""

    # 重置现有的日志处理器和过滤器，确保日志配置是干净的，避免之前的配置影响当前配置
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # 从环境变量中获取日志目录路径，如果没有设置则使用当前工作目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")  # 拼接路径后转为绝对路径
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量中获取日志文件的最大大小和备份数量，如果没有设置则使用默认值
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(  # 配置日志
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {  #  定义日志格式，分为默认格式和详细格式
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {  # 包含时间、日志名称、日志级别和日志消息
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(
        logging.INFO
    )  # 将日志器的级别设置为 INFO ，表示只记录 INFO 及以上级别的日志
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
os.environ["AGE_GRAPH_NAME"] = "dickens"

os.environ["POSTGRES_HOST"] = "192.168.0.147"
os.environ["POSTGRES_PORT"] = "8994"
os.environ["POSTGRES_USER"] = "root"
os.environ["POSTGRES_PASSWORD"] = "helloworld123"
os.environ["POSTGRES_DATABASE"] = "lightrag"

async def llm_model_func(
    # 定义一个异步函数 llm_model_func，用于调用 OpenAI 的聊天模型
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        **kwargs,
    )


# async def embedding_func(texts: list[str]) -> np.ndarray:
#     return await ollama_embed(
#         texts=texts,
#         embed_model="bge-m3:latest",
#         host="http://m4.lan.znipower.com:11434",
#     )
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)
# asyncio.run(test_funcs())


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        auto_manage_storages_states=False,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
async def query(rag: LightRAG):
        """
        LightRAG 提供了多种查询模式，适用于不同的检索需求：
        1. 朴素模式（Naive）：直接从知识库中提取最相关的内容，适合简单问题。
        2. 局部模式（Local）：仅检索与输入问题直接相关的区域，快速定位具体答案。
        3. 全局模式（Global）：基于知识图谱的关系扩展，提供更全面的上下文信息。
        4. 混合模式（Hybrid）：结合局部与全局检索，平衡精度与广度。
        """
        query = "这个故事主题是什么？用中文回答"
    # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        resp = await rag.aquery(query=query,
            param=QueryParam(mode="naive", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        resp = await rag.aquery(query=query,
            param=QueryParam(mode="local", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        resp = await rag.aquery(query=query,
            param=QueryParam(mode="global", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        resp = await rag.aquery(query=query,
            param=QueryParam(mode="hybrid", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)
async def insert_multi_txt(rag: LightRAG):
    from pathlib import Path
    files_path = Path(os.getenv("MULTI_FILE_PATH"))
    contents=[]
    path=[]
    if not files_path.exists():
        print(f"File not found: {files_path}")
        return 
    for file_path in files_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        path.append(file_path)   
        contents.append(content)
    await rag.ainsert(contents,file_paths=path)
    print("批量文件构建索引完成")
async def insert_singer_txt(rag: LightRAG):
    file_path =os.getenv("SINGER_FILE_PATH")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 
    with open(file_path, "r", encoding="utf-8") as file:
        await rag.ainsert(file.read())
    print("单个文件构建索引完成")


async def pdf_insert(rag: LightRAG):
    """
    遍历指定文件夹中的 PDF 文件，并将它们的内容插入到图谱中。

    :param rag: LightRAG 实例
    :param folder_path: 包含 PDF 文件的文件夹路径
    """
    try:
        folder_path =os.getenv("MULTI_FILE_PATH")
        # 检查文件夹路径是否存在
        if not os.path.exists(folder_path):
            print("文件夹路径不存在。")
            return
        # 获取文件夹中所有的 PDF 文件
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("未找到 PDF 文件。")
            return

        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            content:  str = ""

            # 打开并读取 PDF 文件内容
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # 提取每一页的内容
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text
            await rag.ainsert(content)
            print("pdf构建完成。")

    except Exception as e:
        print(f"处理 PDF 文件时发生错误: {e}")
async def deltete_entity(rag: LightRAG):
    rag.delete_by_entity("Sun Wukong")
async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func
        # 插入TXT文件
        # await insert_singer_txt(rag)
        # 插入PDF文件
        await pdf_insert(rag)
        # 开始查询
        # await query(rag)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'rag' in locals() and rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
    # asyncio.run(test_funcs())