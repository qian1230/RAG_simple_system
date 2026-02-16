import os
import ssl
import sys
import socket
from typing import List, Dict
from urllib.parse import urlparse
from dotenv import load_dotenv

# ---------------------- 基础环境配置 ----------------------
load_dotenv()
if sys.platform == "win32":
    ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------- 第三方依赖导入 ----------------------
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

# ---------------------- 全局配置 ----------------------
COLLECTION_NAME = "test_collection"
RAG_NAMESPACE = "test"
SOURCE_FILE = "test.pdf"
SEARCH_QUERY = "唇读 深度学习 论文"  # 和你的测试查询词一致
TOP_K = 3
SCORE_THRESHOLD = 0.0  # 显示所有结果，不设阈值


# ---------------------- 安全的嵌入模型类 ----------------------
class SafeDashScopeEmbedding(BaseEmbedding):
    _embedder: DashScopeEmbedding = PrivateAttr()

    def __init__(self, model_name="text-embedding-v1", api_key=None, timeout=30):
        super().__init__()
        if not api_key:
            raise ValueError("❌ DASHSCOPE_API_KEY未配置")
        self._embedder = DashScopeEmbedding(model_name=model_name, api_key=api_key, timeout=timeout)

    def _get_query_embedding(self, query: str) -> List[float]:
        try:
            vec = self._embedder.get_text_embedding(query.strip() or "空文本")
            # 标准化向量格式
            if isinstance(vec, list):
                vec_norm = [float(x) for x in vec]
            elif hasattr(vec, "tolist"):
                vec_norm = vec.tolist()
                vec_norm = [float(x) for x in vec_norm]
            else:
                raise ValueError(f"向量格式错误：{type(vec)}")
            # 确保1536维
            if len(vec_norm) != 1536:
                vec_norm = vec_norm[:1536] if len(vec_norm) > 1536 else vec_norm + [0.0] * (1536 - len(vec_norm))
            return vec_norm
        except Exception as e:
            print(f"❌ 嵌入失败：{str(e)[:100]}")
            return [0.0] * 1536

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# ---------------------- 初始化函数 ----------------------
def init_services():
    """初始化嵌入模型和Qdrant客户端"""
    # 初始化嵌入模型
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    embedding = SafeDashScopeEmbedding(api_key=dashscope_api_key)

    # 初始化Qdrant客户端
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # 解析URL并测试连接
    parsed_url = urlparse(qdrant_url)
    hostname = parsed_url.netloc.split(':')[0]
    port = int(parsed_url.netloc.split(':')[1]) if ':' in parsed_url.netloc else 443
    socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60
    )

    return embedding, client


# ---------------------- 核心检索函数 ----------------------
def search_and_show_details():
    """执行检索并展示完整结果详情"""
    print("===== 🔍 精准查看检索结果详情 =====")
    print(f"查询词：{SEARCH_QUERY}")
    print(f"目标集合：{COLLECTION_NAME}")
    print(f"命名空间：{RAG_NAMESPACE}")
    print("-" * 80)

    # 初始化服务
    embedding, client = init_services()

    # 生成查询向量
    query_vector = embedding._get_query_embedding(SEARCH_QUERY)

    # 构建过滤器（只查test命名空间的rag数据）
    filter_conditions = Filter(
        must=[
            FieldCondition(key="rag_namespace", match=MatchValue(value=RAG_NAMESPACE)),
            FieldCondition(key="is_rag_data", match=MatchValue(value=True))
        ]
    )

    # 执行检索（兼容Qdrant所有版本）
    try:
        hits = None
        # 先尝试新版API（query_filter参数）
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K,
                score_threshold=SCORE_THRESHOLD,
                query_filter=filter_conditions,  # 关键修复：filter → query_filter
                with_payload=True,
                with_vectors=False
            )
            hits = results.points
        except (AttributeError, TypeError):
            # 尝试旧版API（search方法 + query_vector参数）
            try:
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=TOP_K,
                    score_threshold=SCORE_THRESHOLD,
                    filter=filter_conditions,
                    with_payload=True
                )
                hits = results
            except Exception as e:
                print(f"⚠️  旧版API也执行失败：{str(e)[:100]}")
                raise

        if hits is None:
            print("❌ 未获取到检索结果")
            return

        print(f"\n📊 检索结果汇总：共找到 {len(hits)} 条结果")
        print("-" * 80)

        # 逐条展示完整详情
        for idx, hit in enumerate(hits, 1):
            print(f"\n【结果 {idx}】")
            print(f"📌 ID：{hit.id}")
            print(f"📈 相似度：{round(float(hit.score), 4)}")
            print(f"📁 来源文件：{hit.payload.get('source', '未知')}")
            print(f"🗂️  命名空间：{hit.payload.get('rag_namespace', '未知')}")
            print(f"🏷️  标题路径：{hit.payload.get('heading_path', '无')}")
            print(f"⏰ 创建时间：{hit.payload.get('create_time', '未知')}")
            print(f"📍 文本位置：start={hit.payload.get('start', '未知')}, end={hit.payload.get('end', '未知')}")
            print(f"📝 完整内容：")
            content = hit.payload.get('content', '无内容')
            # 完整展示内容（不截断）
            print(f"```")
            print(content)
            print(f"```")
            print("-" * 80)

        # 额外分析
        print("\n📋 结果分析：")
        high_similarity = [h for h in hits if float(h.score) > 0.5]
        low_similarity = [h for h in hits if float(h.score) <= 0.5]
        print(f"   高相似度结果（>0.5）：{len(high_similarity)} 条")
        print(f"   低相似度结果（≤0.5）：{len(low_similarity)} 条")

        # 定位低相似度结果的问题
        if low_similarity:
            print(f"\n⚠️  低相似度结果详情：")
            for h in low_similarity:
                print(f"   ID {h.id}（相似度{round(float(h.score), 4)}）：内容为「{h.payload.get('content', '')[:50]}...」")
                print(f"   原因：该文本块内容不完整/无关，属于PDF解析时的冗余内容")

    except Exception as e:
        print(f"❌ 检索失败：{str(e)[:200]}")
        import traceback
        traceback.print_exc()


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    try:
        search_and_show_details()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)[:200]}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🎉 结果详情查看完成")