import os
import ssl
import sys
import time
import uuid
import re
import socket
from typing import List, Dict, Optional, Awaitable
from urllib.parse import urlparse
from dotenv import load_dotenv

# ---------------------- 基础环境配置 ----------------------
load_dotenv()
if sys.platform == "win32":
    ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------- 第三方依赖导入 ----------------------
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

# ---------------------- 智能助手核心导入 ----------------------
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools.builtin.rag_tool import RAGTool

# ---------------------- 全局嵌入模型实例 ----------------------
_global_embedder = None


# ---------------------- 修复：完整的安全嵌入模型包装类 ----------------------
class SafeDashScopeEmbedding(BaseEmbedding):
    """安全的DashScope嵌入模型包装类，完全兼容llama-index规范"""
    _embedder: DashScopeEmbedding = PrivateAttr()

    def __init__(
            self,
            model_name: str = "text-embedding-v1",
            api_key: str = None,
            timeout: int = 30,
    ):
        super().__init__()
        if not api_key:
            raise ValueError("API key must be provided for DashScopeEmbedding")

        # 初始化原生DashScope嵌入模型
        self._embedder = DashScopeEmbedding(
            model_name=model_name,
            api_key=api_key,
            timeout=timeout
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        """生成查询向量（同步版，符合BaseEmbedding规范）"""
        try:
            vec = self._embedder.get_text_embedding(query.strip() or "空文本")
            # 格式标准化
            if isinstance(vec, list):
                vec_norm = [float(x) for x in vec]
            elif hasattr(vec, "tolist"):
                vec_norm = vec.tolist()
                vec_norm = [float(x) for x in vec_norm]
            else:
                raise ValueError(f"向量格式错误：{type(vec)}")

            # 维度校验（固定1536维）
            if len(vec_norm) != 1536:
                print(f"⚠️ 向量维度异常：期望1536，实际{len(vec_norm)}，自动修正")
                if len(vec_norm) > 1536:
                    vec_norm = vec_norm[:1536]
                else:
                    vec_norm += [0.0] * (1536 - len(vec_norm))

            # 检查是否全零
            if all(v == 0.0 for v in vec_norm):
                print(f"⚠️ 查询嵌入返回全零向量：{query[:50]}...")

            return vec_norm
        except Exception as e:
            print(f"❌ 查询嵌入失败：{query[:50]}... 错误：{str(e)[:100]}")
            return [0.0] * 1536

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """生成查询向量（异步版，必须实现的抽象方法）"""
        # 同步转异步
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """生成文本向量（同步版，符合BaseEmbedding规范）"""
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """生成文本向量（异步版，必须实现的抽象方法）"""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量（同步版，核心修复）"""
        valid_vectors = []
        for text in texts:
            valid_vectors.append(self._get_text_embedding(text))
        return valid_vectors

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量（异步版，必须实现的抽象方法）"""
        return self._get_text_embeddings(texts)

    # 兼容旧代码的方法
    def encode(self, texts: List[str]) -> List[List[float]]:
        """兼容旧代码的批量嵌入方法"""
        return self._get_text_embeddings(texts)

    def get_text_embedding(self, text: str) -> List[float]:
        """兼容原生DashScopeEmbedding的方法"""
        return self._get_text_embedding(text)


# ---------------------- 辅助函数：基础工具 ----------------------
def _enhanced_pdf_processing(path: str) -> str:
    """增强PDF处理（完整版，支持多页PDF解析+格式清理）"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = ""
        page_count = len(doc)
        # 遍历所有页面，保留页码信息
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            # 清理PDF文本中的多余空格和换行
            page_text = page_text.replace("\n\n", "\n").strip()
            if page_text:
                text += f"=== 第{page_num}页 ===\n{page_text}\n\n"
        doc.close()
        print(f"[RAG] PDF解析成功: {path}，共{page_count}页，提取文本{len(text)}字符")
        return text
    except ImportError:
        print("[ERROR] PyMuPDF未安装！请执行：pip install pymupdf")
        return ""
    except Exception as e:
        print(f"[WARNING] PDF增强处理失败 {path}: {str(e)[:100]}")
        return _fallback_text_reader(path)


def _get_markitdown_instance():
    """模拟MarkItDown实例（兼容非PDF文件）"""

    class MockMarkItDown:
        def convert(self, path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return type('obj', (object,), {"text_content": f.read()})

    return MockMarkItDown()


def _fallback_text_reader(path: str) -> str:
    """降级文本读取器"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"[WARNING] 降级读取失败 {path}: {e}")
        return ""


def get_text_embedder():
    """获取统一嵌入模型（基于DashScope）- 修复核心问题"""
    global _global_embedder
    if _global_embedder is not None:
        return _global_embedder

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("❌ DASHSCOPE_API_KEY 未配置！")

    # 初始化修复后的安全嵌入模型
    embedder = SafeDashScopeEmbedding(
        model_name="text-embedding-v1",
        api_key=dashscope_api_key,
        timeout=30
    )

    _global_embedder = embedder
    return embedder


def embed_query(query: str) -> List[float]:
    """单独的查询嵌入函数"""
    embedder = get_text_embedder()
    try:
        return embedder._get_query_embedding(query)
    except Exception as e:
        print(f"❌ 查询嵌入失败：{query[:50]}... 错误：{e}")
        return [0.0] * 1536


def get_dimension(default_dim: int = 1536) -> int:
    """固定返回DashScope的1536维"""
    return 1536


def _create_default_vector_store(dimension: int = 1536) -> QdrantClient:
    """创建Qdrant客户端"""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("❌ Qdrant配置缺失！")

    print(f"[RAG] 尝试连接Qdrant: {qdrant_url}")

    try:
        # 先测试URL是否可访问
        parsed_url = urlparse(qdrant_url)
        hostname = parsed_url.netloc.split(':')[0]
        port = int(parsed_url.netloc.split(':')[1]) if ':' in parsed_url.netloc else 443

        print(f"[RAG] 解析Qdrant地址: {hostname}:{port}")

        # 测试DNS解析
        try:
            addrinfo = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            print(f"[RAG] DNS解析成功: {addrinfo[0][4]}")
        except socket.gaierror as e:
            print(f"[ERROR] DNS解析失败: {e}")
            print("[INFO] 请检查网络连接和QDRANT_URL配置")
            raise

        # 创建Qdrant客户端
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60  # 加长超时，适配PDF大文件
        )

        # 测试连接
        print("[RAG] 测试Qdrant连接...")
        client.get_collection("test_collection")
        print("[RAG] Qdrant连接成功！")
        return client
    except Exception as e:
        print(f"[ERROR] Qdrant连接失败: {e}")
        print("[INFO] 检查QDRANT_URL和QDRANT_API_KEY配置是否正确")
        print("[INFO] 检查网络连接是否正常")
        raise


def _preprocess_markdown_for_embedding(text: str) -> str:
    """预处理文本提升嵌入质量"""
    # 移除PDF页码标记、多余空格
    text = re.sub(r'=== 第\d+页 ===', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # 保留中英文和基础标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''()（）【】]', '', text)
    return text.strip()


# ---------------------- 辅助函数：查询扩展 ----------------------
def _prompt_mqe(query: str, expansions: int = 2) -> List[str]:
    """多查询扩展（Multi-Query Expansion）"""
    try:
        # 初始化LLM用于生成扩展查询
        llm_config = {
            "api_key": os.getenv("LLM_API_KEY"),
            "model_id": os.getenv("LLM_MODEL_ID", "qwen-turbo"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "timeout": 30
        }
        llm = HelloAgentsLLM(**{k: v for k, v in llm_config.items() if v})

        # MQE提示词
        messages = [{
            "role": "user",
            "content": f"""
            你是一个专业的查询扩展助手。请基于以下原始查询，生成{expansions}个不同的、语义相似的查询语句，用于向量数据库检索。
            要求：
            1. 每个查询保持与原始查询的核心语义一致
            2. 表达方式不同，覆盖不同的关键词和句式
            3. 仅返回查询语句，每行一个，不要添加额外说明
            4. 语言与原始查询保持一致

            原始查询：{query}
            """
        }]

        # 调用LLM生成扩展查询
        response = llm.think(messages, temperature=0.5)
        if not response:
            raise Exception("LLM未返回有效响应")
        
        # 处理响应，确保它是一个字符串
        if isinstance(response, str):
            response_text = response.strip()
        else:
            # 如果是生成器，尝试转换为字符串
            try:
                response_text = ''.join(response).strip()
            except:
                # 如果转换失败，返回原始查询
                raise Exception("无法处理LLM响应")
        
        mqe_queries = [line.strip() for line in response_text.split("\n") if line.strip()]

        # 确保生成数量符合要求
        if len(mqe_queries) < expansions:
            mqe_queries += [query] * (expansions - len(mqe_queries))

        print(f"[MQE] 生成{len(mqe_queries)}个扩展查询：{mqe_queries}")
        return mqe_queries[:expansions]
    except Exception as e:
        print(f"[WARNING] MQE扩展失败：{e}，使用原始查询")
        return [query] * expansions


def _prompt_hyde(query: str) -> str:
    """HYDE（Hypothetical Document Embeddings）"""
    try:
        # 初始化LLM
        llm_config = {
            "api_key": os.getenv("LLM_API_KEY"),
            "model_id": os.getenv("LLM_MODEL_ID", "qwen-turbo"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "timeout": 30
        }
        llm = HelloAgentsLLM(**{k: v for k, v in llm_config.items() if v})

        # HYDE提示词
        messages = [{
            "role": "user",
            "content": f"""
            你是一个专业的文档生成助手。请基于以下查询，生成一段假设的、相关的文档内容（约100-200字）。
            要求：
            1. 内容与查询高度相关，符合该主题的真实文档特征
            2. 语言流畅，结构合理
            3. 仅返回生成的文档内容，不要添加额外说明

            查询：{query}
            """
        }]

        # 调用LLM生成假设文档
        response = llm.think(messages, temperature=0.5)
        if not response:
            raise Exception("LLM未返回有效响应")
        
        # 处理响应，确保它是一个字符串
        if isinstance(response, str):
            hyde_text = response.strip()
        else:
            # 如果是生成器，尝试转换为字符串
            try:
                hyde_text = ''.join(response).strip()
            except:
                # 如果转换失败，返回空字符串
                raise Exception("无法处理LLM响应")
        
        print(f"[HYDE] 生成假设文档：{hyde_text[:100]}...")
        return hyde_text
    except Exception as e:
        print(f"[WARNING] HYDE生成失败：{e}")
        return ""


# ---------------------- 核心函数：增强版向量检索 ----------------------
def search_vectors_expanded(
        store=None,
        query: str = "",
        top_k: int = 8,
        rag_namespace: Optional[str] = None,
        only_rag_data: bool = True,
        score_threshold: Optional[float] = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """增强版检索：支持MQE多查询扩展、HYDE假设文档、多候选池聚合"""
    if not query:
        return []

    # 创建默认存储
    if store is None:
        store = _create_default_vector_store()

    # 1. 查询扩展：基础查询 + MQE + HYDE
    expansions: List[str] = [query]

    # MQE多查询扩展
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))

    # HYDE假设文档扩展
    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    # 去重和修剪扩展查询
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]
    print(f"[RAG] 最终扩展查询列表：{expansions}")

    # 2. 计算候选池大小
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))
    print(f"[RAG] 候选池大小：{pool}，每个查询取{per}条结果")

    # 3. 构建过滤器
    filter_conditions = []
    if only_rag_data:
        filter_conditions.extend([
            FieldCondition(key="is_rag_data", match=MatchValue(value=True)),
            FieldCondition(key="data_source", match=MatchValue(value="rag_pipeline"))
        ])
    if rag_namespace:
        filter_conditions.append(
            FieldCondition(key="rag_namespace", match=MatchValue(value=rag_namespace))
        )

    qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

    # 4. 收集所有扩展查询的检索结果
    agg: Dict[str, Dict] = {}
    for q in expansions:
        # 生成查询向量
        qv = embed_query(q)

        # 适配Qdrant最新版API
        try:
            # 新版API：query_points
            results = store.query_points(
                collection_name="test_collection",
                query=qv,
                limit=per,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            )
            hits = results.points
        except AttributeError:
            # 旧版API：search
            results = store.search(
                collection_name="test_collection",
                query_vector=qv,
                limit=per,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True
            )
            hits = results

        # 转换结果格式并聚合
        for h in hits:
            # 构建统一的结果格式
            hit_dict = {
                "id": h.id,
                "score": float(h.score),
                "metadata": h.payload,
                "content": h.payload.get("content") or h.payload.get("text")
            }
            mid = hit_dict["id"]

            # 保留最高分的结果
            if mid not in agg or hit_dict["score"] > agg[mid]["score"]:
                agg[mid] = hit_dict

    # 5. 按分数排序并返回top_k
    merged = list(agg.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    print(f"[RAG] 聚合后共{len(merged)}条结果，返回前{top_k}条")
    return merged[:top_k]


# ---------------------- 核心文档处理模块 ----------------------
def _convert_to_markdown(path: str) -> str:
    """多格式文档转Markdown（优先处理PDF）"""
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return ""

    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)

    # 其他格式通用转换
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)

    try:
        result = md_instance.convert(path)
        markdown_text = getattr(result, "text_content", None)
        if isinstance(markdown_text, str) and markdown_text.strip():
            print(f"[RAG] 转换成功: {path} -> {len(markdown_text)} 字符")
            return markdown_text
        return ""
    except Exception as e:
        print(f"[WARNING] 转换失败 {path}: {e}")
        return _fallback_text_reader(path)


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按标题/段落分割文本（优化版：添加进度+性能提升）"""
    lines = text.splitlines()
    total_lines = len(lines)
    print(f"[RAG] 开始分割文本：共{total_lines}行")

    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if len(content) < 10:  # 过滤过短的无效段落
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })

    # 逐行处理并打印进度
    for idx, ln in enumerate(lines):
        # 每处理100行打印一次进度
        if idx % 100 == 0 and idx > 0:
            print(f"[RAG] 文本分割进度：{idx}/{total_lines}行")

        raw = ln.strip()
        if not raw:
            flush_buf(char_pos)
            buf = []
        elif raw.startswith("#"):
            # 处理Markdown标题
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
        else:
            # 普通段落内容
            buf.append(raw)
        char_pos += len(ln) + 1

    flush_buf(char_pos)
    print(f"[RAG] 文本分割完成：共提取{len(paragraphs)}个段落")

    # 兜底：如果没有分割出段落，整段作为一个
    if not paragraphs:
        clean_text = text.strip()
        if clean_text:
            paragraphs = [{"content": clean_text, "heading_path": None, "start": 0, "end": len(clean_text)}]
            print(f"[RAG] 兜底处理：整段文本作为1个段落")

    return paragraphs


def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int = 512, overlap_tokens: int = 50) -> List[Dict]:
    """智能分块（优化版：添加进度+提前计算Token）"""
    total_paragraphs = len(paragraphs)
    print(f"[RAG] 开始智能分块：共{total_paragraphs}个段落，目标块大小{chunk_tokens}Token，重叠{overlap_tokens}Token")

    # 提前计算所有段落的Token数，避免重复计算
    paragraph_tokens = []
    for idx, p in enumerate(paragraphs):
        token_len = _approx_token_len(p["content"]) or 1
        paragraph_tokens.append(token_len)
        if idx % 50 == 0 and idx > 0:
            print(f"[RAG] Token计算进度：{idx}/{total_paragraphs}段落")

    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0

    while i < total_paragraphs:
        p = paragraphs[i]
        p_tokens = paragraph_tokens[i]

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            # 生成分块
            content = "\n\n".join(x["content"] for x in cur)
            chunks.append({
                "content": content,
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "heading_path": next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None),
            })
            # 保留重叠部分
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

        # 打印分块进度
        if len(chunks) % 10 == 0 and len(chunks) > 0:
            print(f"[RAG] 分块进度：已生成{len(chunks)}个块，处理{i}/{total_paragraphs}段落")

    # 处理最后一个分块
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        chunks.append({
            "content": content,
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "heading_path": next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None),
        })

    print(f"[RAG] 分块完成：共生成{len(chunks)}个文本块")
    return chunks


def _approx_token_len(text: str) -> int:
    """近似Token长度（中英文混合）"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


def _is_cjk(ch: str) -> bool:
    """判断CJK字符"""
    code = ord(ch)
    return (
            0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
            0xF900 <= code <= 0xFAFF
    )


def index_chunks(
        store=None,
        chunks: List[Dict] = None,
        cache_db: Optional[str] = None,
        batch_size: int = 32,
        rag_namespace: str = "default",
        collection_name: str = "test_collection"
) -> None:
    """批量向量入库（优化版：添加详细进度+跳过重复索引+超时处理）"""
    if not chunks:
        print("[RAG] 无分块数据可入库")
        return

    # 初始化嵌入模型
    embedder = get_text_embedder()
    dimension = get_dimension(1536)

    # 创建Qdrant客户端
    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] 初始化Qdrant存储，维度：{dimension}")

    # 预处理文本
    processed_texts = []
    total_chunks = len(chunks)
    print(f"[RAG] 开始预处理文本：共{total_chunks}个分块")

    for idx, c in enumerate(chunks):
        processed_content = _preprocess_markdown_for_embedding(c["content"])
        if processed_content and len(processed_content) > 10:  # 过滤过短文本
            processed_texts.append(processed_content)
        else:
            processed_texts.append("无效文本")  # 避免空文本嵌入

        # 打印预处理进度
        if idx % 20 == 0 and idx > 0:
            print(f"[RAG] 文本预处理进度：{idx}/{total_chunks}分块")

    print(f"[RAG] 开始嵌入：共{len(processed_texts)}个分块，批次大小{batch_size}")

    # 批量生成向量
    vecs: List[List[float]] = []
    total_batches = (len(processed_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(processed_texts))
        part = processed_texts[start_idx:end_idx]

        print(f"[RAG] 处理嵌入批次 {batch_idx + 1}/{total_batches}：分块{start_idx + 1}-{end_idx}")

        try:
            # 使用修复后的批量嵌入方法
            part_vecs = embedder._get_text_embeddings(part)
            vecs.extend(part_vecs)
            print(f"[RAG] 批次{batch_idx + 1}嵌入成功：{len(part_vecs)}个向量")
        except Exception as e:
            print(f"[WARNING] 批次{batch_idx + 1}嵌入失败：{e}，为每个文本填充零向量")
            vecs.extend([[0.0] * dimension for _ in part])

    # 批量插入Qdrant
    points = []
    valid_count = 0
    print(f"[RAG] 开始准备Qdrant入库数据：共{len(chunks)}个分块")

    for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
        # 过滤全零向量
        if all(v == 0.0 for v in vec):
            if idx % 50 == 0:
                print(f"[RAG] 过滤进度：{idx}/{len(chunks)}分块（已过滤{valid_count}个有效向量）")
            continue

        # 生成唯一UUID
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{rag_namespace}_{collection_name}_pdf_{idx}"))
        point = PointStruct(
            id=chunk_id,
            vector=vec,
            payload={
                "content": chunk["content"],
                "heading_path": chunk.get("heading_path"),
                "start": chunk["start"],
                "end": chunk["end"],
                "namespace": rag_namespace,
                "rag_namespace": rag_namespace,
                "source": "test.pdf",
                "is_rag_data": True,
                "data_source": "rag_pipeline",
                "memory_type": "rag_chunk",
                "create_time": time.time()
            }
        )
        points.append(point)
        valid_count += 1

        # 打印入库准备进度
        if idx % 50 == 0 and idx > 0:
            print(f"[RAG] 入库数据准备进度：{idx}/{len(chunks)}分块（有效向量{valid_count}个）")

    # 分批次插入
    if points:
        total_insert_batches = (len(points) + 99) // 100
        print(f"[RAG] 开始插入Qdrant：共{len(points)}个向量，分{total_insert_batches}批次")

        for batch_idx in range(total_insert_batches):
            start_idx = batch_idx * 100
            end_idx = min((batch_idx + 1) * 100, len(points))
            batch_points = points[start_idx:end_idx]

            print(f"[RAG] 插入批次 {batch_idx + 1}/{total_insert_batches}：{len(batch_points)}个向量")

            # 设置超时时间，避免卡住
            try:
                store.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True
                )
                print(f"[RAG] 批次{batch_idx + 1}插入成功")
            except Exception as e:
                print(f"[WARNING] 批次{batch_idx + 1}插入失败：{e}")
                continue

        print(f"✅ PDF入库完成：共{valid_count}个有效向量，集合：{collection_name}")
    else:
        print("❌ 无有效向量可入库，请检查DashScope嵌入结果")


# ---------------------- Qdrant ID生成工具 ----------------------
def generate_valid_qdrant_id(document_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, document_id))


# ---------------------- 增强版RAGTool ----------------------
class EnhancedQdrantRAGTool(RAGTool):
    def __init__(self, knowledge_base_path, collection_name, rag_namespace, qdrant_client=None, embedding_model=None):
        super().__init__(
            knowledge_base_path=knowledge_base_path,
            collection_name=collection_name,
            rag_namespace=rag_namespace,
            expandable=False
        )
        self.extra_qdrant_client = qdrant_client
        self.extra_embedding_model = embedding_model

    def execute(self, action, **kwargs):
        """执行工具操作"""
        return self.run({"action": action, **kwargs})

    def get_name(self):
        """获取工具名称"""
        return "rag"

    def get_description(self):
        """获取工具描述"""
        return "增强版RAG工具，支持PDF导入、文本添加和智能检索"

    def get_parameters(self):
        """获取工具参数"""
        return super().get_parameters()


# ---------------------- 初始化服务函数 ----------------------
def init_services():
    """初始化Qdrant和DashScope"""
    # DashScope嵌入模型
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("❌ 请配置DASHSCOPE_API_KEY到.env文件")

    # 使用修复后的安全嵌入模型
    embedding = SafeDashScopeEmbedding(
        model_name="text-embedding-v1",
        api_key=dashscope_api_key
    )
    print("✅ DashScope嵌入模型初始化成功")

    # Qdrant客户端
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("❌ 请配置QDRANT_URL和QDRANT_API_KEY到.env文件")

    print(f"[RAG] 尝试初始化Qdrant客户端: {qdrant_url}")

    # 先测试DNS解析
    try:
        parsed_url = urlparse(qdrant_url)
        hostname = parsed_url.netloc.split(':')[0]
        port = int(parsed_url.netloc.split(':')[1]) if ':' in parsed_url.netloc else 443

        print(f"[RAG] 解析Qdrant地址: {hostname}:{port}")

        # 测试DNS解析
        addrinfo = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        print(f"[RAG] DNS解析成功: {addrinfo[0][4]}")

    except socket.gaierror as e:
        print(f"[ERROR] DNS解析失败: {e}")
        print("[INFO] 请检查网络连接和QDRANT_URL配置")
        print("[INFO] 尝试使用IP地址替代域名")

        # 尝试使用硬编码的IP地址
        print("[INFO] 尝试使用备用IP地址...")
        backup_ip = "34.248.146.137"  # 示例IP，实际需替换为你的解析结果
        backup_url = f"{parsed_url.scheme}://{backup_ip}:{port}{parsed_url.path}"
        print(f"[INFO] 使用备用URL: {backup_url}")
        qdrant_url = backup_url

        # 再次测试连接
        try:
            print("[RAG] 测试备用URL连接...")
            test_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
            test_client.get_collections()
            print("[RAG] 备用URL连接成功！")
        except Exception as backup_e:
            print(f"[ERROR] 备用URL连接失败: {backup_e}")
            raise

    except Exception as e:
        print(f"[ERROR] Qdrant初始化前置检查失败: {e}")
        raise

    # 创建Qdrant客户端
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # 测试连接
        print("[RAG] 测试Qdrant连接...")
        client.get_collections()
        print("[RAG] Qdrant连接成功！")

    except Exception as e:
        print(f"[ERROR] Qdrant客户端初始化失败: {e}")
        raise

    # 创建集合（确保存在）
    collection_name = "test_collection"
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance="Cosine")
            )
            print(f"✅ 创建Qdrant集合：{collection_name}")
        else:
            print(f"✅ Qdrant集合 {collection_name} 已存在")
    except Exception as e:
        print(f"[ERROR] 集合操作失败: {e}")
        raise

    return client, embedding


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    try:
        # 1. 初始化基础服务
        print("[RAG] 初始化核心服务...")
        qdrant_client, embedding_model = init_services()

        # 2. 创建LLM和Agent
        print("[RAG] 创建LLM和Agent...")
        llm_config = {
            "api_key": os.getenv("LLM_API_KEY"),
            "model_id": os.getenv("LLM_MODEL_ID", "qwen-turbo"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "timeout": 60
        }
        llm = HelloAgentsLLM(**{k: v for k, v in llm_config.items() if v})
        agent = SimpleAgent(name="PDF知识助手", llm=llm)

        # 3. 注册增强版RAG工具
        print("[RAG] 注册增强版RAG工具...")
        rag_tool = EnhancedQdrantRAGTool(
            knowledge_base_path="./knowledge_base",
            collection_name="test_collection",
            rag_namespace="test",
            qdrant_client=qdrant_client,
            embedding_model=embedding_model
        )
        tool_registry = ToolRegistry()
        tool_registry.register_tool(rag_tool)
        agent.tool_registry = tool_registry
        print("✅ 增强版RAG工具注册成功")

        # 4. 导入test.pdf（优化版：添加完整进度反馈）
        print("\n===== 导入test.pdf =====")
        if os.path.exists("./test.pdf"):
            start_time = time.time()
            print(f"[RAG] 开始处理PDF：{time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 手动调用解析和入库流程
            pdf_text = _convert_to_markdown("./test.pdf")

            if pdf_text:
                # 文本分割
                paragraphs = _split_paragraphs_with_headings(pdf_text)

                # 智能分块
                chunks = _chunk_paragraphs(paragraphs)
                print(f"[RAG] PDF分割出 {len(chunks)} 个有效分块")

                # 向量入库
                index_chunks(
                    store=qdrant_client,
                    chunks=chunks,
                    rag_namespace="test",
                    collection_name="test_collection"
                )

                # 计算总耗时
                total_time = time.time() - start_time
                print(f"[RAG] PDF处理完成！总耗时：{total_time:.2f}秒")
            else:
                print("❌ PDF解析失败，无有效文本")
        else:
            print("⚠️ test.pdf 文件不存在，跳过导入操作")

        # 5. 基础检索测试
        print("\n===== 基础检索测试 ======")
        search_result = search_vectors_expanded(
            store=qdrant_client,
            query="唇读 深度学习 论文",
            rag_namespace="test",
            top_k=3,
            enable_mqe=False  # 先关闭MQE，简化测试
        )
        if search_result:
            for idx, res in enumerate(search_result, 1):
                print(f"\n📝 检索结果{idx}（相似度：{res['score']:.4f}）:")
                print(f"   内容：{res['content'][:200]}...")
        else:
            print("❌ 未检索到相关内容")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序执行")
    except Exception as e:
        print(f"❌ 程序运行失败：{str(e)[:500]}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🎉 程序执行完成")
