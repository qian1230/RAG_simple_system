#!/usr/bin/env python3
"""
极简 RAG 可视化 web 应用
- 基于 Flask
- 集成 RAG 检索功能
- 集成 check_qdrant.py 可视化功能
- 支持 PDF 上传和检索
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse, quote

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, url_for
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入 RAG 相关模块
from check_qdrant import search_and_show_details as check_qdrant_details
from check_qdrant import init_services as init_check_services

# 导入 8_RAG.py 中的相关功能
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("eight_rag", os.path.join(os.path.dirname(__file__), "..", "8_RAG.py"))
    eight_rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eight_rag)
    search_vectors_expanded = eight_rag.search_vectors_expanded
    _convert_to_markdown = eight_rag._convert_to_markdown
    _split_paragraphs_with_headings = eight_rag._split_paragraphs_with_headings
    _chunk_paragraphs = eight_rag._chunk_paragraphs
    index_chunks = eight_rag.index_chunks
    SafeDashScopeEmbedding = eight_rag.SafeDashScopeEmbedding
except Exception as e:
    print(f"导入 8_RAG.py 失败: {e}")
    raise

# 全局 LLM 实例
_global_llm = None
_global_react_agent = None

# 先定义 ToolExecutor 类，以便 react_agent.py 可以使用
class ToolExecutor:
    def __init__(self):
        self.tools = {}
    
    def registerTool(self, name, description, func):
        self.tools[name] = {'description': description, 'func': func}
    
    def getAvailableTools(self):
        tools_desc = []
        for name, info in self.tools.items():
            tools_desc.append(f"{name}: {info['description']}")
        return "\n".join(tools_desc)
    
    def getTool(self, name):
        if name in self.tools:
            return self.tools[name]['func']
        return None

# 添加当前目录到系统路径，以便 react_agent.py 可以导入当前模块
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入 ReAct 相关模块
try:
    import importlib.util
    
    # 先导入 llm_client.py
    llm_spec = importlib.util.spec_from_file_location("llm_client", os.path.join(os.path.dirname(__file__), "..", "react", "llm_client.py"))
    llm_client = importlib.util.module_from_spec(llm_spec)
    llm_spec.loader.exec_module(llm_client)
    
    # 在导入 react_agent.py 之前，将 ToolExecutor 添加到系统模块中
    import sys
    sys.modules['tools'] = type('tools', (), {'ToolExecutor': ToolExecutor})()
    
    # 导入 react_agent.py
    react_spec = importlib.util.spec_from_file_location("react_agent", os.path.join(os.path.dirname(__file__), "..", "react", "react_agent.py"))
    react_agent = importlib.util.module_from_spec(react_spec)
    react_spec.loader.exec_module(react_agent)
    
    ReActAgent = react_agent.ReActAgent
    HelloAgentsLLM = llm_client.HelloAgentsLLM
    
    # 确保 ToolExecutor 可用
    if 'ToolExecutor' not in dir():
        ToolExecutor = react_agent.ToolExecutor
except Exception as e:
    print(f"导入 ReAct 模块失败: {e}")
    # 即使导入失败，也要确保 ToolExecutor 可用
    if 'ToolExecutor' not in dir():
        pass  # ToolExecutor 已经在上面定义了
    raise

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局服务实例
_embedding = None
_qdrant_client = None


def get_services():
    """获取嵌入模型和 Qdrant 客户端"""
    global _embedding, _qdrant_client
    if _embedding is None or _qdrant_client is None:
        from check_qdrant import init_services
        _embedding, _qdrant_client = init_services()
    return _embedding, _qdrant_client


def get_llm():
    """获取 LLM 实例"""
    global _global_llm
    if _global_llm is None:
        # 从环境变量获取 LLM 配置
        llm_config = {
            "model": os.getenv("LLM_MODEL_ID"),
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "timeout": int(os.getenv("LLM_TIMEOUT", "60"))
        }
        # 过滤掉 None 值
        filtered_config = {k: v for k, v in llm_config.items() if v}
        _global_llm = HelloAgentsLLM(**filtered_config)
    return _global_llm


def get_react_agent():
    """获取 ReAct 智能体实例"""
    global _global_react_agent
    if _global_react_agent is None:
        # 初始化 LLM
        llm = get_llm()
        
        # 初始化工具执行器
        tool_exec = ToolExecutor()
        
        # 注册私有库检索工具
        def rag_search(query):
            """在私有知识库中检索"""
            try:
                print(f"🔍 开始检索：{query}")
                
                # 第一次搜索：禁用 MQE（因为 HelloAgentsLLM 没有 complete 方法），启用 HYDE
                result = search_vectors_expanded(
                    query=query,
                    top_k=10,
                    rag_namespace="test",
                    enable_mqe=False,
                    enable_hyde=True
                )
                
                print(f"✅ 检索完成，返回 {len(result)} 条结果")
                
                # 格式化结果
                formatted_results = []
                retrieved_content = ""
                
                for item in result:
                    content = item.get('content', '')
                    score = round(float(item.get('score', 0.0)), 4)
                    print(f"📊 结果：相似度={score}")
                    
                    formatted_results.append({
                        'id': item.get('memory_id', item.get('id', '')),
                        'score': score,
                        'content': content,
                        'metadata': item.get('metadata', {})
                    })
                    # 合并检索到的内容，降低相似度阈值
                    if score >= 0.3:
                        retrieved_content += f"相似度: {score}\n内容: {content}\n\n"
                        print(f"✅ 找到高相似度结果，添加到检索内容")
                
                print(f"📝 检索内容长度：{len(retrieved_content)}")
                
                if not retrieved_content:
                    # 第二次搜索：禁用 MQE 和 HYDE，使用更广泛的参数
                    print("🔄 第一次搜索未找到内容，尝试第二次搜索...")
                    result = search_vectors_expanded(
                        query=query,
                        top_k=15,
                        rag_namespace="test",
                        enable_mqe=False,
                        enable_hyde=False
                    )
                    
                    print(f"✅ 第二次检索完成，返回 {len(result)} 条结果")
                    
                    for item in result:
                        content = item.get('content', '')
                        score = round(float(item.get('score', 0.0)), 4)
                        print(f"📊 结果：相似度={score}")
                        
                        if score >= 0.2:
                            retrieved_content += f"相似度: {score}\n内容: {content}\n\n"
                            print(f"✅ 找到高相似度结果，添加到检索内容")
                
                print(f"📝 检索内容长度：{len(retrieved_content)}")
                
                if not retrieved_content:
                    # 第三次搜索：尝试使用更简单的查询
                    print("🔄 第二次搜索未找到内容，尝试第三次搜索...")
                    simple_query = query.split()[0]  # 只使用第一个词
                    print(f"🔍 使用简化查询：{simple_query}")
                    
                    result = search_vectors_expanded(
                        query=simple_query,
                        top_k=15,
                        rag_namespace="test",
                        enable_mqe=False,
                        enable_hyde=False
                    )
                    
                    print(f"✅ 第三次检索完成，返回 {len(result)} 条结果")
                    
                    for item in result:
                        content = item.get('content', '')
                        score = round(float(item.get('score', 0.0)), 4)
                        print(f"📊 结果：相似度={score}")
                        
                        if score >= 0.1:
                            retrieved_content += f"相似度: {score}\n内容: {content}\n\n"
                            print(f"✅ 找到高相似度结果，添加到检索内容")
                
                print(f"📝 最终检索内容长度：{len(retrieved_content)}")
                
                if not retrieved_content:
                    print("❌ 未找到相关内容")
                    return "私有知识库中未找到相关内容"
                
                print("✅ 检索成功，返回检索结果")
                return f"私有知识库检索结果：\n{retrieved_content}"
            except Exception as e:
                print(f"❌ 检索失败：{str(e)}")
                return f"检索失败：{str(e)}"
        
        # 注册 Google 搜索工具（使用 SerpAPI）
        def baidu_search(query):
            """使用 SerpAPI 调用 Google 搜索获取网络信息"""
            try:
                # 导入 SerpAPI
                import serpapi
                import json
                
                # 配置搜索参数
                params = {
                    "engine": "google",  # 搜索引擎
                    "q": query,  # 搜索关键词
                    "api_key": os.getenv('SERPAPI_KEY', 'ee491ea64d8b5f24d20a8254cca74a84b0c014953cfac4ea9a9089c372f44b09'),  # API Key
                    "hl": "zh-CN",  # 中文语言
                    "gl": "cn",  # 中国地区
                    "num": 3  # 返回结果数量
                }
                
                # 调用 SerpAPI 搜索
                print("🚀 正在调用 SerpAPI 搜索...")
                results = serpapi.search(params)
                
                # 将结果转换为字典
                results_dict = results.as_dict()
                
                # 提取有机搜索结果
                if "organic_results" in results_dict and len(results_dict["organic_results"]) > 0:
                    organic_results = results_dict["organic_results"]
                    results_text = []
                    
                    for idx, result in enumerate(organic_results, 1):
                        if idx > 3:  # 只返回前 3 个结果
                            break
                        
                        title = result.get('title', '无')
                        link = result.get('link', '无')
                        snippet = result.get('snippet', '无')
                        
                        results_text.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}\n")
                    
                    return f"Google 搜索结果：\n{''.join(results_text)}"
                else:
                    return f"Google 搜索结果：\n未找到相关内容。"
            except ImportError:
                return f"搜索失败：请安装 serpapi 库（pip install serpapi）"
            except Exception as e:
                return f"搜索失败：{str(e)}"
        
        # 注册工具
        tool_exec.registerTool(
            name="RAGSearch",
            description="私有知识库检索：用于在上传的 PDF 文档中搜索相关内容，输入为搜索关键词。对于涉及上传文档内容的问题，应优先使用此工具。特别是关于唇读、深度学习、论文等相关问题，应首先使用此工具进行检索。",
            func=rag_search
        )
        
        tool_exec.registerTool(
            name="BaiduSearch",
            description="Google 搜索：用于获取网络上的公开信息，输入为搜索关键词。当私有知识库中没有相关信息时，可使用此工具获取网络信息。",
            func=baidu_search
        )
        
        # 初始化 ReAct 智能体
        _global_react_agent = ReActAgent(
            llm_client=llm,
            tool_executor=tool_exec,
            max_steps=5
        )
    return _global_react_agent


def generate_rag_answer(query, retrieved_content):
    """使用 RAG 生成回答"""
    llm = get_llm()
    
    # 构建提示词
    prompt = [
        {
            "role": "system",
            "content": "你是一个能根据论文内容回答问题的智能助手。请严格根据提供的论文内容回答问题，不要添加任何外部信息。如果提供的内容不足以回答问题，请明确说明。"
        },
        {
            "role": "user",
            "content": f"论文内容：\n{retrieved_content}\n\n问题：{query}"
        }
    ]
    
    # 调用 LLM
    try:
        response = llm.think(prompt)
        return response
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return f"生成回答时出错: {str(e)}"


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """执行 RAG 检索并使用 ReAct 智能体生成回答"""
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': '查询词不能为空'}), 400

        # 获取 ReAct 智能体
        react_agent = get_react_agent()
        
        # 执行 ReAct 智能体
        answer = react_agent.run(query)
        
        # 执行传统检索获取详细结果（无论 ReAct 智能体是否返回答案）
        result = search_vectors_expanded(
            query=query,
            top_k=5,
            rag_namespace="test",
            enable_mqe=False,
            enable_hyde=False
        )

        # 格式化结果
        formatted_results = []
        retrieved_content = ""
        
        for item in result:
            content = item.get('content', '')
            formatted_results.append({
                'id': item.get('memory_id', item.get('id', '')),
                'score': round(float(item.get('score', 0.0)), 4),
                'content': content,
                'metadata': item.get('metadata', {})
            })
            # 合并检索到的内容，只保留相似度高的结果
            if item.get('score', 0.0) >= 0.5:
                retrieved_content += f"{content}\n\n"

        # 如果 ReAct 智能体未返回答案，使用传统 RAG
        if not answer:
            # 如果没有检索到内容，生成提示信息
            if not retrieved_content:
                retrieved_content = "未检索到相关内容"
                answer = "抱歉，我没有找到与您的问题相关的论文内容。"
            else:
                # 使用 LLM 生成回答
                answer = generate_rag_answer(query, retrieved_content)

        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'total': len(formatted_results),
            'retrieved_content': retrieved_content,
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/check_qdrant')
def check_qdrant():
    """检查 Qdrant 状态和数据"""
    try:
        # 导入 check_qdrant.py 的功能
        from check_qdrant import search_and_show_details
        
        # 执行检查并捕获输出
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            search_and_show_details()
        output = f.getvalue()
        
        # 解析输出并返回
        return jsonify({
            'success': True,
            'output': output
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload():
    """上传 PDF 文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '请选择文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '请选择文件'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': '只支持 PDF 文件'}), 400

        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 处理文件并入库
        pdf_text = _convert_to_markdown(file_path)
        if not pdf_text:
            return jsonify({
                'success': False,
                'error': 'PDF 解析失败，无有效文本'
            }), 400

        # 文本分割
        paragraphs = _split_paragraphs_with_headings(pdf_text)
        # 智能分块
        chunks = _chunk_paragraphs(paragraphs)
        # 向量入库
        index_chunks(
            chunks=chunks,
            rag_namespace="test",
            collection_name="test_collection"
        )

        return jsonify({
            'success': True,
            'filename': file.filename,
            'chunks': len(chunks)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/collections')
def get_collections():
    """获取 Qdrant 集合列表"""
    try:
        _, client = get_services()
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return jsonify({
            'success': True,
            'collections': collection_names
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search_web', methods=['POST'])
def search_web():
    """执行联网搜索"""
    try:
        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({'error': '查询词不能为空'}), 400

        # 执行百度搜索
        def baidu_search(query):
            """使用百度搜索获取网络信息"""
            try:
                # 使用百度搜索 API
                url = f"https://www.baidu.com/s?wd={quote(query)}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.encoding = 'utf-8'
                
                # 简单解析搜索结果
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 获取搜索结果
                results = []
                for i, result in enumerate(soup.select('.result')):
                    if i >= 5:  # 只返回前 5 个结果
                        break
                    
                    title = result.select_one('h3').text.strip() if result.select_one('h3') else ''
                    summary = result.select_one('.c-abstract').text.strip() if result.select_one('.c-abstract') else ''
                    link = result.select_one('a')['href'] if result.select_one('a') else ''
                    
                    if title:
                        results.append({
                            'title': title,
                            'summary': summary,
                            'link': link
                        })
                
                if not results:
                    return []
                
                return results
            except Exception as e:
                print(f"搜索失败：{str(e)}")
                return []

        # 执行搜索
        results = baidu_search(query)

        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
