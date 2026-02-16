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
from typing import List, Dict, Any
from urllib.parse import urlparse

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


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """执行 RAG 检索"""
    try:
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': '查询词不能为空'}), 400

        # 执行检索
        result = search_vectors_expanded(
            query=query,
            top_k=5,
            rag_namespace="test",
            enable_mqe=False,
            enable_hyde=False
        )

        # 格式化结果
        formatted_results = []
        for item in result:
            formatted_results.append({
                'id': item.get('memory_id', item.get('id', '')),
                'score': round(float(item.get('score', 0.0)), 4),
                'content': item.get('content', ''),
                'metadata': item.get('metadata', {})
            })

        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'total': len(formatted_results)
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
