import os
from typing import List

# ===================================================
# 파일 경로 설정
# ===================================================
class PathConfig:
    # 기본값들
    DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
    DEFAULT_DOCUMENT_FILENAME = "sample.pdf"
    
    # 환경 변수 또는 기본값 사용
    PERSIST_DIRECTORY = os.getenv("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIRECTORY)
    
    # 문서 파일 경로 (환경변수로 override 가능)
    DOCUMENT_PATH = os.getenv(
        "RAG_DOCUMENT_PATH", 
        os.path.join(os.path.dirname(__file__), DEFAULT_DOCUMENT_FILENAME)
    )

# ===================================================
# 텍스트 처리 설정
# ===================================================
class TextSplitterConfig:
    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]

# ===================================================
# LLM 설정
# ===================================================
class LLMConfig:
    MODEL = os.getenv("RAG_LLM_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0"))

# ===================================================
# 검색 설정
# ===================================================
class SearchConfig:
    # 검색할 유사 문서 수
    SIMILARITY_SEARCH_K = int(os.getenv("RAG_SEARCH_K", "5"))
    
    # DB 유효성 검사용 테스트 쿼리
    TEST_QUERY = "test"

# ===================================================
# 로그 출력 설정
# ===================================================
class LogConfig:
    # API 키 표시 길이
    API_KEY_DISPLAY_LENGTH = 20
    
    # 문서/청크 미리보기 길이
    DOCUMENT_PREVIEW_LENGTH = 100
    CHUNK_PREVIEW_LENGTH = 100
    SEARCH_RESULT_PREVIEW_LENGTH = 50

# ===================================================
# 프롬프트 템플릿
# ===================================================
class PromptTemplates:
    # RAG 질문-응답 프롬프트 템플릿
    QA_PROMPT_TEMPLATE = """다음은 관련 문서들입니다:

{context}

질문: {question}

위 문서들의 내용을 바탕으로 질문에 대해 구체적이고 정확한 답변을 제공해주세요. 문서에 명시된 내용만을 기반으로 답변하고, 추측이나 일반적인 지식은 사용하지 마세요.

답변:"""

    @staticmethod
    def format_context_documents(context_docs) -> str:
        """검색된 문서들을 컨텍스트 형태로 포맷팅"""
        formatted_context = ""
        for i, doc in enumerate(context_docs, 1):
            formatted_context += f"[문서 {i}]\n{doc.page_content}\n\n"
        return formatted_context

# ===================================================
# 에러 메시지
# ===================================================
class ErrorMessages:
    OPENAI_API_KEY_MISSING = "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
    FILE_NOT_FOUND = "문서 파일을 찾을 수 없습니다: {file_path}"
    UNSUPPORTED_FILE_FORMAT = "지원하지 않는 파일 형식입니다."
    VECTORSTORE_NOT_INITIALIZED = "벡터 스토어가 초기화되지 않았습니다."

# ===================================================
# 지원 파일 형식
# ===================================================
class SupportedFormats:
    PDF_EXTENSIONS = [".pdf"]
    TEXT_EXTENSIONS = [".txt"]
    
    @classmethod
    def get_all_extensions(cls) -> List[str]:
        return cls.PDF_EXTENSIONS + cls.TEXT_EXTENSIONS
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in cls.get_all_extensions())
