# RAG Service 설정 가이드

## 개요
RAG (Retrieval-Augmented Generation) 시스템의 다양한 설정을 환경 변수 또는 설정 파일을 통해 커스터마이즈할 수 있습니다.

## 설정 파일 구조

### `app/config.py`
모든 설정값들이 클래스별로 정리되어 있습니다:

- `PathConfig`: 파일 경로 설정
- `TextSplitterConfig`: 텍스트 분할 설정  
- `LLMConfig`: 언어 모델 설정
- `SearchConfig`: 검색 관련 설정
- `LogConfig`: 로깅 출력 설정
- `PromptTemplates`: 프롬프트 템플릿
- `ErrorMessages`: 에러 메시지
- `SupportedFormats`: 지원 파일 형식

## 환경 변수 설정

### 필수 설정
```bash
OPENAI_API_KEY=your_api_key_here
```

### 선택 설정
```bash
# 파일 경로
RAG_PERSIST_DIR=./custom_chroma_db
RAG_DOCUMENT_PATH=./documents/my_document.pdf

# 텍스트 처리
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# LLM 설정
RAG_LLM_MODEL=gpt-4
RAG_LLM_TEMPERATURE=0.1

# 검색 설정
RAG_SEARCH_K=3
```

## 설정 예제

### 1. 기본 설정으로 실행
```bash
export OPENAI_API_KEY="your_key"
# 기본값들이 자동으로 사용됩니다
```

### 2. 커스텀 문서로 실행
```bash
export OPENAI_API_KEY="your_key"
export RAG_DOCUMENT_PATH="./my_documents/research_paper.pdf"
export RAG_PERSIST_DIR="./research_db"
```

### 3. 고급 설정으로 실행
```bash
export OPENAI_API_KEY="your_key"
export RAG_LLM_MODEL="gpt-4"
export RAG_CHUNK_SIZE="1200"
export RAG_SEARCH_K="7"
export RAG_LLM_TEMPERATURE="0.2"
```

## 프롬프트 커스터마이징

`config.py`의 `PromptTemplates` 클래스에서 프롬프트를 수정할 수 있습니다:

```python
class PromptTemplates:
    QA_PROMPT_TEMPLATE = """당신은 전문 문서 분석가입니다.
    
다음 문서들을 참조하세요:
{context}

질문: {question}

정확하고 구체적인 답변을 제공하세요."""
```

## 지원 파일 형식

현재 지원되는 파일 형식:
- PDF (`.pdf`)
- 텍스트 (`.txt`)

새로운 형식을 추가하려면 `config.py`의 `SupportedFormats` 클래스를 수정하고, `rag_service.py`의 `get_loader()` 함수에 해당 로더를 추가하세요.

## 트러블슈팅

### 1. 환경 변수가 적용되지 않는 경우
- 터미널을 재시작하거나 `source ~/.bashrc` 실행
- `.env` 파일을 사용하는 경우 python-dotenv 패키지 설치

### 2. 커스텀 문서 경로 사용 시
- 파일 경로가 정확한지 확인
- 파일 권한 확인
- 지원되는 파일 형식인지 확인

### 3. 벡터 DB 관련 이슈
- `RAG_PERSIST_DIR` 디렉토리 권한 확인
- 충분한 디스크 공간 확인
- 기존 DB를 삭제하고 재생성 시도
