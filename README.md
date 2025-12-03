# 📄 DocuMind

**DocuMind**는 RAG(Retrieval-Augmented Generation) 기술을 활용한 지능형 문서 질의응답 시스템입니다. PDF 문서를 업로드하고 문서 내용에 대해 자연어로 질문하면, AI가 문서 내용을 기반으로 정확한 답변을 제공합니다.

## ✨ 주요 기능

-  **스마트 문서 검색**: 벡터 데이터베이스를 활용한 의미 기반 문서 검색
-  **AI 기반 답변**: OpenAI GPT 모델을 활용한 정확한 답변 생성
-  **다양한 파일 형식 지원**: PDF, TXT 파일 처리
-  **REST API**: FastAPI 기반의 고성능 웹 API
-  **유연한 설정**: 환경변수를 통한 세밀한 커스터마이징
-  **문서 출처 제공**: 답변과 함께 근거가 된 문서 부분 표시

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   사용자 질문      │───▶│   FastAPI 서버   │───▶│   RAG Service   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  OpenAI GPT     │◀───│  Vector Store   │
                       │  (답변 생성)      │    │  (ChromaDB)     │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/your-username/documind.git
cd documind

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는 Windows의 경우: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에 OpenAI API 키 설정
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 문서 준비

분석하고 싶은 PDF 문서를 `app/sample.pdf`에 배치하거나, 환경변수로 경로를 지정하세요:

```bash
export RAG_DOCUMENT_PATH="./documents/my_document.pdf"
```

### 4. 서버 실행

```bash
# 개발 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 시작되면 `http://localhost:8000`에서 API에 접근할 수 있습니다.

## 📖 API 사용법

### 기본 상태 확인
```bash
curl http://localhost:8000/
```

### 문서 질의응답
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "문서의 주요 내용이 무엇인가요?"}'
```

### API 응답 예시
```json
{
  "answer": "문서의 주요 내용은 머신러닝 모델의 성능 평가 방법에 관한 것입니다...",
  "source_documents": [
    {
      "source": "sample.pdf:1",
      "content": "머신러닝 모델의 성능을 평가하는 방법에는 여러 가지가 있습니다..."
    }
  ]
}
```

## ⚙️ 상세 설정

### 환경 변수 옵션

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `OPENAI_API_KEY` | - | OpenAI API 키 (필수) |
| `RAG_PERSIST_DIR` | `./chroma_db` | 벡터 DB 저장 경로 |
| `RAG_DOCUMENT_PATH` | `./app/sample.pdf` | 분석할 문서 경로 |
| `RAG_CHUNK_SIZE` | `800` | 텍스트 청크 크기 |
| `RAG_CHUNK_OVERLAP` | `100` | 청크 간 중복 크기 |
| `RAG_LLM_MODEL` | `gpt-3.5-turbo` | 사용할 OpenAI 모델 |
| `RAG_LLM_TEMPERATURE` | `0` | 모델 창의성 설정 |
| `RAG_SEARCH_K` | `5` | 검색할 문서 수 |

### 고급 설정 예시

```bash
# 고성능 설정
export RAG_LLM_MODEL="gpt-4"
export RAG_CHUNK_SIZE="1200"
export RAG_SEARCH_K="7"

# 빠른 응답 설정
export RAG_LLM_MODEL="gpt-3.5-turbo"
export RAG_CHUNK_SIZE="600"
export RAG_SEARCH_K="3"
```

## 🗂️ 프로젝트 구조

```
documind/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 애플리케이션
│   ├── rag_service.py       # RAG 핵심 로직
│   ├── config.py            # 설정 관리
│   └── sample.pdf           # 샘플 문서
├── chroma_db/               # 벡터 데이터베이스 (자동 생성)
├── requirements.txt         # Python 의존성
├── .env.example            # 환경변수 예시
├── CONFIG_GUIDE.md         # 상세 설정 가이드
└── README.md               # 이 파일
```

## 🔧 개발 가이드

### 새로운 파일 형식 추가

1. `config.py`의 `SupportedFormats` 클래스에 확장자 추가:
```python
class SupportedFormats:
    PDF_EXTENSIONS = [".pdf"]
    TEXT_EXTENSIONS = [".txt"]
    DOCX_EXTENSIONS = [".docx"]  # 새로운 형식
```

2. `rag_service.py`의 `get_loader()` 함수에 로더 추가:
```python
elif any(PathConfig.DOCUMENT_PATH.endswith(ext) for ext in SupportedFormats.DOCX_EXTENSIONS):
    return DocxLoader(PathConfig.DOCUMENT_PATH)
```

### 프롬프트 커스터마이징

`config.py`의 `PromptTemplates` 클래스에서 프롬프트 수정:

```python
class PromptTemplates:
    QA_PROMPT_TEMPLATE = """당신은 전문 문서 분석가입니다.
    
다음 문서를 참조하세요:
{context}

질문: {question}

전문가답게 답변해주세요:"""
```

## 📊 성능 최적화

### 텍스트 청킹 최적화
- **짧은 청크 (400-600)**: 정확한 검색, 느린 처리
- **긴 청크 (1000-1500)**: 빠른 처리, 넓은 컨텍스트

### 모델 선택
- **gpt-3.5-turbo**: 빠른 응답, 비용 효율적
- **gpt-4**: 높은 품질, 복잡한 추론

### 벡터 검색 튜닝
- **적은 문서 (k=3-5)**: 빠른 응답
- **많은 문서 (k=7-10)**: 포괄적 답변

## 🛠️ 트러블슈팅

### 자주 발생하는 문제

**1. OpenAI API 키 오류**
```bash
# 환경변수 확인
echo $OPENAI_API_KEY
# 또는 .env 파일 확인
```

**2. 문서 로드 실패**
```bash
# 파일 경로와 권한 확인
ls -la app/sample.pdf
# 지원 형식 확인 (PDF, TXT만 지원)
```

**3. 벡터 DB 오류**
```bash
# DB 디렉토리 삭제 후 재생성
rm -rf chroma_db/
# 서버 재시작
```

**4. 메모리 부족**
- 청크 크기를 줄이세요: `RAG_CHUNK_SIZE=600`
- 검색 문서 수를 줄이세요: `RAG_SEARCH_K=3`

---

**DocuMind**로 문서 속 지식을 쉽고 빠르게 찾아보세요!
