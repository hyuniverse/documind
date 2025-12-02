from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from app.rag_service import answer_question, initialize_RAG_database

class QuestionRequest(BaseModel):
    """
    사용자로부터 질문을 받을 때 사용되는 요청 스키마
    """
    question: str

class DocumentSource(BaseModel):
    """
    RAG 응답에 포함될 문서 출처 정보
    """
    source: str
    content: str

class AnswerResponse(BaseModel):
    """
    질문에 대한 AI 응답 및 근거 문서를 담는 응답 스키마.
    """
    answer: str
    source_documents: List[DocumentSource]

load_dotenv()
app = FastAPI()

# FastAPI 앱 시작 시 RAG 데이터베이스 초기화
@app.on_event("startup")
async def startup_event():
    """
    FastAPI 서버 시작 시 RAG 데이터베이스를 초기화합니다.
    """
    print("[FastAPI] 서버 시작 - RAG 데이터베이스 초기화 중...")
    initialize_RAG_database()
    print("[FastAPI] RAG 데이터베이스 초기화 완료!")

@app.get("/")
def read_root():
    return {"message": "DOCUMIND API is running."}

@app.post("/ask", response_model=AnswerResponse)
def ask_document_qa(request: QuestionRequest):
    """
    RAG 시스템에 질문을 보내고, AI의 답변과 근거 문서를 반환합니다.
    Args:
        request (QuestionRequest): 사용자 질문이 포함된 요청 객체
    Returns:
        AnswerResponse: AI 답변과 출처 문서 리스트가 포함된 응답 객체
    """
    try:
        # 1. RAG 서비스 함수 호출
        answer, source_docs = answer_question(request.question)
    except RuntimeError as e:
        # RAG 데이터베이스가 초기화되지 않은 경우
        if "벡터 스토어가 초기화되지 않았습니다" in str(e):
            print("[FastAPI] RAG 데이터베이스가 초기화되지 않아 재초기화를 시도합니다...")
            initialize_RAG_database()
            answer, source_docs = answer_question(request.question)
        else:
            raise HTTPException(status_code=500, detail=str(e))

    # 2. 응답 스키마에 맞게 데이터 가공
    processed_sources: List[DocumentSource] = []
    for doc in source_docs:
        # 파일 명과 페이지 번호를 결합하여 제공
        source_name = f"{doc.metadata.get('source', 'N/A')}:{doc.metadata.get('page', 'N/A')}"
        processed_sources.append(
            DocumentSource(
                source = source_name,
                content = doc.page_content # 검색된 청크의 내용
            )
        )

    # Pydantic 응답 모델 생성 및 반환
    return AnswerResponse(
        answer=answer,
        source_documents=processed_sources
    ) 