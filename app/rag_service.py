import os
from typing import Optional, Tuple, List, Any
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough, Runnable

# 설정 파일 임포트
from .config import (
    PathConfig, 
    TextSplitterConfig, 
    LLMConfig, 
    SearchConfig, 
    LogConfig, 
    PromptTemplates, 
    ErrorMessages, 
    SupportedFormats
)

# Global variables
vectorstore: Optional[VectorStore] = None
qa_chain: Optional[Runnable] = None   

def get_loader() -> BaseLoader:
    """
    문서 파일 확장자에 따라 적절한 Document Loader를 반환합니다.
    
    Returns:
        BaseLoader: PDF 또는 텍스트 파일에 맞는 로더
        
    Raises:
        FileNotFoundError: 문서 파일이 존재하지 않는 경우
        ValueError: 지원하지 않는 파일 형식인 경우
    """
    if not os.path.exists(PathConfig.DOCUMENT_PATH):
        raise FileNotFoundError(ErrorMessages.FILE_NOT_FOUND.format(file_path=PathConfig.DOCUMENT_PATH))
    
    if any(PathConfig.DOCUMENT_PATH.endswith(ext) for ext in SupportedFormats.PDF_EXTENSIONS):
        return PDFPlumberLoader(PathConfig.DOCUMENT_PATH)
    elif any(PathConfig.DOCUMENT_PATH.endswith(ext) for ext in SupportedFormats.TEXT_EXTENSIONS):
        return TextLoader(PathConfig.DOCUMENT_PATH)
    else:
        raise ValueError(ErrorMessages.UNSUPPORTED_FILE_FORMAT)

# ===================================================
# RAG Database Initialization
# ===================================================

def initialize_RAG_database():
    """
    RAG 시스템용 벡터 데이터베이스를 초기화합니다.
    기존 DB가 있으면 로드하고, 없거나 비어있으면 새로 생성합니다.
    """
    global vectorstore

    # API 키 검증
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(ErrorMessages.OPENAI_API_KEY_MISSING)
    
    print(f"[RAG] API 키 확인: {openai_api_key[:LogConfig.API_KEY_DISPLAY_LENGTH]}...")

    db_needs_rebuild = False
    
    # 기존 DB 존재 여부 및 상태 확인
    if os.path.exists(PathConfig.PERSIST_DIRECTORY):
        print("[RAG] 기존 Chroma DB 확인 중...")
        try:
            embeddings = OpenAIEmbeddings()
            temp_vectorstore = Chroma(persist_directory=PathConfig.PERSIST_DIRECTORY, embedding_function=embeddings)
            
            # DB 유효성 검사 (테스트 검색으로 확인)
            test_results = temp_vectorstore.similarity_search(SearchConfig.TEST_QUERY, k=1)
            
            if len(test_results) == 0:
                print("[RAG] 기존 DB가 비어있습니다. 재생성합니다.")
                db_needs_rebuild = True
            else:
                print(f"[RAG] 기존 DB 확인됨. 로드 완료.")
                vectorstore = temp_vectorstore
                return
                
        except Exception as e:
            print(f"[RAG] 기존 DB 로드 실패: {e}")
            db_needs_rebuild = True
    else:
        print("[RAG] Chroma DB가 존재하지 않습니다.")
        db_needs_rebuild = True
    
    # DB 재생성 프로세스
    if db_needs_rebuild:
        # 1. 기존 DB 정리
        if os.path.exists(PathConfig.PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(PathConfig.PERSIST_DIRECTORY)
            print(f"[RAG] 기존 DB 폴더 삭제: {PathConfig.PERSIST_DIRECTORY}")
        
        # 2. 문서 로드
        print(f"[RAG] 문서 로드 시작: {PathConfig.DOCUMENT_PATH}")
        loader = get_loader()
        documents = loader.load()
        print(f"[RAG] {len(documents)}개 문서 로드 완료")
        
        if documents:
            print(f"[RAG] 문서 샘플: {documents[0].page_content[:LogConfig.DOCUMENT_PREVIEW_LENGTH]}...")

        # 3. 문서 분할 (청킹)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TextSplitterConfig.CHUNK_SIZE,
            chunk_overlap=TextSplitterConfig.CHUNK_OVERLAP,
            separators=TextSplitterConfig.SEPARATORS
        )
        texts = text_splitter.split_documents(documents)
        print(f"[RAG] {len(texts)}개 청크 생성 완료")
        
        if texts:
            print(f"[RAG] 청크 샘플: {texts[0].page_content[:LogConfig.CHUNK_PREVIEW_LENGTH]}...")

        # 4. 임베딩 생성 및 벡터 DB 저장
        print("[RAG] 임베딩 생성 중...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PathConfig.PERSIST_DIRECTORY
        )

        print(f"[RAG] 벡터 DB 저장 완료: {PathConfig.PERSIST_DIRECTORY}")
        print(f"[RAG] 저장된 벡터 수: {len(texts)}")


def get_retriever():
    """
    초기화된 벡터 스토어에서 문서 검색기를 반환합니다.
    
    Returns:
        VectorStoreRetriever: 설정된 수만큼 유사 문서를 검색하는 retriever
        
    Raises:
        RuntimeError: 벡터 스토어가 초기화되지 않은 경우
    """
    global vectorstore

    if vectorstore is None:
        raise RuntimeError(ErrorMessages.VECTORSTORE_NOT_INITIALIZED)

    return vectorstore.as_retriever(search_kwargs={"k": SearchConfig.SIMILARITY_SEARCH_K})

def get_qa_chain():
    """
    RAG 기반 질문-응답 체인을 생성하고 반환합니다.
    
    Returns:
        RunnableSequence: LangChain 실행 가능한 QA 체인
        
    Raises:
        RuntimeError: 벡터 스토어가 초기화되지 않은 경우
    """
    global qa_chain, vectorstore

    if qa_chain is not None:
        return qa_chain
    
    if vectorstore is None:
        raise RuntimeError(ErrorMessages.VECTORSTORE_NOT_INITIALIZED)
    
    # LLM 및 Retriever 설정
    llm = ChatOpenAI(temperature=LLMConfig.TEMPERATURE, model=LLMConfig.MODEL)
    retriever = get_retriever()

    def format_prompt(inputs):
        """
        검색된 문서와 질문을 포맷팅하여 LLM용 프롬프트를 생성합니다.
        
        Args:
            inputs: context (검색된 문서들)와 question을 포함한 딕셔너리
            
        Returns:
            str: 포맷팅된 프롬프트 문자열
        """
        question = inputs["question"]
        context_docs = inputs["context"]
        
        # 검색된 문서들을 포맷팅
        formatted_context = PromptTemplates.format_context_documents(context_docs)
        
        # 템플릿을 사용하여 프롬프트 생성
        prompt = PromptTemplates.QA_PROMPT_TEMPLATE.format(
            context=formatted_context,
            question=question
        )
        return prompt

    # RAG 파이프라인: 문서검색 -> 프롬프트 포맷팅 -> LLM 실행
    qa_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | format_prompt
        | llm
    )
    return qa_chain


    

def answer_question(question: str) -> Tuple[str, List[Document]]:
    """
    사용자 질문에 대해 RAG 시스템을 통해 답변을 생성합니다.
    
    Args:
        question: 사용자의 질문 문자열
        
    Returns:
        Tuple[str, List[Document]]: (생성된 답변, 참조된 문서들)
        
    Raises:
        RuntimeError: 벡터 스토어가 초기화되지 않은 경우
        Exception: 문서 검색 또는 QA Chain 실행 중 오류 발생 시
    """
    print(f"[RAG] 질문: {question}")
    
    if vectorstore is None:
        raise RuntimeError(ErrorMessages.VECTORSTORE_NOT_INITIALIZED)
    
    # 1. 관련 문서 검색
    try:
        retriever = get_retriever()
        sources = retriever.invoke(question)
        print(f"[RAG] 검색된 문서: {len(sources)}개")
        
        if sources:
            for i, doc in enumerate(sources):
                print(f"[RAG] 문서 {i+1}: {doc.page_content[:LogConfig.SEARCH_RESULT_PREVIEW_LENGTH]}...")
        else:
            print("[RAG] 경고: 검색된 문서가 없습니다!")
            
    except Exception as e:
        print(f"[RAG] 문서 검색 오류: {e}")
        raise
    
    # 2. QA Chain 실행하여 답변 생성
    try:
        qa = get_qa_chain()
        result = qa.invoke(question)
        
        # AIMessage 객체에서 텍스트 추출
        answer_text = str(result.content) if hasattr(result, 'content') else str(result)
        print(f"[RAG] 답변 완료: {len(answer_text)}자")
        
        return answer_text, sources
        
    except Exception as e:
        print(f"[RAG] QA Chain 오류: {e}")
        raise
