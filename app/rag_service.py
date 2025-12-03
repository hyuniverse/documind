import os
from typing import Optional, Tuple, List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough, Runnable

# ===================================================
# Configuration
# ===================================================
PERSIST_DIRECTORY: str = "./chroma_db"
DOCUMENT_PATH: str = os.path.join(os.path.dirname(__file__), "sample.pdf")

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
    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {DOCUMENT_PATH}")
    
    if DOCUMENT_PATH.endswith(".pdf"):
        return PyPDFLoader(DOCUMENT_PATH)
    elif DOCUMENT_PATH.endswith(".txt"):
        return TextLoader(DOCUMENT_PATH)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

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
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    print(f"[RAG] API 키 확인: {openai_api_key[:20]}...")

    db_needs_rebuild = False
    
    # 기존 DB 존재 여부 및 상태 확인
    if os.path.exists(PERSIST_DIRECTORY):
        print("[RAG] 기존 Chroma DB 확인 중...")
        try:
            embeddings = OpenAIEmbeddings()
            temp_vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            
            # DB 유효성 검사 (테스트 검색으로 확인)
            test_results = temp_vectorstore.similarity_search("test", k=1)
            
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
        if os.path.exists(PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(PERSIST_DIRECTORY)
            print(f"[RAG] 기존 DB 폴더 삭제: {PERSIST_DIRECTORY}")
        
        # 2. 문서 로드
        print(f"[RAG] 문서 로드 시작: {DOCUMENT_PATH}")
        loader = get_loader()
        documents = loader.load()
        print(f"[RAG] {len(documents)}개 문서 로드 완료")
        
        if documents:
            print(f"[RAG] 문서 샘플: {documents[0].page_content[:100]}...")

        # 3. 문서 분할 (청킹)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,           # 청크 크기 (토큰 수)
            chunk_overlap=100,        # 청크 간 중복 토큰 수
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # 분할 우선순위
        )
        texts = text_splitter.split_documents(documents)
        print(f"[RAG] {len(texts)}개 청크 생성 완료")
        
        if texts:
            print(f"[RAG] 청크 샘플: {texts[0].page_content[:100]}...")

        # 4. 임베딩 생성 및 벡터 DB 저장
        print("[RAG] 임베딩 생성 중...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

        print(f"[RAG] 벡터 DB 저장 완료: {PERSIST_DIRECTORY}")
        print(f"[RAG] 저장된 벡터 수: {len(texts)}")


def get_retriever():
    """
    초기화된 벡터 스토어에서 문서 검색기를 반환합니다.
    
    Returns:
        VectorStoreRetriever: 상위 5개 유사 문서를 검색하는 retriever
        
    Raises:
        RuntimeError: 벡터 스토어가 초기화되지 않은 경우
    """
    global vectorstore

    if vectorstore is None:
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")

    return vectorstore.as_retriever(search_kwargs={"k": 5})

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
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
    
    # LLM 및 Retriever 설정
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
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
        
        # 검색된 문서들을 번호와 함께 포맷팅
        formatted_context = ""
        for i, doc in enumerate(context_docs, 1):
            formatted_context += f"[문서 {i}]\n{doc.page_content}\n\n"
        
        prompt = f"""다음은 관련 문서들입니다:

{formatted_context}

질문: {question}

위 문서들의 내용을 바탕으로 질문에 대해 구체적이고 정확한 답변을 제공해주세요. 문서에 명시된 내용만을 기반으로 답변하고, 추측이나 일반적인 지식은 사용하지 마세요.

답변:"""
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
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
    
    # 1. 관련 문서 검색
    try:
        retriever = get_retriever()
        sources = retriever.invoke(question)
        print(f"[RAG] 검색된 문서: {len(sources)}개")
        
        if sources:
            for i, doc in enumerate(sources):
                print(f"[RAG] 문서 {i+1}: {doc.page_content[:50]}...")
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
