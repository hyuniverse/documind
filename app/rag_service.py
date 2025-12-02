import os
from typing import Optional, Tuple, List, Any # 새로 추가된 타입 힌트
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
# 1. 환경 설정 및 전역 변수
# ===================================================

# Chroma DB 파일이 저장될 로컬 디렉토리 경로
# 프로젝트 루트에 저장되도록 하여 git 관리 용이하도록 함
PERSIST_DIRECTORY: str = "./chroma_db"
# RAG 시스템에 사용될 원본 문서 경로
# TODO: 실제 운영 시 동적으로 변경될 수 있도록 개선 필요
DOCUMENT_PATH: str = os.path.join(os.path.dirname(__file__), "sample.pdf")

# 초기화된 벡터 DB 인스턴스
# FastAPI 서비스 전역에서 접근 가능한 캐시 역할을 수행함.
# None으로 초기화하며, initialize_rag_database() 호출 후 Chroma 인스턴스로 업데이트됨.
vectorstore: Optional[VectorStore] = None

qa_chain: Optional[Runnable] = None


# ===================================================
# 2. Utility Functions
# ===================================================   

def get_loader() -> BaseLoader:
    """
    문서 파일 경로(DOCUMENT_PATH)에 따라 적절한 Langchain Loader 인스턴스를 반환합니다.
    Returns:
        BaseLoader: 파일 타입에 맞는 로더(TextLoader or PyPDFLoader) 인스턴스
    Raises:
        ValueError: 지원하지 않는 파일 형식인 경우
        FileNotFoundError: 파일이 존재하지 않는 경우
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
# 3. RAG 데이터베이스 초기화 함수
# ===================================================

def initialize_RAG_database():
    global vectorstore

    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    print(f"[RAG Service] OpenAI API 키 확인: {openai_api_key[:20]}..." if len(openai_api_key) > 20 else openai_api_key)

    # 1. DB가 이미 로컬에 존재할 경우
    #    새로 임베딩을 수행하지 않고 로드
    if os.path.exists(PERSIST_DIRECTORY):
        print("[RAG Service] 기존 Chroma DB 로드 중...")

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("[RAG Service] Chroma DB 로드 완료.")
        return
    
    # 2. DB가 로컬에 존재하지 않을 경우 새로 생성합니다.(Indexing Pipeline 실행)
    print(f"[{DOCUMENT_PATH}] 문서 로드 및 DB 생성 시작...")
    print(f"[RAG Service] 문서 경로: {DOCUMENT_PATH}")
    print(f"[RAG Service] 문서 파일 존재 여부: {os.path.exists(DOCUMENT_PATH)}")

    loader = get_loader()
    documents = loader.load()
    print(f"총 {len(documents)}개의 LangChain Document 객체 로드 완료.")
    
    # 문서 내용 샘플 출력
    if documents:
        print(f"[RAG Service] 첫 번째 문서 내용 미리보기: {documents[0].page_content[:200]}...")
        print(f"[RAG Service] 첫 번째 문서 메타데이터: {documents[0].metadata}")

    # 문서 분할(Chunking)
    # RecursiveCharacterTextSplitter는 다양한 구분자(newline, space, punctuation 등)를 활용하여
    # 텍스트가 의미 있는 단위로 쪼개지도록 시도합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200 # 200토큰 중복 설정으로 청크 간의 맥락 유지
    )

    texts = text_splitter.split_documents(documents)
    print(f"문서 분할 완료. 총 {len(texts)}개의 청크 생성됨.")
    
    # 청크 내용 샘플 출력
    if texts:
        print(f"[RAG Service] 첫 번째 청크 내용: {texts[0].page_content[:200]}...")

    # 임베딩 생성 및 벡터 DB 저장 (실제 RAG 데이터 구조화 단계)
    # OpenAI 임베딩 모델 사용해 각 청크의 벡터 표현 생성(비용 발생 💸)
    print("[RAG Service] 임베딩 생성 중... (OpenAI API 호출)")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"✅ [RAG Service] 벡터 데이터베이스가 [{PERSIST_DIRECTORY}]에 성공적으로 저장되었습니다.")
    
    # 저장된 벡터 수 확인
    try:
        # Chroma의 새로운 API 사용
        collection_count = len(texts)  # 저장된 문서 수와 동일
        print(f"[RAG Service] 저장된 벡터 수: {collection_count}")
    except Exception as e:
        print(f"[RAG Service] 벡터 수 확인 중 오류: {e}")


def get_retriever():
    global vectorstore

    if vectorstore is None:
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다. RAG 데이터베이스를 먼저 초기화하세요.")

    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_qa_chain():
    """
    초기화된 vectorstore를 기반으로 RetrievalQA Chain을 생성하고 반환합니다.

    Returns:
        RetrievalQA: 질문 응답을 처리하는 Langchain 체인 인스턴스
    Raises:
        RuntimeError: 벡터 스토어가 초기화되지 않은 경우
    """
    global qa_chain, vectorstore

    if qa_chain is not None:
        return qa_chain
    
    if vectorstore is None:
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다. RAG 데이터베이스를 먼저 초기화하세요.")
    
    # 1. LLM 정의 - OpenAI의 GPT-3.5 Turbo 모델 사용
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # 2. Retriever 정의 - DB에서 검색기(Retriever) 생성
    # search_kwargs={"k": 3} 은 질문에 대해 가장 유사한 문서 3개를 검색하라는 의미입니다.
    retriever = get_retriever()

    qa_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough() # query는 그대로 LLM에 전달
        })
        | (lambda x: f"질문: {x['question']}\n문서: {x['context']}\n답변:")  # 프롬프트 포맷팅
        | llm
    )
    return qa_chain


    

def answer_question(question: str) -> Tuple[str, List[Document]]:
    """
    주어진 질문에 대해 RAG 시스템을 통해 답변을 생성합니다.

    Args:
        question (str): 사용자로부터 받은 질문 문자열

    Returns:
        Tuple[str, List[Document]]: 생성된 답변 문자열과 근거 문서 리스트
    """
    print(f"[RAG Service] 질문 받음: {question}")
    
    # 벡터스토어 상태 확인
    if vectorstore is None:
        print("[RAG Service] 오류: vectorstore가 None입니다.")
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")
    
    # 먼저 retriever로 문서 검색 테스트
    try:
        retriever = get_retriever()
        sources = retriever.invoke(question)
        print(f"[RAG Service] 검색된 문서 수: {len(sources)}")
        
        if sources:
            for i, doc in enumerate(sources):
                print(f"[RAG Service] 문서 {i+1}: {doc.page_content[:100]}...")
        else:
            print("[RAG Service] 경고: 검색된 문서가 없습니다!")
            
    except Exception as e:
        print(f"[RAG Service] 문서 검색 중 오류: {e}")
        raise
    
    # QA Chain 실행
    try:
        qa = get_qa_chain()
        print("[RAG Service] QA Chain 실행 중...")
        result = qa.invoke(question)
        print(f"[RAG Service] LLM 응답 타입: {type(result)}")
        
        # AIMessage의 content를 문자열로 안전하게 변환
        answer_text = str(result.content) if hasattr(result, 'content') else str(result)
        print(f"[RAG Service] 최종 답변: {answer_text[:200]}...")
        
        return answer_text, sources
        
    except Exception as e:
        print(f"[RAG Service] QA Chain 실행 중 오류: {e}")
        raise
