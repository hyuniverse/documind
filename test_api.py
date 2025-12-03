#!/usr/bin/env python3
"""
DocuMind API 사용 예제 스크립트

이 스크립트는 DocuMind API를 사용하여 문서에 질문하는 방법을 보여줍니다.
"""

import requests
import json
import sys

# API 서버 설정
API_BASE_URL = "http://localhost:8000"

def check_server_status():
    """서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            print("✅ 서버 연결 성공!")
            return True
        else:
            print(f"❌ 서버 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행중인지 확인하세요.")
        return False

def ask_question(question):
    """문서에 질문하고 답변 받기"""
    url = f"{API_BASE_URL}/ask"
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}
    
    try:
        print(f"\n🤔 질문: {question}")
        print("⏳ AI가 답변을 생성하는 중...")
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n🤖 답변:\n{result['answer']}")
            print(f"\n📚 참조 문서:")
            
            for i, source in enumerate(result['source_documents'], 1):
                print(f"\n[문서 {i}] {source['source']}")
                print(f"내용: {source['content'][:100]}...")
                
        else:
            print(f"❌ API 오류: {response.status_code}")
            print(f"오류 내용: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 오류: {e}")

def main():
    """메인 함수"""
    print("🚀 DocuMind API 테스트 스크립트")
    print("=" * 50)
    
    # 서버 상태 확인
    if not check_server_status():
        print("\n💡 서버 실행 방법:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # 예제 질문들
    example_questions = [
        "문서의 주요 내용이 무엇인가요?",
        "이 문서에서 다루는 핵심 주제는 무엇입니까?",
        "문서에 언급된 중요한 수치나 데이터가 있나요?",
        "결론 부분에서 제시하는 주요 인사이트는 무엇인가요?"
    ]
    
    print("\n📝 예제 질문을 사용하거나 직접 질문을 입력하세요.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    # 예제 질문 표시
    print("📋 예제 질문들:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    while True:
        print("\n" + "─" * 50)
        user_input = input("\n질문을 입력하세요 (숫자 1-4 또는 직접 입력): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 프로그램을 종료합니다.")
            break
        
        if user_input.isdigit() and 1 <= int(user_input) <= 4:
            question = example_questions[int(user_input) - 1]
        elif user_input:
            question = user_input
        else:
            print("❌ 유효한 질문을 입력해주세요.")
            continue
        
        ask_question(question)

if __name__ == "__main__":
    main()
