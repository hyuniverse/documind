# DocuMind 개발용 Makefile

.PHONY: help install dev test clean setup api-test

# 기본 Python 설정
PYTHON := python3
PIP := pip
VENV := venv
PORT := 8000

# 도움말 표시
help:
	@echo "DocuMind 개발 명령어:"
	@echo ""
	@echo "  setup      - 초기 환경 설정 (가상환경 + 패키지 설치)"
	@echo "  install    - 의존성 패키지 설치"
	@echo "  dev        - 개발 서버 실행"
	@echo "  test       - API 테스트 실행"
	@echo "  api-test   - 대화형 API 테스트"
	@echo "  clean      - 생성된 파일들 정리"
	@echo "  requirements - requirements.txt 업데이트"
	@echo ""

# 초기 환경 설정
setup:
	@echo "🚀 DocuMind 초기 환경 설정 중..."
	$(PYTHON) -m venv $(VENV)
	@echo "📦 가상환경이 생성되었습니다."
	@echo ""
	@echo "다음 명령어로 가상환경을 활성화하세요:"
	@echo "  source $(VENV)/bin/activate  # Linux/macOS"
	@echo "  $(VENV)\\Scripts\\activate     # Windows"
	@echo ""
	@echo "그 다음 'make install'을 실행하세요."

# 의존성 설치
install:
	@echo "📦 의존성 패키지 설치 중..."
	$(PIP) install -r requirements.txt
	@echo "✅ 패키지 설치 완료!"

# 개발 서버 실행
dev:
	@echo "🚀 개발 서버 시작 (포트: $(PORT))..."
	@echo "💡 Swagger UI: http://localhost:$(PORT)/docs"
	@echo "💡 API 문서: http://localhost:$(PORT)/redoc"
	@echo ""
	uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT)

# API 테스트 실행
api-test:
	@echo "🧪 대화형 API 테스트 실행..."
	$(PYTHON) test_api.py

# 빠른 API 상태 확인
test:
	@echo "⚡ 서버 상태 확인..."
	@curl -s http://localhost:$(PORT)/ | grep -q "DOCUMIND API is running" && echo "✅ 서버 정상 동작" || echo "❌ 서버 연결 실패"

# 정리 작업
clean:
	@echo "🧹 생성된 파일들 정리 중..."
	rm -rf __pycache__/
	rm -rf app/__pycache__/
	rm -rf chroma_db/
	rm -rf .pytest_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ 정리 완료!"

# requirements.txt 업데이트
requirements:
	@echo "📋 requirements.txt 업데이트 중..."
	$(PIP) freeze > requirements.txt
	@echo "✅ requirements.txt 업데이트 완료!"

# .env 파일 생성
env:
	@if [ ! -f .env ]; then \
		echo "📝 .env 파일 생성 중..."; \
		cp .env.example .env; \
		echo "✅ .env 파일이 생성되었습니다. OpenAI API 키를 설정하세요."; \
	else \
		echo "⚠️  .env 파일이 이미 존재합니다."; \
	fi

# 전체 설정 (최초 1회 실행)
init: setup env
	@echo ""
	@echo "🎉 초기 설정이 완료되었습니다!"
	@echo ""
	@echo "다음 단계:"
	@echo "1. 가상환경 활성화: source $(VENV)/bin/activate"
	@echo "2. 패키지 설치: make install"
	@echo "3. .env 파일에 OpenAI API 키 설정"
	@echo "4. 개발 서버 실행: make dev"

# 개발 워크플로우 (의존성 확인 후 서버 실행)
run: install dev

# 문서 확인
docs:
	@echo "📖 프로젝트 문서들:"
	@echo "  - README.md: 프로젝트 전체 가이드"
	@echo "  - CONFIG_GUIDE.md: 상세 설정 가이드"
	@echo "  - API 문서: http://localhost:$(PORT)/docs (서버 실행 후)"
