# AI 배경 합성 스튜디오

SAM(Segment Anything Model)을 활용한 AI 기반 배경 제거 및 합성 웹 애플리케이션입니다.

## 주요 기능

- AI 기반 자동 배경 제거 (SAM 모델 사용)
- 사용자 정의 배경 이미지 합성
- 실시간 크기 조절 및 위치 이동
- 합성된 이미지 다운로드

## 기술 스택

### Backend
- Python 3.10
- Flask
- PyTorch
- Transformers (HuggingFace)
- Segment Anything Model (SAM)
- PIL (Pillow)
- NumPy

### Frontend
- HTML5 Canvas
- Tailwind CSS
- Vanilla JavaScript

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/hwanbit/ai-background-studio.git
cd ai-background-studio
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 필요한 패키지 설치

```bash
pip install flask torch torchvision transformers pillow numpy
```

또는 requirements.txt 파일을 생성하여 설치:

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
ai-background-studio/
│
├── app.py                # Flask 백엔드 서버
├── templates/
│   └── index.html        # 프론트엔드 UI
├── static/
│   └── imgFavicon.ico    # 파비콘
└── requirements.txt      # Python 의존성
```

## 사용 방법

### 1. 서버 실행

```bash
python app.py
```

서버가 시작되면 다음 주소로 접속:
```
http://localhost:5000
```

### 2. 이미지 합성 과정

1. **사람 이미지 업로드**: "사람 이미지" 섹션에서 이미지 파일 선택
2. **AI 배경 제거 실행**: "AI 배경 합성 실행" 버튼 클릭
3. **배경 이미지 선택 (선택사항)**: 원하는 배경 이미지 업로드
4. **크기 및 위치 조정**:
   - 슬라이더로 사람 이미지 크기 조절
   - 캔버스에서 드래그하여 위치 이동
5. **이미지 다운로드**: "이미지 다운로드" 버튼으로 결과 저장

## 주요 특징

### SAM 모델 활용
- Facebook의 Segment Anything Model (SAM) 사용
- 3개의 포인트 프롬프트로 정확한 사람 영역 분할
- IoU 점수 기반 최적 마스크 자동 선택

### 실시간 편집
- Canvas API를 활용한 실시간 렌더링
- 드래그 앤 드롭으로 직관적인 위치 조정
- 슬라이더를 통한 부드러운 크기 조절

### 투명 배경 지원
- RGBA 형식으로 투명 배경 처리
- 배경 미선택 시 체크보드 패턴으로 투명도 표시

## GPU 가속

CUDA 지원 GPU가 있는 경우 자동으로 GPU를 사용하여 추론 속도를 높입니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 주의사항

- 첫 실행 시 SAM 모델 다운로드로 인해 시간이 소요될 수 있습니다 (~360MB)
- GPU가 없는 환경에서는 처리 시간이 다소 길어질 수 있습니다
- 고해상도 이미지는 메모리 사용량이 증가할 수 있습니다
