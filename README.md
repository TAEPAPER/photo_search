# photo-search

자연어로 내 사진첩을 검색하는 토이 프로젝트.
**CLIP** (멀티모달 임베딩) + **Qdrant** (벡터DB) + **FastAPI** + **Streamlit**.


---

## 현재 상태

- [x] 환경 셋업 (uv + Python 3.12 + Apple Silicon MPS)
- [x] CLIP 임베딩 (`photo_search/embedding.py`)
- [x] Qdrant 벡터 스토어 추상화 (`photo_search/vector_store.py`)
- [x] 사진 폴더 인덱싱 CLI (`photo_search/indexer.py`)
- [x] 텍스트 → top-K 검색 CLI (`photo_search/search.py`)
- [x] FastAPI 엔드포인트 (`photo_search/api.py`)
- [x] Streamlit UI (`ui/app.py`)
- [ ] 다국어 검색 (multilingual CLIP)
- [ ] 영상 / YOLO 검출 확장

## 스택

| 영역 | 도구 |
|---|---|
| 임베딩 모델 | OpenAI CLIP — `openai/clip-vit-base-patch32` (512-dim) |
| ML 런타임 | PyTorch + Hugging Face Transformers (4.x) |
| GPU 가속 | Apple Silicon MPS (자동 감지, CUDA / CPU fallback) |
| 벡터DB | Qdrant (Docker) |
| 이미지 I/O | Pillow + `pillow-heif` (HEIC 지원) |
| 패키지 매니저 | uv |

## 폴더 구조

```
photo-search/
├── pyproject.toml          # 의존성 (uv가 관리)
├── uv.lock
├── .python-version         # Python 3.12 핀
├── data/
│   └── photos/             # 인덱싱 대상 (gitignore)
├── photo_search/           # 메인 패키지
│   ├── __init__.py
│   ├── config.py           # 설정 상수 (모델/DB URL/경로/배치 크기)
│   ├── embedding.py        # CLIPEmbedder — 이미지/텍스트 → 512-dim 벡터
│   ├── vector_store.py     # Qdrant 추상화 (ensure/upsert/search/count)
│   ├── indexer.py          # 폴더 스캔 → 임베딩 → DB 저장 (CLI)
│   ├── search.py           # 텍스트 → top-K 사진 (CLI)
│   └── api.py              # FastAPI 앱 (예정)
├── scripts/
│   └── explore_clip.py     # 일회용 탐험 스크립트 (CLIP 단독 동작 확인용)
├── ui/
│   └── app.py              # Streamlit UI (예정)
└── qdrant_storage/         # Qdrant 데이터 볼륨 (gitignore)
```

## 요구사항

- macOS (Apple Silicon 권장 — MPS 가속, CUDA 머신도 동작)
- Python 3.12 이상
- Docker (Qdrant 실행용)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`

## 셋업

### 1. 의존성 설치

```bash
uv sync
```

`pyproject.toml`과 `uv.lock`을 따라 가상환경(`.venv/`)을 만들고 모든 패키지를 설치한다.

### 2. Qdrant 띄우기

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

대시보드: <http://localhost:6333/dashboard>

### 3. 사진 넣기

`data/photos/` 폴더에 JPEG / PNG / HEIC / WebP 등 이미지 파일을 복사한다.
하위 폴더도 자동으로 재귀 스캔한다.

## 사용법

### 인덱싱

```bash
uv run python -m photo_search.indexer
```

- `data/photos/` 전체를 재귀로 스캔
- CLIP으로 일괄 임베딩 → Qdrant에 저장
- 파일 절대경로 → UUID 변환으로 **멱등**. 같은 사진 다시 돌리면 덮어쓰기

### 검색

```bash
uv run python -m photo_search.search "a photo of food"
uv run python -m photo_search.search "a beautiful landscape with mountains"
uv run python -m photo_search.search "a person smiling"
```

top-5 결과가 `점수  파일명` 형태로 출력된다.

## 점수 해석

CLIP 코사인 유사도 (L2 정규화된 벡터들의 내적, 범위 -1 ~ +1).

| 점수 | 해석 |
|---|---|
| `0.40+` | 거의 확실 |
| `0.30 ~ 0.40` | 꽤 강한 매치 |
| `0.20 ~ 0.30` | 관련 있음 |
| `< 0.20` | 약한 관련 또는 무관 |

**절대값보다 "1등과 2등의 격차"가 더 의미 있다.**
격차가 크면 모델이 확신, 격차가 작으면 쿼리가 너무 일반적이거나 후보들이 비슷한 경우.

## 설계 노트

- **CLIP은 영어 모델이다.** 한국어 쿼리도 토큰화는 되지만 검색 품질은 형편없음.
  다국어 모델 (예: `sentence-transformers/clip-ViT-B-32-multilingual-v1`)로
  `MODEL_NAME` 한 줄만 바꾸면 한국어 검색이 작동한다.
- **무거운 객체(`CLIPEmbedder`, `VectorStore`)는 한 번 생성하고 재사용.**
  `PhotoSearcher.default()` 또는 FastAPI lifespan에서 한 번만 만든다.
- **벡터DB 교체 가능.** `vector_store.py`의 4개 메서드
  (`ensure_collection`, `upsert`, `search`, `count`)만 다시 구현하면
  pgvector 등으로 교체 가능. `indexer.py` / `search.py`는 수정 없음.
- **`stable_id()`** — 파일 절대경로의 MD5 해시를 UUID로 변환.
  같은 파일은 항상 같은 ID → 재인덱싱이 안전.

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `PIL.UnidentifiedImageError` | HEIC 파일을 `.jpg`로 위장한 경우. `pillow-heif`는 등록되어 있지만, 파일 자체가 손상됐을 수 있음. `file <경로>` 로 매직 바이트 확인 |
| `AttributeError: 'QdrantClient' object has no attribute 'search'` | qdrant-client 신버전. `query_points`로 마이그레이션 필요 (이미 적용됨) |
| `ModuleNotFoundError: photo_search` | `photo-search/` 디렉터리에서 실행 안 한 경우. `cd ~/Projects/ml/photo-search` 후 `uv run ...` |
| MPS 점수가 CPU와 미세하게 다름 | 정상. MPS 백엔드의 부동소수점 정밀도 차이. 검색 결과엔 영향 없음 |
