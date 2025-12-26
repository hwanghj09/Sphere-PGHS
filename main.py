from fastapi import FastAPI, Request, Query, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import httpx
from datetime import datetime, date
import pytz
import openai
import os
import hashlib
from dotenv import load_dotenv
import json
import shutil
from typing import List, Optional
from pydantic import BaseModel

# .env 파일 로드
load_dotenv()
seoul_tz = pytz.timezone('Asia/Seoul')

# OpenAI API 키 설정
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# 필요한 디렉토리 생성
os.makedirs("meal_img", exist_ok=True)
os.makedirs("lost_and_found_images", exist_ok=True)

# 정적 파일 경로 마운트
app.mount("/static", StaticFiles(directory="meal_img"), name="static")
app.mount("/lost_images", StaticFiles(directory="lost_and_found_images"), name="lost_images")

templates = Jinja2Templates(directory="templates")

MEAL_CACHE_PATH = "meal_cache.json"
meal_cache = {"meal_date": None, "meal_list": [], "neis_error": None}

def load_cached_meals():
    if not os.path.exists(MEAL_CACHE_PATH):
        return None
    try:
        with open(MEAL_CACHE_PATH, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception as e:
        print(f"[MEAL CACHE] 캐시 파일 로드 실패: {e}")
    return None

def save_meal_cache(data: dict):
    try:
        with open(MEAL_CACHE_PATH, "w", encoding="utf-8") as cache_file:
            json.dump(data, cache_file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[MEAL CACHE] 캐시 파일 저장 실패: {e}")

async def fetch_meals_from_neis(target_date: str):
    """나이스 API에서 식단 데이터를 직접 가져옵니다."""
    url = (
        "https://open.neis.go.kr/hub/mealServiceDietInfo"
        f"?Type=json&pIndex=1&pSize=100"
        f"&ATPT_OFCDC_SC_CODE=J10&SD_SCHUL_CODE=7531255&MLSV_YMD={target_date}"
    )
    meal_info_list = []
    neis_error = None

    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(url)
            data = response.json()

            if "mealServiceDietInfo" in data:
                raw_meal_str = data["mealServiceDietInfo"][1]["row"][0]["DDISH_NM"]
                # <br/> 태그 정제 및 리스트화
                meal_info_list = [item.strip() for item in raw_meal_str.split("<br/>") if item.strip()]
                print(f"[SUCCESS] {target_date} 식단 추출 성공: {meal_info_list}")
            else:
                neis_error = "조회된 급식 정보가 없습니다."
                print(f"[INFO] {target_date}: 급식 정보 없음")
    except Exception as e:
        print(f"[ERROR] API 호출 오류: {e}")
        neis_error = "급식 정보를 가져오는 중 오류가 발생했습니다."

    return meal_info_list, neis_error

@app.on_event("startup")
async def initialize_meal_cache():
    # 서버 기동 시 무조건 한국 시간 기준 오늘 날짜로 초기화
    today_str = datetime.now(seoul_tz).strftime("%Y%m%d")
    cached = load_cached_meals()

    if cached and cached.get("meal_date") == today_str:
        meal_cache.update(cached)
        return

    # 캐시가 없거나 날짜가 다르면 새로 가져옴
    meal_list, neis_error = await fetch_meals_from_neis(today_str)
    updated_cache = {
        "meal_date": today_str,
        "meal_list": meal_list,
        "neis_error": neis_error,
    }
    meal_cache.update(updated_cache)
    save_meal_cache(updated_cache)

# --- Agent 모델 정의 ---
class AgentAction(BaseModel):
    type: str
    target: Optional[str] = None
    message: Optional[str] = None

class AgentRequest(BaseModel):
    message: str
    current_path: Optional[str] = "/"

class AgentResponse(BaseModel):
    reply: str
    actions: List[AgentAction]

# --- 루트 엔드포인트 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, meal_description: str = Query(None)):
    # 매 요청마다 오늘 날짜 확인 (날짜가 바뀌었을 경우 대응)
    current_date = datetime.now(seoul_tz).strftime("%Y%m%d")
    
    if meal_cache.get("meal_date") != current_date:
        meal_list, neis_error = await fetch_meals_from_neis(current_date)
        meal_cache.update({"meal_date": current_date, "meal_list": meal_list, "neis_error": neis_error})
        save_meal_cache(meal_cache)

    meal_info_list = meal_cache.get("meal_list", [])
    neis_error = meal_cache.get("neis_error")

    # 이미지 생성 로직
    image_url = None
    if meal_description:
        hasher = hashlib.sha256()
        hasher.update(meal_description.encode("utf-8"))
        filename_base = hasher.hexdigest()[:10]
        filename = f"{current_date}_{filename_base}.png"
        filepath = os.path.join("meal_img", filename)

        if os.path.exists(filepath):
            image_url = f"/static/{filename}"
        else:
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=(
                        "당신은 학교 급식 예시 메뉴를 그리는 에이전트이다. 스테인리스 급식판에 음식을 실사처럼 담아라. "
                        f"메뉴 리스트: {meal_description}"
                    ),
                    size="1024x1024",
                    n=1,
                )
                generated_image_url = response.data[0].url
                async with httpx.AsyncClient(timeout=20.0) as image_client:
                    img_res = await image_client.get(generated_image_url)
                    with open(filepath, "wb") as f:
                        f.write(img_res.content)
                image_url = f"/static/{filename}"
            except Exception as e:
                print(f"[IMAGE ERROR]: {e}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "meal_date": current_date,
            "meal_list": meal_info_list,
            "image_url": image_url,
            "meal_description": meal_description,
            "neis_error": neis_error,
        },
    )

# --- 분실물 관리 엔드포인트 ---
@app.get("/lost")
async def get_lost_items(request: Request):
    lost_items = []
    if os.path.exists("lost_items.json") and os.path.getsize("lost_items.json") > 0:
        with open("lost_items.json", "r", encoding="utf-8") as f:
            lost_items = json.load(f)
    return templates.TemplateResponse("lost_list.html", {"request": request, "lost_items": lost_items})

@app.get("/lost/new")
async def new_lost_item_form(request: Request):
    return templates.TemplateResponse("lost_form.html", {"request": request})

@app.post("/lost/new")
async def create_lost_item(request: Request, item_name: str = Form(...), item_description: str = Form(...), image: UploadFile = File(...)):
    now = datetime.now(seoul_tz)
    image_filename = f"{now.strftime('%Y%m%d%H%M%S')}_{image.filename}"
    image_path = os.path.join("lost_and_found_images", image_filename)
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    lost_items = []
    if os.path.exists("lost_items.json") and os.path.getsize("lost_items.json") > 0:
        with open("lost_items.json", "r", encoding="utf-8") as f:
            lost_items = json.load(f)
            
    new_item = {
        "name": item_name, 
        "description": item_description, 
        "image_url": f"/lost_images/{image_filename}",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
    }
    lost_items.append(new_item)
    
    with open("lost_items.json", "w", encoding="utf-8") as f:
        json.dump(lost_items, f, indent=4, ensure_ascii=False)

    return templates.TemplateResponse("lost_list.html", {"request": request, "lost_items": lost_items})

@app.get("/lost/search")
async def search_lost_items(request: Request, query: str = Query(None)):
    lost_items = []
    if os.path.exists("lost_items.json") and os.path.getsize("lost_items.json") > 0:
        with open("lost_items.json", "r", encoding="utf-8") as f:
            lost_items = json.load(f)
    
    search_results = [item for item in lost_items if query.lower() in item['name'].lower() or query.lower() in item['description'].lower()] if query else lost_items
    return templates.TemplateResponse("lost_search.html", {"request": request, "lost_items": search_results, "query": query})
@app.post("/api/agent", response_model=AgentResponse)
async def ai_agent_endpoint(payload: AgentRequest):
    system_prompt = (
        "너는 Sphere-PGHS의 Zero-touch UI 에이전트이다. 사용자의 명령을 이해해 페이지 이동, 필터 설정, 게시글 작성"
        " 흐름 등 UI 제어를 돕는다. 모든 응답은 JSON 형식이어야 하며, 친근한 한국어 `reply`와 함께 실행할 동작을"
        " `actions` 배열로 제공한다.\n\n"
        "동작 타입:\n"
        "- navigate: `target`에 이동할 경로(URL path) 지정.\n"
        "- announce: 화면 상에 안내만 하고 동작 없음.\n"
        "- search_lost: 분실물 검색을 실행. target에는 검색어를 채우고, 검색 페이지로 이동이 필요하면 /lost/search?q=키워드로 이동.\n"
        "- open_form 또는 open_lost_form: 분실물 등록 폼으로 이동.\n"
        "- fill_field: target에 item_name/item_description을 넣고 message에는 채울 값을 넣어 필드를 완성.\n"
        "- submit_lost_form: 작성된 폼을 제출. 필요 시 open_form→fill_field→submit_lost_form 순서를 붙인다.\n"
        "- focus_search: 검색창을 포커스.\n"
        "- toggle_overlay: target에 open/close를 넣어 AI 오버레이를 열거나 닫음.\n"
        "- open_meal_image: 급식 카드의 '보기' 버튼을 클릭해 이미지를 띄움. target에 급식 메뉴 텍스트를 포함.\n"
        "답변 예시: {\"reply\":\"분실물 등록을 열게요\", \"actions\":[{\"type\":\"open_form\", \"target\":\"/lost/new\"},{\"type\":\"fill_field\",\"target\":\"item_name\",\"message\":\"검은색 지갑\"}]}"
    )

    user_prompt = (
        f"사용자 메시지: {payload.message}\n"
        f"현재 페이지 경로: {payload.current_path}\n"
        "- 분실물 검색 의도는 search_lost로 target에 검색 키워드를 담아라.\n"
        "- 급식 이미지를 열어달라는 요청은 open_meal_image로 대응하고 target은 급식 메뉴 텍스트로 작성하라.\n"
        "- 페이지 이동은 navigate로, 등록 폼 이동은 open_form으로 표시한다. 폼 입력은 fill_field와 submit_lost_form을 사용한다. 반드시 JSON 객체 하나만 응답한다."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    try:
        content = completion.choices[0].message.content
        parsed = json.loads(content)
    except Exception:
        parsed = {"reply": "지금은 요청을 이해하지 못했어요. 다시 한 번 말씀해 주세요!", "actions": [{"type": "announce"}]}

    reply_text = parsed.get("reply", "")
    actions_raw = parsed.get("actions", [])
    actions = []
    for action in actions_raw:
        if isinstance(action, dict) and "type" in action:
            actions.append(AgentAction(type=action.get("type", "announce"), target=action.get("target"), message=action.get("message")))

    if not actions:
        actions = [AgentAction(type="announce", message="확인했어요.")]

    return AgentResponse(reply=reply_text, actions=actions)
