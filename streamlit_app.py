#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
import re
import requests
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from pytrends.request import TrendReq
import matplotlib.font_manager as fm
import matplotlib as mpl
import streamlit as st

# === ✅ 한글 폰트 자동 설정 ===
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

def set_korean_font():
    font_path = ""
    font_candidates = ['malgun.ttf', 'NanumGothic.ttf', 'AppleGothic.ttf']

    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(fname=font_name))
            if font_path:
                break
        except:
            continue

    if font_path:
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rc('font', family=font_name)
        st.success(f"✅ 한글 폰트 설정됨: {font_name}")
    else:
        st.warning("⚠️ 한글 폰트가 설정되지 않았습니다. 기본 폰트를 사용합니다.")

set_korean_font()

# === Clova API 설정 ===
api_key = "nv-b21936503dc049a488669d28299ad294s0i2"
invoke_url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"

# 1. 데이터 불러오기
df = pd.read_excel("매매기록_랜덤시간적용.xlsx")
df["매수일"] = pd.to_datetime(df["매수일"])
df["매도일"] = pd.to_datetime(df["매도일"])
df["보유일수"] = (df["매도일"] - df["매수일"]).dt.days

# 2. 문장화 및 임베딩
def convert_trade_to_text(row):
    return f"{row['매수일'].strftime('%Y-%m-%d')}에 {row['종목코드']} 종목을 매수하여 {row['매도일'].strftime('%Y-%m-%d')}에 매도, 수익률 {row['수익률']}%, RSI {row['RSI']}, MA5: {row['MA5']}"
df["문장"] = df.apply(convert_trade_to_text, axis=1)

model = SentenceTransformer("jhgan/ko-sbert-nli")
embeddings = model.encode(df["문장"].tolist(), convert_to_tensor=False)
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(embeddings)

# 3. 날짜 추출
def extract_dates_from_question(question):
    pattern = r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일(?:부터)?\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일'
    match = re.search(pattern, question)
    if match:
        y1, m1, d1, y2, m2, d2 = map(int, match.groups())
        return datetime(y1, m1, d1), datetime(y2, m2, d2)
    return None, None

# 4. Clova X 호출
def clova_x_generate(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "당신은 투자 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
    res = requests.post(invoke_url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()['result']['message']['content']
    else:
        return f"❌ Clova 호출 실패: {res.status_code}"

# 5. Clova 기반 키워드 확장
def expand_keywords_with_clova(user_keyword):
    prompt = (
        f"'{user_keyword}'이라는 키워드의 실사용 유사 표기들을 2~5개 리스트로 추천해줘. "
        f"예: '네 이버' → ['네이버', 'naver', 'NAVER']\n"
        f"오타, 기업 전체 이름, 띄어쓰기 오차 정도까지만 포함하고, 반드시 Python 리스트 형식으로만 출력해줘."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "당신은 검색 키워드 정제 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        "maxTokens": 150,
        "temperature": 0.3
    }
    try:
        res = requests.post(invoke_url, headers=headers, json=payload)
        if res.status_code != 200:
            return [user_keyword]
        result = res.json()['result']['message']['content'].strip()
        match = re.search(r"\[(.*?)\]", result)
        if match:
            items = match.group(1).split(',')
            cleaned = [item.strip().strip("'\"") for item in items if item.strip()]
            return cleaned if cleaned else [user_keyword]
        else:
            return [user_keyword]
    except Exception:
        return [user_keyword]

# 6. 트렌드 분석
def keyword_trend_analysis_multi(user_input, start_date, end_date):
    keywords = expand_keywords_with_clova(user_input)
    st.write(f"🧾 확장된 키워드: {keywords}")
    pytrends = TrendReq(hl='ko', tz=540)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    pytrends.build_payload(keywords, timeframe=timeframe, geo='KR')
    trend_df = pytrends.interest_over_time().reset_index()
    if trend_df.empty:
        st.warning("⚠️ 트렌드 데이터 없음")
        return [], len(df)
    trend_df["통합검색량"] = trend_df[keywords].max(axis=1)
    threshold = trend_df["통합검색량"].mean() + trend_df["통합검색량"].std()
    trend_df["급등"] = trend_df["통합검색량"] > threshold
    급등일 = trend_df[trend_df["급등"]]["date"].tolist()
    매치된_매매 = []
    for trade_day in df["매수일"]:
        for surge_day in 급등일:
            if abs((trade_day.date() - surge_day.date()).days) <= 2:
                매치된_매매.append(trade_day.date())
                break
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend_df["date"], trend_df["통합검색량"], label="검색량")
    ax.axhline(threshold, color='red', linestyle='--', label="급등 기준선")
    ax.set_title(f"'{', '.join(keywords)}' 검색 트렌드")
    ax.set_xlabel("날짜")
    ax.set_ylabel("검색량")
    ax.legend()
    st.pyplot(fig)
    return 매치된_매매, len(df)

# 7. 유사 문장 검색
def retrieve_top_k(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=k)
    return df.iloc[indices[0]]

# 8. Clova 분석 프롬프트 생성
def make_clova_prompt(query, retrieved_df):
    combined = "\n".join(retrieved_df["문장"].tolist())
    return (
        f"사용자 질문: {query}\n"
        f"참고할 유사 매매 기록:\n{combined}\n"
        f"이 유사 매매 기록을 바탕으로 다음을 분석해주세요:\n"
        f"1. 반복된 손실 원인\n"
        f"2. 보완할 점\n"
        f"3. 수익을 낸 전략의 공통점\n"
        f"간결하고 실용적인 투자 조언을 중심으로 답해주세요."
    )

# 9. Streamlit 앱 실행
def main():
    st.title("📈 투자 복기 & 키워드 분석")
    user_input = st.text_input("❓ 분석 질문 입력 (예: 2024년 5월 30일부터 2024년 6월 2일까지 매매기록 분석)")
    if user_input:
        start, end = extract_dates_from_question(user_input)
        if not (start and end):
            st.warning("날짜 인식 실패. 다시 입력해주세요.")
            return
        st.success(f"🗓️ 분석 기간: {start.date()} ~ {end.date()}")
        keyword = st.text_input("🔎 키워드 입력 (예: 네 이버)")
        if keyword:
            matches, total_trades = keyword_trend_analysis_multi(keyword, start, end)
            st.info(f"📊 '{keyword}' 검색 급등일 반응 매매: {len(matches)}건 / 총 {total_trades}건")
        sub_df = df[(df["매수일"] >= start) & (df["매수일"] <= end)]
        if not sub_df.empty:
            summary_prompt = "\n".join(sub_df["문장"].tolist())
            st.subheader("🧠 Clova 분석 결과")
            st.write(clova_x_generate(summary_prompt))
        else:
            st.warning("해당 기간 매매 없음 → 유사 기록 분석")
            retrieved = retrieve_top_k(user_input)
            prompt = make_clova_prompt(user_input, retrieved)
            st.subheader("🧠 Clova 분석 결과")
            st.write(clova_x_generate(prompt))

# 실행
if __name__ == "__main__":
    main()



