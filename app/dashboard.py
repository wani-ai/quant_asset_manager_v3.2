# /app/dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 시스템의 모든 기능을 관장하는 메인 컨트롤러 클래스를 불러옵니다.
from app import QuantSystem

# -----------------------------------------------------------------------------
# Streamlit 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="퀀트 자산 관리 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 캐싱: 시스템의 핵심 객체 및 데이터 로드를 최적화합니다.
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    """
    QuantSystem 객체를 한 번만 로드하여 애플리케이션 전체에서 재사용합니다.
    모델 로딩 등 무거운 초기화 작업을 반복하지 않게 해줍니다.
    """
    system = QuantSystem()
    return system

@st.cache_data
def get_top_stocks(strategy, top_n):
    """
    랭킹 데이터를 캐시하여 동일한 요청에 대해 시스템을 반복 호출하는 것을 방지합니다.
    """
    return system.get_top_ranked_stocks(strategy=strategy, top_n=top_n)

@st.cache_data
def get_deepdive_report(ticker):
    """종목 심층 분석 리포트를 캐시합니다."""
    return system.get_stock_deep_dive_report(ticker=ticker)

@st.cache_data
def get_portfolio_risk(tickers):
    """포트폴리오 리스크 분석 결과를 캐시합니다."""
    return system.get_portfolio_risk_dashboard(portfolio_tickers=tickers)


# -----------------------------------------------------------------------------
# 각 대시보드 페이지를 그리는 함수들
# -----------------------------------------------------------------------------

def display_summary_dashboard():
    """페이지 1: 총괄 현황 대시보드"""
    st.subheader("종합 현황 대시보드")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### 📈 Top-Rated 주식 랭킹")
        strategy = st.selectbox(
            "투자 전략 선택",
            options=['blend', 'value', 'growth'],
            index=0,
            help="종합 점수 산출 시 사용할 가중치 프로필을 선택합니다."
        )
        top_n = st.slider("조회할 상위 주식 개수", 5, 50, 10)

        with st.spinner(f"'{strategy}' 전략 기반 상위 {top_n}개 주식을 분석 중입니다..."):
            ranked_df = get_top_stocks(strategy, top_n)
            st.dataframe(ranked_df)

    with col2:
        st.markdown("##### 🛡️ 포트폴리오 리스크")
        # 예시 포트폴리오 (실제로는 사용자 입력 또는 DB 연동)
        default_portfolio = "AAPL, MSFT, GOOGL, NVDA, AMZN, TSLA"
        portfolio_input = st.text_area("분석할 포트폴리오 티커 (쉼표로 구분)", value=default_portfolio)
        
        if st.button("포트폴리오 리스크 분석"):
            tickers = [ticker.strip().upper() for ticker in portfolio_input.split(',')]
            with st.spinner(f"{len(tickers)}개 종목의 포트폴리오 리스크를 분석 중입니다..."):
                risk_data = get_portfolio_risk(tuple(tickers)) # 리스트는 캐시 안되므로 튜플로 변환

                if risk_data:
                    risk_score = risk_data.get('composite_risk_score', 0)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        title = {'text': "종합 리스크 점수 (낮을수록 안전)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "black"},
                            'steps' : [
                                 {'range': [0, 40], 'color': "green"},
                                 {'range': [40, 70], 'color': "yellow"},
                                 {'range': [70, 100], 'color': "red"}],
                        }))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    cvar = risk_data.get('cvar_99', 0) * 100
                    mdd = risk_data.get('max_drawdown', 0) * 100
                    sharpe = risk_data.get('sharpe_ratio', 0)

                    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                    kpi_col1.metric("99% CVaR (월)", f"{cvar:.2f}%")
                    kpi_col2.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%")
                    kpi_col3.metric("샤프 지수 (Sharpe)", f"{sharpe:.2f}")


def display_stock_deep_dive():
    """페이지 2: 개별 종목 심층 분석"""
    st.subheader("개별 종목 심층 분석")
    ticker_input = st.text_input("분석할 주식 티커를 입력하세요 (예: AAPL)", value="AAPL").upper()
    
    if st.button(f"{ticker_input} 심층 분석 실행"):
        with st.spinner(f"'{ticker_input}'에 대한 모든 분석 데이터를 가져오는 중입니다..."):
            report = get_deepdive_report(ticker_input)
            
            if report and 'error' not in report:
                st.markdown(f"### **{report.get('ticker')} 분석 리포트**")
                
                # 각 분석 카테고리를 expander 안에 넣어 UI를 깔끔하게 구성
                with st.expander("수익성(Profitability) 분석", expanded=True):
                    st.json(report.get('profitability'))

                with st.expander("안정성(Stability) 분석"):
                    st.json(report.get('stability'))

                with st.expander("유동성(Liquidity) 분석"):
                    st.json(report.get('liquidity'))
                    
                with st.expander("효율성(Efficiency) 분석"):
                    st.json(report.get('efficiency'))

                with st.expander("성장성(Growth) 분석"):
                    st.json(report.get('growth'))
                    
                with st.expander("가치평가(Valuation) 분석"):
                    st.json(report.get('valuation'))
                    
                with st.expander("기술적 분석"):
                    st.json(report.get('technical_analysis'))
                    
                with st.expander("변동성 리스크 분석"):
                    st.json(report.get('volatility_analysis'))
            else:
                st.error(f"'{ticker_input}'에 대한 데이터를 가져오는 데 실패했습니다. 티커를 확인해주세요.")

# -----------------------------------------------------------------------------
# 메인 애플리케이션 로직
# -----------------------------------------------------------------------------

# 1. 시스템 객체 로드 (캐싱됨)
system = load_system()

# 2. 사이드바 네비게이션 메뉴 생성
st.sidebar.title("퀀트 시스템 v3.2")
page = st.sidebar.radio(
    "메뉴를 선택하세요",
    ("종합 현황", "개별 종목 심층 분석")
)

# 3. 선택된 페이지에 따라 해당 함수 호출
if page == "종합 현황":
    display_summary_dashboard()
elif page == "개별 종목 심층 분석":
    display_stock_deep_dive()

