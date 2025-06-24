# /data/connectors.py

import requests
import time
import logging
import os
import threading
import random
from collections import deque
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class SmartDataManager:
    """
    여러 금융 데이터 API와의 통신을 중앙에서 관리하고,
    API 호출 제한, 캐싱, 데이터 통합을 담당하는 지능형 데이터 관리자.
    강화된 Rate Limiting과 안전장치 포함.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        load_dotenv()
        self.api_keys = {
            'alpha_vantage': os.getenv("ALPHA_VANTAGE_API_KEY"),
            'quandl': os.getenv("QUANDL_API_KEY"),
            'fmp': os.getenv("FMP_API_KEY"),
            'dart': os.getenv("DART_API_KEY"),
            'twelve_data': os.getenv("TWELVE_DATA_API_KEY"),
            'eodhd': os.getenv("EODHD_API_KEY"),
            'polygon': os.getenv("POLYGON_IO_API_KEY"),
            'finnhub': os.getenv("FINNHUB_API_KEY"),
            'serper': os.getenv("SERPER_API_KEY"),
        }
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'QuantitativeAssetManager/1.0'})
        
        self.base_urls = {
            'fmp': 'https://financialmodelingprep.com/api/v3',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'quandl': 'https://data.nasdaq.com/api/v3',
            'finnhub': 'https://finnhub.io/api/v1',
            'polygon': 'https://api.polygon.io',
            'twelve_data': 'https://api.twelvedata.com',
            'eodhd': 'https://eodhistoricaldata.com/api',
            'dart': 'https://opendart.fss.or.kr/api',
            'serper': 'https://google.serper.dev',
        }

        self.rate_limits = {
            'alpha_vantage': (4, 60), 'finnhub': (50, 60), 'polygon': (4, 60),
            'twelve_data': (7, 60), 'quandl': (1800, 600), 'fmp': (20, 60), 'eodhd': (4, 60),
        }
        
        self.api_call_tracker = {api_name: deque() for api_name in self.rate_limits.keys()}
        self.api_locks = {api_name: threading.Lock() for api_name in self.rate_limits.keys()}
        self.cache = {}
        self.cache_ttl = {}
        self.default_cache_duration = 3600
        self.backoff_delays = {api_name: 0 for api_name in self.rate_limits.keys()}
        self.max_retries = 3
        self.api_health = {api_name: {'consecutive_failures': 0, 'last_success': time.time()} 
                           for api_name in self.rate_limits.keys()}
        
        self.logger.info(".env 파일로부터 API 키 로드를 완료했습니다.")
        self.logger.info("Enhanced Rate Limiting이 활성화되었습니다.")

    def _is_cache_valid(self, cache_key: tuple) -> bool:
        if cache_key not in self.cache: return False
        if cache_key not in self.cache_ttl: return True
        return time.time() < self.cache_ttl[cache_key]

    def _set_cache(self, cache_key: tuple, data: Any, ttl_seconds: Optional[int] = None):
        self.cache[cache_key] = data
        duration = ttl_seconds if ttl_seconds is not None else self.default_cache_duration
        self.cache_ttl[cache_key] = time.time() + duration

    def _enforce_rate_limit(self, api_name: str):
        if api_name not in self.rate_limits: return
        with self.api_locks[api_name]:
            limit, period = self.rate_limits[api_name]
            tracker = self.api_call_tracker[api_name]
            while True:
                current_time = time.time()
                while tracker and tracker[0] <= current_time - period:
                    tracker.popleft()
                if len(tracker) < limit:
                    tracker.append(current_time)
                    break
                wait_time = (tracker[0] + period) - current_time
                safety_margin = random.uniform(0.1, 0.5)
                backoff = self.backoff_delays.get(api_name, 0)
                total_wait = wait_time + safety_margin + backoff
                self.logger.warning(
                    f"'{api_name}' API 호출 제한 도달. {total_wait:.2f}초 대기합니다. "
                    f"(기본: {wait_time:.2f}s, 안전마진: {safety_margin:.2f}s, 백오프: {backoff:.2f}s)"
                )
                time.sleep(total_wait)

    def _handle_api_failure(self, api_name: str, error: Exception):
        if api_name not in self.api_health: return
        health = self.api_health[api_name]
        health['consecutive_failures'] += 1
        backoff_time = min(2 ** health['consecutive_failures'], 60)
        self.backoff_delays[api_name] = backoff_time
        self.logger.warning(
            f"'{api_name}' API 연속 실패 {health['consecutive_failures']}회. "
            f"백오프 지연을 {backoff_time}초로 설정합니다."
        )

    def _handle_api_success(self, api_name: str):
        if api_name not in self.api_health: return
        health = self.api_health[api_name]
        health['consecutive_failures'] = 0
        health['last_success'] = time.time()
        self.backoff_delays[api_name] = 0

    def _get_data(self, api_name: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                  cache_ttl: Optional[int] = None) -> Any:
        cache_key = (api_name, endpoint, tuple(sorted(params.items())) if params else ())
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Cache hit for {api_name}/{endpoint}")
            return self.cache[cache_key]
        
        if not self.api_keys.get(api_name):
            self.logger.error(f"'{api_name}' API 키가 설정되지 않았습니다.")
            return None
        
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit(api_name)
                current_params = params.copy() if params else {}
                key_param_map = {
                    'fmp': 'apikey', 'alpha_vantage': 'apikey', 'quandl': 'api_key', 
                    'dart': 'crtfc_key', 'twelve_data': 'apikey', 'eodhd': 'api_token', 
                    'polygon': 'apiKey', 'finnhub': 'token', 'serper': 'api_key'
                }
                if api_name in key_param_map:
                    current_params[key_param_map[api_name]] = self.api_keys.get(api_name)

                url = f"{self.base_urls[api_name]}/{endpoint}"
                self.logger.info(f"[시도 {attempt + 1}/{self.max_retries}] Requesting data from {url}...")
                
                response = self.session.get(url, params=current_params, timeout=30)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    self.logger.warning(f"'{api_name}' API 429 에러. 서버가 요청한 {retry_after}초 후 재시도합니다.")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                
                result = response.json()
                self._handle_api_success(api_name)
                self._set_cache(cache_key, result, cache_ttl)
                return result
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"'{api_name}' API 요청 중 오류 발생 (시도 {attempt + 1}): {e}")
                self._handle_api_failure(api_name, e)
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_delays.get(api_name, 2) + random.uniform(0, 1)
                    self.logger.info(f"{wait_time:.2f}초 후 재시도합니다...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"'{api_name}' API 요청 최대 재시도 횟수 초과")
                    return None
        return None

    def get_nasdaq_symbols(self, cache_hours: int = 24) -> List[str]:
        """
        FMP API를 사용하여 나스닥에 상장된 모든 기업의 티커 리스트를 가져옵니다.
        :param cache_hours: 캐시 유지 시간 (시간)
        :return: 필터링된 티커 심볼 리스트
        """
        self.logger.info("FMP API를 통해 나스닥 상장 기업 리스트를 조회합니다...")
        
        nasdaq_stocks = self._get_data(
            api_name='fmp',
            endpoint='nasdaq_constituent', # 나스닥 구성 종목 전용 엔드포인트
            cache_ttl=cache_hours * 3600
        )
        
        if not nasdaq_stocks:
            self.logger.warning("나스닥 종목 리스트를 가져올 수 없습니다.")
            return []
        
        # FMP 응답에서 'symbol' 키만 추출
        tickers = [stock.get('symbol', '') for stock in nasdaq_stocks]
        
        self.logger.info(f"총 {len(tickers)}개의 나스닥 상장 종목을 가져왔습니다.")
      

        # 디버깅 코드 (유지)
        self.logger.info(f"Finnhub API로부터 총 {len(all_stocks)}개의 종목 정보를 수신했습니다.")
        if len(all_stocks) > 5:
            self.logger.info("수신된 데이터 샘플 (상위 5개):")
            for i in range(5):
                print(all_stocks[i])
        
        # ✨✨✨ 핵심 수정 사항: 필터링 로직 수정 ✨✨✨
        tickers = []
        for stock in all_stocks:
            symbol = stock.get('symbol', '')
            # .upper()를 사용하여 대소문자 구분 없이 비교
            stock_type = stock.get('type', '').upper()
            
            if (stock_type == 'COMMON STOCK' and 
                '.' not in symbol and '$' not in symbol):
                tickers.append(symbol)
        
        self.logger.info(f"총 {len(tickers)}개의 종목을 필터링했습니다.")
        return tickers
    
    # ... (get_historical_market_cap, get_financial_ratios_ttm 등 다른 메서드들은 이전과 동일) ...
    def get_historical_market_cap(self, ticker: str, limit: int = 12, 
                                  cache_hours: int = 6) -> List[Dict[str, Any]]:
        self.logger.info(f"'{ticker}' 종목의 과거 시가총액 데이터를 조회합니다...")
        return self._get_data(
            api_name='fmp',
            endpoint=f"historical-market-capitalization/{ticker}",
            params={'limit': limit},
            cache_ttl=cache_hours * 3600
        )

    def get_financial_ratios_ttm(self, ticker: str, cache_hours: int = 1) -> Optional[Dict[str, Any]]:
        self.logger.info(f"'{ticker}' 종목의 TTM 재무 비율을 조회합니다...")
        data = self._get_data(
            api_name='fmp',
            endpoint=f"ratios-ttm/{ticker}",
            cache_ttl=cache_hours * 3600
        )
        return data[0] if data and isinstance(data, list) and len(data) > 0 else None
        
    def get_historical_prices(self, ticker: str, period: str = "1y", 
                              cache_hours: int = 1) -> pd.DataFrame:
        self.logger.info(f"'{ticker}' 종목의 {period} 가격 데이터를 조회합니다...")
        data = self._get_data(
            api_name='fmp',
            endpoint=f"historical-price-full/{ticker}",
            cache_ttl=cache_hours * 3600
        )
        if data and 'historical' in data:
            df = pd.DataFrame(data['historical'])
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                if period == "1y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=1)]
                elif period == "5y":
                    df = df[df.index >= pd.Timestamp.now() - pd.DateOffset(years=5)]
                return df
        return pd.DataFrame()

    def get_api_status(self) -> Dict[str, Any]:
        status = {}
        for api_name in self.rate_limits.keys():
            limit, period = self.rate_limits[api_name]
            current_calls = len(self.api_call_tracker[api_name])
            health = self.api_health[api_name]
            status[api_name] = {
                'current_calls': current_calls, 'limit': limit, 'period_seconds': period,
                'utilization_percent': (current_calls / limit) * 100,
                'consecutive_failures': health['consecutive_failures'],
                'backoff_delay': self.backoff_delays[api_name],
                'last_success': datetime.fromtimestamp(health['last_success']).strftime('%Y-%m-%d %H:%M:%S')
            }
        return status

    def clear_cache(self):
        self.cache.clear()
        self.cache_ttl.clear()
        self.logger.info("캐시가 초기화되었습니다.")

    def batch_request(self, requests_list: List[Dict], batch_delay: float = 1.0) -> List[Any]:
        results = []
        total_requests = len(requests_list)
        self.logger.info(f"배치 요청 시작: 총 {total_requests}개 요청")
        for i, request in enumerate(requests_list):
            self.logger.info(f"배치 진행률: {i+1}/{total_requests}")
            result = self._get_data(
                api_name=request['api_name'], endpoint=request['endpoint'],
                params=request.get('params'), cache_ttl=request.get('cache_ttl')
            )
            results.append(result)
            if i < total_requests - 1:
                time.sleep(batch_delay)
        self.logger.info("배치 요청 완료")
        return results
