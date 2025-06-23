# /data/connectors.py

import requests
import time
import logging
import os
from collections import deque
from dotenv import load_dotenv

class SmartDataManager:
    """
    여러 금융 데이터 API와의 통신을 중앙에서 관리하고,
    API 호출 제한, 캐싱, 데이터 통합을 담당하는 지능형 데이터 관리자.
    """
    def __init__(self):
        """
        SmartDataManager의 생성자.
        .env 파일에서 직접 API 키를 로드하고, 모든 데이터 소스에 대한
        연결 설정, 호출 제한 관리자, 로거 등을 초기화합니다.
        """
        # 1단계: 로깅 시스템 설정
        self.logger = logging.getLogger(self.__class__.__name__)
         # 로거 핸들러가 중복 추가되는 것을 방지
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("SmartDataManager 초기화를 시작합니다.")

        # 2단계: .env 파일에서 환경 변수(API 키) 로드
        # 프로젝트 루트의 .env 파일을 찾아 그 안의 변수들을 로드합니다.
        load_dotenv()
        
        # 로드된 환경 변수를 self.api_keys 딕셔너리에 저장
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
        self.logger.info(".env 파일로부터 API 키 로드를 완료했습니다.")

        # 3단계: HTTP 요청을 위한 세션 초기화
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'QuantitativeAssetManager/1.0'})
        self.logger.info("HTTP 요청 세션을 초기화했습니다.")

        # 4단계: API별 특성 중앙 관리 (URL 및 호출 제한 규칙)
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
        
        # 각 API의 무료 요금제 호출 제한 규칙 정의
        self.rate_limits = {
            'alpha_vantage': (5, 60),      # 60초에 5번
            'finnhub': (60, 60),           # 60초에 60번
            'polygon': (5, 60),            # 60초에 5번
            'twelve_data': (8, 60),      # 60초에 8번
            'quandl': (3000, 600),         # 10분에 2000건
            # 일일 제한이 더 중요한 API는 분당 제한을 보수적으로 설정하여 버스트 콜 방지
            'fmp': (30, 60),
            'eodhd': (5, 60),
        }
        self.logger.info("API별 기본 URL 및 호출 제한 규칙 설정을 완료했습니다.")

        # 5단계: 지능형 호출 제어(Rate Limiting) 시스템 초기화
        self.api_call_tracker = {
            api_name: deque() for api_name in self.rate_limits.keys()
        }
        self.logger.info("API 호출 제어(Rate Limiting) 시스템을 초기화했습니다.")
        
        # 6단계: 인메모리 캐시 설정
        self.cache = {}
        self.logger.info("인메모리 캐시를 활성화했습니다.")

        self.logger.info("SmartDataManager 초기화가 성공적으로 완료되었습니다. 모든 데이터 채널이 준비되었습니다.")

def _enforce_rate_limit(self, api_name: str):
        """API 호출 전에 호출 제한을 확인하고, 필요시 대기하는 내부 메서드."""
        if api_name not in self.rate_limits:
            return

        limit, period = self.rate_limits[api_name]
        tracker = self.api_call_tracker[api_name]
        
        current_time = time.time()
        
        # 'period'초가 지난 오래된 타임스탬프를 큐에서 제거
        while tracker and tracker[0] <= current_time - period:
            tracker.popleft()
            
        # 현재 윈도우 내의 호출 횟수가 제한에 도달했는지 확인
        if len(tracker) >= limit:
            wait_time = (tracker[0] + period) - current_time
            if wait_time > 0:
                self.logger.warning(f"'{api_name}' API 호출 제한 도달. {wait_time:.2f}초 대기합니다.")
                time.sleep(wait_time)
        
        # 새로운 호출 타임스탬프 기록
        tracker.append(time.time())

def get_data(self, api_name: str, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        중앙 집중식 데이터 요청 메서드. 호출 제한, 캐싱 등을 자동으로 처리합니다.
        """
        # 1. 캐시 확인 로직
        # 파라미터를 정렬하여 순서에 상관없이 동일한 캐시 키를 생성
        if params:
            cache_key = (api_name, endpoint, tuple(sorted(params.items())))
        else:
            cache_key = (api_name, endpoint)
            
        if cache_key in self.cache:
            self.logger.info(f"Cache hit for {api_name}/{endpoint}. Returning cached data.")
            return self.cache[cache_key]

        # 2. 호출 제한 준수
        self._enforce_rate_limit(api_name)
        
        # 3. API 키 자동 추가
        # 파라미터가 없으면 빈 딕셔너리로 초기화 (중복 제거)
        if params is None:
            params = {}
        
        # key_param_map을 사용하여 확장 가능한 방식으로 키 추가 (중복 제거)
        key_param_map = {
            'alpha_vantage': 'apikey', 'quandl': 'api_key', 'fmp': 'apikey',
            'dart': 'crtfc_key', 'twelve_data': 'apikey', 'eodhd': 'api_token',
            'polygon': 'apiKey', 'finnhub': 'token', 'serper': 'api_key'
        }
        if api_name in key_param_map:
            param_name = key_param_map[api_name]
            params[param_name] = self.api_keys.get(api_name)

        # 4. URL 생성 및 API 요청
        # URL 생성 로직을 if 블록 밖으로 이동
        url = f"{self.base_urls[api_name]}/{endpoint}"
        
        try:
            # 단일 try...except 블록으로 통합
            self.logger.info(f"Cache miss. Requesting data from {url}...")
            response = self.session.get(url, params=params)
            response.raise_for_status() # 2xx 상태 코드가 아니면 예외 발생
            
            result = response.json()
            
            # 5. 성공적인 응답을 캐시에 저장
            self.cache[cache_key] = result
            
            return result
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"'{api_name}' API 요청 중 오류 발생: {e}")
            return None