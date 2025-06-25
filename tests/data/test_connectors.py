# /tests/data/test_connectors.py

import pytest
import requests
import time
from unittest.mock import patch, MagicMock

# 테스트할 대상 클래스를 불러옵니다.
from data.connectors import SmartDataManager

# --- 테스트에 사용할 가짜 데이터와 객체를 미리 만들어주는 Pytest Fixture ---

@pytest.fixture
def manager(monkeypatch):
    """
    테스트에 사용될 초기화된 SmartDataManager 인스턴스를 생성합니다.
    .env 파일에서 API 키를 로드하는 것처럼 시뮬레이션합니다.
    """
    # os.getenv가 항상 가짜 키를 반환하도록 설정
    mock_keys = {
        "FMP_API_KEY": "fake_fmp_key",
        "FINNHUB_API_KEY": "fake_finnhub_key",
    }
    monkeypatch.setattr("data.connectors.os.getenv", lambda k: mock_keys.get(k))
    
    return SmartDataManager()

# --- SmartDataManager 클래스를 위한 메인 테스트 클래스 ---

class TestSmartDataManager:

    def test_initialization(self, manager):
        """클래스가 .env 파일의 키를 올바르게 로드하는지 테스트합니다."""
        assert manager.api_keys['fmp'] == "fake_fmp_key"
        assert manager.api_keys['finnhub'] == "fake_finnhub_key"
        assert 'fmp' in manager.rate_limits
        assert isinstance(manager.session, requests.Session)

    @patch('data.connectors.requests.Session.get')
    def test_get_data_success(self, mock_get, manager):
        """
        _get_data 메서드가 API를 성공적으로 호출하고 JSON을 반환하는지 테스트합니다.
        """
        # API가 성공적으로 응답하는 상황을 시뮬레이션
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'symbol': 'AAPL', 'price': 150}
        mock_get.return_value = mock_response

        data = manager._get_data('fmp', 'test_endpoint')

        # 1. API가 실제로 호출되었는지 확인
        mock_get.assert_called_once()
        # 2. 반환된 데이터가 예상과 같은지 확인
        assert data['symbol'] == 'AAPL'
        # 3. API 성공 핸들러가 호출되었는지 확인
        assert manager.api_health['fmp']['consecutive_failures'] == 0

    @patch('data.connectors.requests.Session.get')
    def test_caching_logic(self, mock_get, manager):
        """동일한 요청에 대해 캐시가 올바르게 작동하는지 테스트합니다."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test_data'}
        mock_get.return_value = mock_response

        # 첫 번째 호출 - API를 호출해야 함
        result1 = manager._get_data('fmp', 'cached_endpoint', params={'p': 1})
        assert mock_get.call_count == 1
        assert result1['data'] == 'test_data'

        # 두 번째 호출 - 캐시에서 결과를 가져와야 하므로, API를 다시 호출하면 안 됨
        result2 = manager._get_data('fmp', 'cached_endpoint', params={'p': 1})
        # API 호출 횟수는 여전히 1이어야 함
        assert mock_get.call_count == 1
        assert result2['data'] == 'test_data'

    @patch('data.connectors.time.sleep')
    @patch('data.connectors.requests.Session.get')
    def test_rate_limiting(self, mock_get, mock_sleep, manager):
        """API 호출 제한(Rate Limit) 로직이 올바르게 작동하는지 테스트합니다."""
        manager.rate_limits['fmp'] = (2, 60) # 테스트를 위해 60초에 2회로 제한

        # 3번 연속 호출 시도
        for _ in range(3):
            manager._enforce_rate_limit('fmp')

        # 2번까지는 정상 호출, 3번째 호출 전에 sleep이 호출되어