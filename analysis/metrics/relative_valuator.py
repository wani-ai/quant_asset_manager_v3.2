# /analysis/relative_valuator.py

import pandas as pd
import numpy as np
import logging
from sqlalchemy import Engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
from data.database import load_data_from_db

class RelativeValuator:
    """
    듀얼 평가 시스템을 사용하여 기업의 상대적 가치와 절대적 퀄리티를 평가하는 모듈.
    1. 머신러닝 클러스터링을 통해 '동적 피어 그룹'을 생성하고, 그룹 내에서 상대 평가를 수행합니다.
    2. '꾸준한 성장 기업' 벤치마크와 비교하여 절대적인 퀄리티를 평가합니다.
    """
    def __init__(self, db_engine: Engine):
        """
        RelativeValuator의 생성자.

        :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
        """
        self.db_engine = db_engine
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quality_benchmark = self._load_quality_benchmark()

    def _load_financial_data(self) -> pd.DataFrame:
        """분석에 필요한 모든 기업의 재무 지표를 데이터베이스에서 로드합니다."""
        self.logger.info("데이터베이스에서 전체 재무 지표 데이터를 로드합니다...")
        # 가정: financial_metrics 테이블에 모든 필요한 지표가 저장되어 있음.
        query = "SELECT * FROM financial_metrics"
        df = load_data_from_db(query, self.db_engine)
        return df.set_index('ticker')

    def _load_quality_benchmark(self) -> pd.Series:
        """'꾸준한 성장 기업'의 평균 재무 프로필(퀄리티 벤치마크)을 로드합니다."""
        self.logger.info("퀄리티 벤치마크 데이터를 로드합니다...")
        query = "SELECT * FROM quality_benchmark LIMIT 1"
        benchmark_df = load_data_from_db(query, self.db_engine)
        if benchmark_df.empty:
            self.logger.warning("퀄리티 벤치마크 데이터가 없습니다. 절대 퀄리티 평가는 생략됩니다.")
            return None
        return benchmark_df.iloc[0]

    def _create_dynamic_peer_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """K-Means 클러스터링을 사용하여 재무적 특성이 유사한 동적 피어 그룹을 생성합니다."""
        self.logger.info("머신러닝 클러스터링을 사용하여 동적 피어 그룹을 생성합니다...")
        
        # 클러스터링에 사용할 특징(Feature) 선택 (config.py에서 관리)
        features_for_clustering = self.config.CLUSTERING_FEATURES
        
        data_for_clustering = df[features_for_clustering].dropna()
        
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
        
        # K-Means 실행 (최적의 k는 사전 분석 또는 config에서 결정)
        k = self.config.OPTIMAL_K_CLUSTERS
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled_data)
        
        # 원본 데이터프레임에 클러스터 라벨 추가
        data_for_clustering['cluster_label'] = clusters
        df = df.join(data_for_clustering['cluster_label'])
        
        self.logger.info(f"{k}개의 동적 피어 그룹 생성을 완료했습니다.")
        return df

    def _calculate_relative_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """클러스터 그룹 내에서 각 지표의 Z-점수를 계산하여 상대 강도 점수를 산출합니다."""
        self.logger.info("피어 그룹 내 상대 강도 점수(Z-Score)를 계산합니다...")
        
        scored_df = df.copy()
        metrics_to_score = self.config.METRICS_FOR_SCORING # config 파일에 정의된 평가 지표 목록
        
        for metric, higher_is_better in metrics_to_score.items():
            if metric in scored_df.columns:
                mean = scored_df.groupby('cluster_label')[metric].transform('mean')
                std = scored_df.groupby('cluster_label')[metric].transform('std')
                
                # std가 0인 경우(그룹 내 모든 값이 동일) Z-점수는 0으로 처리
                z_score = (scored_df[metric] - mean) / std
                z_score.fillna(0, inplace=True)
                
                if not higher_is_better:
                    z_score = z_score * -1 # 방향성 통일 (높을수록 좋게)
                
                scored_df[f'{metric}_z_score'] = z_score

        return scored_df

    def _calculate_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """'퀄리티 벤치마크'와 비교하여 절대 퀄리티 점수를 산출합니다."""
        if self.quality_benchmark is None:
            df['quality_score'] = 0
            return df

        self.logger.info("'챔피언 그룹' 벤치마크와 비교하여 절대 퀄리티 점수를 계산합니다...")
        
        quality_scores = []
        for index, row in df.iterrows():
            score = 0
            # 예시: ROE, 부채비율, 영업이익률 3가지 항목을 벤치마크와 비교
            if row['roe'] > self.quality_benchmark.get('avg_roe', 0):
                score += 1
            if row['debtToEquity'] < self.quality_benchmark.get('avg_debt_to_equity', 999):
                score += 1
            if row['operatingMargin'] > self.quality_benchmark.get('avg_operating_margin', 0):
                score += 1
            quality_scores.append(score)
            
        df['quality_score'] = quality_scores
        # 0~100점 척도로 정규화 (최대 점수는 비교 항목 수)
        df['quality_score'] = (df['quality_score'] / 3) * 100
        return df

    def run_full_valuation(self, strategy: str = 'blend') -> pd.DataFrame:
        """
        듀얼 평가 시스템의 전체 워크플로우를 실행하여 최종 랭킹을 반환합니다.
        """
        try:
            # 1. 데이터 로드
            financial_df = self._load_financial_data()
            
            # 2. 동적 피어 그룹 생성
            clustered_df = self._create_dynamic_peer_groups(financial_df)
            
            # 3. 상대 강도 점수 계산
            relative_scored_df = self._calculate_relative_scores(clustered_df)
            
            # 4. 절대 퀄리티 점수 계산
            dual_scored_df = self._calculate_quality_scores(relative_scored_df)

            # 5. 최종 종합 점수 산출
            self.logger.info("카테고리별 점수 및 최종 종합 점수를 산출합니다...")
            strategy_weights = self.config.STRATEGY_WEIGHTS[strategy]
            
            # 카테고리별 Z-점수 평균 계산
            for category, metrics in self.config.METRIC_CATEGORIES.items():
                score_cols = [f'{m}_z_score' for m in metrics if f'{m}_z_score' in dual_scored_df.columns]
                if score_cols:
                    dual_scored_df[f'{category}_score'] = dual_scored_df[score_cols].mean(axis=1)
            
            # 펀더멘털 종합 점수 (상대 강도)
            dual_scored_df['fundamental_score'] = 0
            for category, weight in strategy_weights.items():
                if f'{category}_score' in dual_scored_df.columns:
                     dual_scored_df['fundamental_score'] += dual_scored_df[f'{category}_score'] * weight

            # 최종 점수 (상대 점수 + 절대 점수)
            final_weights = self.config.FINAL_COMPOSITE_WEIGHTS
            dual_scored_df['final_score'] = (
                dual_scored_df['fundamental_score'] * final_weights['relative'] +
                dual_scored_df['quality_score'] * final_weights['absolute']
            )

            # 6. 랭킹
            dual_scored_df['rank'] = dual_scored_df['final_score'].rank(ascending=False, method='min')
            
            # 결과 정렬 및 반환
            final_cols = ['rank', 'final_score', 'fundamental_score', 'quality_score']
            final_df = dual_scored_df.sort_values('rank')[final_cols]

            return final_df

        except Exception as e:
            self.logger.error(f"상대가치평가 중 오류 발생: {e}")
            return pd.DataFrame()

