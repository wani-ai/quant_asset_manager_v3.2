# /scripts/update_dynamic_peer_groups.py

import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# 스크립트 실행 경로를 고려하여 sys.path에 프로젝트 루트를 추가해야 할 수 있습니다.
# import sys
sys.path.append('.')
from data.database import get_database_engine, load_data_from_db, save_df_to_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def find_optimal_k(scaled_data, max_k: int = 15):
    """
    K-Means 클러스터링을 위한 최적의 클러스터 개수(k)를
    엘보우 방법(Elbow Method)을 사용하여 찾고 시각화합니다.
    """
    logger = logging.getLogger(__name__)
    logger.info("엘보우 방법을 사용하여 최적의 클러스터 개수(k)를 탐색합니다...")
    
    inertia = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    # 그래프 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid(True)
    # 그래프를 파일로 저장하거나 화면에 표시
    plt.savefig('optimal_k_elbow_plot.png')
    logger.info("엘보우 방법 그래프를 'optimal_k_elbow_plot.png' 파일로 저장했습니다.")
    # plt.show()


def update_dynamic_peer_groups():
    """
    머신러닝 클러스터링을 실행하여 모든 기업의 '동적 피어 그룹' 라벨을
    데이터베이스에 생성 또는 업데이트합니다.
    """
    logger = logging.getLogger(__name__)
    logger.info("동적 피어 그룹 생성을 시작합니다...")

    db_engine = get_database_engine()
    if db_engine is None:
        logger.error("데이터베이스 엔진 연결에 실패했습니다. 작업을 중단합니다.")
        return

    try:
        # --- 1. 특징 데이터 로딩 ---
        logger.info("클러스터링을 위한 재무 특징 데이터를 로딩합니다...")
        # config 파일에서 클러스터링에 사용할 특징(Feature) 목록을 가져옵니다.
        features_for_clustering = config.CLUSTERING_FEATURES
        query = f"SELECT ticker, {', '.join(features_for_clustering)} FROM financial_metrics"
        df = load_data_from_db(query, db_engine)

        if df.empty:
            logger.error("클러스터링을 위한 재무 특징 데이터가 없습니다.")
            return

        df_features = df.set_index('ticker')

        # --- 2. 데이터 전처리 ---
        logger.info("데이터 전처리를 시작합니다 (결측치 처리 및 표준화)...")
        # 결측치를 각 지표의 중앙값(median)으로 대체하여 이상치의 영향을 줄임
        df_imputed = df_features.fillna(df_features.median())
        
        # StandardScaler를 사용하여 데이터를 표준화 (평균 0, 표준편차 1)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_imputed)

        # --- 3. (선택사항) 최적의 k 찾기 ---
        # 처음 실행하거나, 클러스터 개수를 재설정하고 싶을 때 이 함수의 주석을 해제하여 실행합니다.
        # find_optimal_k(features_scaled)

        # --- 4. K-Means 클러스터링 실행 ---
        # config 파일에서 미리 결정된 최적의 k값을 사용
        optimal_k = config.OPTIMAL_K_CLUSTERS
        logger.info(f"{optimal_k}개의 클러스터로 K-Means 모델을 훈련합니다...")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(features_scaled)

        # --- 5. 결과 정리 및 데이터베이스 저장 ---
        result_df = pd.DataFrame(clusters, index=df_imputed.index, columns=['cluster_label'])
        
        logger.info("클러스터링 결과를 'company_peer_groups' 테이블에 저장합니다...")
        logger.info(f"클러스터별 기업 수:\n{result_df['cluster_label'].value_counts()}")
        
        save_df_to_db(result_df, 'company_peer_groups', db_engine, if_exists='replace')

        logger.info("--- 동적 피어 그룹 업데이트가 성공적으로 완료되었습니다. ---")

    except Exception as e:
        logger.error(f"동적 피어 그룹 생성 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    # 이 스크립트를 커맨드 라인에서 직접 실행할 수 있도록 합니다.
    # 예: python -m scripts.update_dynamic_peer_groups
    update_dynamic_peer_groups()
