# /ml_models/training/__init__.py

"""
The training sub-package is responsible for all machine learning and
deep learning model training pipelines.

The scripts within this package take the features engineered by the 'analysis'
package as input, and train predictive models to forecast stock returns,
rankings, or future values of sector-level metrics.

Key responsibilities include:
- Loading feature and target data from the database.
- Splitting data into training, validation, and test sets.
- Training models (e.g., XGBoost for ranking, LSTM/Autoformer for time series).
- Evaluating model performance.
- Saving the trained models to the 'ml_models/saved_models/' directory for
  later use in prediction pipelines.

This __init__.py file marks the directory as a Python package.
"""

# 이 파일은 'training' 디렉토리를 파이썬 패키지로 만들기 위해 존재하며,
# 일반적으로 코드를 포함하지 않습니다.

