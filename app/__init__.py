# /app/__init__.py

"""
The app package serves as the main entry point and controller for the
quantitative asset management system.

It initializes all necessary components and provides high-level methods
for running the system's core functionalities, such as ranking stocks,
performing deep-dive analysis, and assessing portfolio risk.

The modules in this package, like dashboard.py and cli.py, use the
QuantSystem class defined here to interact with the rest of the application.
"""

# 시스템의 모든 기능을 관장하는 메인 컨트롤러 클래스를 불러옵니다.
from .quant_system import QuantSystem


# __all__ 변수는 'from app import *' 구문을 사용할 때,
# 어떤 이름들을 외부로 공개할지 명시적으로 정의합니다.
__all__ = [
    'QuantSystem',
]

