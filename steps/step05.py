# 역전파 이론
"""
* 역전파: 미분 효율적 계산 & 실제값과 오차 더 적음
* 연쇄 법칙 chain rule
    - 합성 함수의 미분 == 구성 함수 각각의 미분을 곱한 것
    - when x > A > a > B > b > C > y,
    - dy/dx = dy/db * db/da * da/dx

* 역전파 원리 도출
    - dy/dx = dy/db * db/da * da/dx
            = dy/da * da/dx
            = dy/dx
    - 즉, 전부 y의 미분값 (= 역전파)
"""






