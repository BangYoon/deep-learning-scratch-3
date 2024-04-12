# 17. 메모리 관리와 순환 참조
"""
17.1 메모리 관리
- 파이썬은 필요 없어진 객체를 메모리 자동 삭제함
- 그럼에도 memory leak / out of memory 발생할 수 있음
- CPython 메모리 관리 방식
    (1) reference 개수 세기 (참조 카운드)
    (2) generation 기준으로 쓸모없는 객체 회수하기 ("GC" - Garbage Collection)

17.2 참조 카운트의 메모리 관리
- 모든 객체는 참조 카운트 0 상태로 생성됨
- 다른 객체가 참조할 때마다 +1  ex) 대입연산자, 함수 인수, 컨테이너 타입 객체 추가(list/tuple/class)
- 참조 끊길 때마다 -1 / 0 되면 인터프리터가 회수함


17.3 순환 참조 circular reference
- 구조 복잡함
- 메모리 부족 시 인터프리터가 GC 자동 호출함  or  gc.collect() 로 호출 가능함
- DeZero의 Variable <-> Function 사이에 순환 참조 존재함 (inputs, outputs, creator)
-> weakref 로 해결하기

17.4 weakref 모듈
- 약한 참조 만듦 : 다른 객체를 참조하되, 카운트는 증가 X
"""

class obj:
    pass

def f(X):
    print(X)

# Ex1
a = obj()   # 변수에 대입: cnt=1
f(a)        # 함수에 전달: cnt=2
            # 함수 완료: cnt=1
a = None    # 대입 해제: cnt=0 -> 메모리 삭제

# Ex 17.2
a = obj()   # a cnt=1
b = obj()   # b cnt=1
c = obj()   # c cnt=1
a.b = b     # b cnt=2
b.c = c     # c cnt=2
a = b = c = None # a=0 b=0 c=0

# Ex 17.3
a = obj()   # a cnt=1
b = obj()   # b cnt=1
c = obj()   # c cnt=1
a.b = b     # b cnt=2
b.c = c     # c cnt=2
c.a = a     # a cnt=2
a = b = c = None # a=1 b=1 c=1 -> 사용자는 셋 다 접근할 수 없음 -> GC 등장

# Ex 17.4
import weakref
import numpy as np

a = np.array([1,2,3])   # general ref
b = weakref.ref(a)      # weak ref
b   # <weakref at 0xxxxxx; to 'numpy.ndarray' at 0xxxxxx>
b() # [1 2 3]

a = None  # cnt=0
b   # <weakref at 0xxxxxx; dead> cnt=0
