import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data

class Function:
	# 함수의 인스턴스를 변수 f에 대입해놓고 f() 하면 __call__ 메서드 호출됨
	def __call__(self,input):
		x = input.data
		y = x**2
		output = Variable(y)
		return output

x = Variable(np.array(10))
f = Function()
y = f(x) # __call__ 호출됨 

print(type(y))
print(y.data)

class Function2:
	def __call__(self, input):
		x = input.data
		y = self.forward(x) 
		output = Variable(y)
		return output

	def forward(self, x):
		raise NotImplementedError()
		

class Square(Function2):
	def forward(self, x):
		return x**2

x = Variable(np.array(10))
f = Square()
y = f(x) # __call__ > forward() > Variable()
print(type(y))
print(y.data)
