# chapter 1
## 1.4.2
class Man:
	def __init__(self, name):
		self.name = name
		print("initialized!")

	def hello(self):
		print("Hello " + self.name + "!")

	def goodbye(self):
		print("Good-bye " + self.name + "!")

david = Man("David")
david.hello()
david.goodbye()