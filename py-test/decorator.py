'''
def foo(bar):
    return bar + 1

print(foo)
print(foo(2))
print(type(foo))


def call_foo_with_arg(foo, arg):
    return foo(arg)

print(call_foo_with_arg(foo, 3))


def parent():
    print("Printing from the parent() function.")

    def first_child():
        return "Printing from the first_child() function."

    def second_child():
        return "Printing from the second_child() function."

    print(first_child())
    print(second_child())

def make_averager():
    series = []

    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return total / len(series)

    return averager

if __name__ == "__main__":
    avg = make_averager()
    print(avg(10))
    print(avg(11))
'''


# 修饰器
'''
def makebold(fn):
    def wrapped():
        return '<b>' + fn() + '</b>'

    return wrapped


def makeitalic(fn):
    def wrapped():
        return '<i>' + fn() + '</i>'

    return wrapped

@makebold
@makeitalic
def hello():
    return "Hello World"

print(hello())
'''



'''
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run()
'''

def decorator_factory(enter_message, exit_message):
    # return this decorator
    print("In decorator_factory")

    def simple_deco(func):
        def wrapper():
            print ('enter_message: ',enter_message)
            func()
            print ('exit_message: ',exit_message)
        return wrapper
    return simple_deco

@decorator_factory("Start", "End")
def hello():
    print("Hello World")

hello()
