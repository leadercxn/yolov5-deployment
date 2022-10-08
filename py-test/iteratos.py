mylist = [2,3,4,5]
for i in mylist:
    print(i)

print("next step1:")
mygenerators = (i*i for i in range(3))
for i in mygenerators:
    print(i)

def create_generator():
    for i in range(3):
        yield i*i   # 类似return ,返回的是生成器
        print("我是生成器")

print("next step2:")
mygenerator = create_generator()
for i in mygenerator:
    print(i)
