class MyIter:
    def __init__(self):
        self._start = 0

    def __iter__(self):
        self._start = -2
        return self

    def __next__(self):
        self._start += 1
        if self._start < 10:
            return self._start
        else:
            raise StopIteration


if __name__ == '__main__':
    a = MyIter()
    for i in a:
        print(i)
