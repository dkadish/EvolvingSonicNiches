from multiprocessing import Process, Queue, Array, Manager


class MultiClass:
    def __init__(self):
        self.manager = Manager()
        self.list = self.manager.list()

    def f(self,q,q2):
        self.list.append(1)
        q.put([42, None, 'hello'])
        print(q2.get())
        print(self.list)

    def g(self,q,q2):
        self.list.append([0,1,0])
        self.list.append([1,1,1])
        self.list.append([0,0,1])
        print(q.get())
        print(self.list)
        q2.put([43, None, 'bye'])


if __name__ == '__main__':
    mc = MultiClass()

    q = Queue()
    q2 = Queue()
    p = Process(target=mc.f, args=(q,q2))
    p.start()
    # print(q.get())    # prints "[42, None, 'hello']"
    p2 = Process(target=mc.g, args=(q,q2))
    p2.start()
    p.join()
    p2.join()