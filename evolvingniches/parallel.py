from multiprocessing import Queue, Process, queues, get_context


class MultiQueue(queues.Queue):

    def __init__(self):
        ctx = get_context()
        super(MultiQueue, self).__init__(ctx=ctx)
        self.queues = []

    def add(self):
        q = Queue()
        self.queues.append(q)
        return q

    def put(self, obj, block=True, timeout=None):
        # super(MultiQueue, self).put()
        for q in self.queues:
            q.put(obj, block, timeout)

    def close(self) -> None:
        for q in self.queues:
            q.close()
        super().close()

    def join_thread(self) -> None:
        for q in self.queues:
            q.join_thread()
        super().join_thread()

    def cancel_join_thread(self) -> None:
        for q in self.queues:
            q.cancel_join_thread()
        super().cancel_join_thread()


############### VERIFICATION CODE ##################
def _populate_queue(q):
    for i in range(1000):
        q.put(i)
    q.put(False)


def _consume_queue(q, prefix):
    c = q.get()
    while c is not False:
        print(prefix, c)
        c = q.get()


if __name__ == '__main__':
    q = MultiQueue()
    c1 = q.add()
    c2 = q.add()

    p = Process(target=_populate_queue, args=(q,))
    p1 = Process(target=_consume_queue, args=(c1, 'Consumer 1: '))
    p2 = Process(target=_consume_queue, args=(c2, 'Consumer 2: '))

    p.start()
    p1.start()
    p2.start()

    p.join()
    p1.join()
    p2.join()
