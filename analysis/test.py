from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Queue as PQueue

my_dict = {
    'url1': 'url2',
    'url3': 'url4',
}

my_q = PQueue()


def test_p(uq):
    q, url = uq[0], uq[1]
    q.put(url, False)


def main():
    global my_dict
    global my_q
    p = Pool(processes=8)

    its = []
    while True:

        # If we go more than 30 seconds without something, die
        try:
            print("Waiting for item from queue for up to 5 seconds")
            i = my_q.get(True, 5)
            print("found %s from the queue !!" % i)
            its.append(i)
        except PQueue.empty():
            print("Caught queue empty exception, done")
            break
    print("processed %d items, completion successful" % len(its))

    p.close()
    p.join()


if __name__ == '__main__':
    main()
