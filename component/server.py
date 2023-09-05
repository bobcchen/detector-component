from component import Component

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--service', default='A')
    parser.add_argument('--next_service', default=None)
    parser.add_argument('--init_retries', default=5, type=int)
    parser.add_argument('--pipeline_id', default=0, type=str)
    args = parser.parse_args()

    server = Component(args)

    run = True

    def handler(signum, frame):
        global run
        run = False


    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    while run:
        server.run()

    server.shutdown()
