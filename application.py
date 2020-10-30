import warnings

from flask import Flask
from server import Server
from argparse import ArgumentParser

warnings.filterwarnings(action='ignore', category=UserWarning)

app = Flask(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    server = Server(app, port=args.port)
    server.run()
