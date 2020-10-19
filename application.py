from flask import Flask
from server import Server

root_dir = "C:\\Users\\MY\\Github\\strabismus-recognition\\"
port = 8080
app = Flask(__name__)

if __name__ == '__main__':
    server = Server(app, port=port, root_dir=root_dir)
    server.run()
