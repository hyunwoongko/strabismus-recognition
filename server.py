from flask import render_template


class Server:
    """Strabismus Detection Application Server"""

    def __init__(self, app):
        self.app = app
        self.front_end()
        self.back_end()

    def front_end(self):
        @self.app.route('/')
        def index():
            return render_template("index.html")

        @self.app.route('/index')
        def index_():
            return render_template("index.html")

        @self.app.errorhandler(404)
        def _404(e):
            return render_template("404.html")

        @self.app.route("/navbar")
        def navbar():
            return render_template("navbar.html")

        @self.app.route("/header")
        def header():
            return render_template("header.html")

        @self.app.route("/footer")
        def footer():
            return render_template("footer.html")

        @self.app.route("/version")
        def version():
            return render_template("version.html")

        @self.app.route("/recognize")
        def recognize():
            return render_template("recognize.html")

        @self.app.route("/manage")
        def manage():
            return render_template("manage.html")

        return self.app

    def back_end(self):

        return self.app
