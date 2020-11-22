import os
import warnings

from flask import render_template, request
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from engine.dataset import Dataset
from engine.model import Model

warnings.filterwarnings(action="ignore", category=UserWarning)


class Server:
    """Strabismus Detection Application Server"""

    def __init__(self, app, port):
        self.app = app
        self.port = port
        self.front_end()
        self.back_end()

    def front_end(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/index")
        def index_():
            return render_template("index.html")

        @self.app.errorhandler(404)
        def _404(e):
            return render_template("404.html")

        @self.app.errorhandler(500)
        def _500(e):
            return render_template("500.html")

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
            return render_template("recognize.html", port=self.port)

        @self.app.route("/charts")
        def charts():
            return render_template("legacy/charts.html")

        return self.app

    def back_end(self):
        @self.app.route("/file_upload", methods=["GET", "POST"])
        def upload_file():
            f = request.files["file"]
            filename = "cache/{}".format(f.filename)
            f.save(filename)

            output_list, img_file_name = self.recognize(filename)
            return render_template(
                "result.html",
                port=self.port,
                output_list=output_list,
                img_file_name=img_file_name,
            )

        return self.app

    def recognize(self, file_name):
        model_dir = os.listdir("saved")
        dataset = Dataset()
        dataset, data_vis = dataset.eval(file_name)
        img_file_name = self.visualize(data_vis, file_name)
        img_file_name = img_file_name.split("cache")
        img_file_name = img_file_name[1].replace("/", "")

        output_list = []
        for model_name in model_dir:
            model = Model(
                model_dir="saved",
                model_id=model_name.replace(".pkl", ""),
                model=RandomForestClassifier(n_estimators=50),
            )
            model.load()
            output = model.predict(dataset)[0]
            output_list.append(output)

        return output_list, img_file_name

    def visualize(self, dataset, file_name):
        file_name = file_name.replace("cache", "static/cache")
        file_name = file_name.replace("csv", "jpg")
        plt.plot(dataset, c="r")
        plt.ylim(-200, 200)
        plt.title("Fixation Data Sequence")
        plt.xlabel("time step")
        plt.ylabel("Right - Left")
        plt.savefig(file_name)
        plt.close()

        return file_name

    def run(self):
        self.app.run(host="0.0.0.0", port=self.port)
