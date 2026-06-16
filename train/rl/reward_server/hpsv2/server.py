"""HPSv2 reward server — human preference score.

Scores generated images against text prompts using HPSv2 (ViT-H-14 based).

Launch with gunicorn::

    NUM_DEVICES=4 GPU_IDS="[0,1,2,3]" PORT=8084 HPSV2_CKPT=/path/to/hps.pt CLIP_PATH=/path/to/clip \
        gunicorn -c gunicorn.conf.py server:app
"""

import pickle
import traceback

from flask import Flask, request, Blueprint
from PIL import Image
from io import BytesIO

from evaluator import load_hpsv2

root = Blueprint("root", __name__)


def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_hpsv2()
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "service": "hpsv2"}, 200


@root.route("/", methods=["POST"])
def inference():
    data = request.get_data()
    if not data:
        return {"error": "empty request body"}, 400
    try:
        payload = pickle.loads(data)
        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in payload["images"]]
        prompts = payload["prompts"]
        scores = INFERENCE_FN(prompts, images)
        return pickle.dumps({"scores": scores}), 200
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb.encode("utf-8"), 500


app = create_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 8084)
