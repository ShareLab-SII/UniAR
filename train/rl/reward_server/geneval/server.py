"""GenEval reward server — compositional image evaluation.

Evaluates generated images for object presence, count, color, and spatial
relationships using Mask2Former detection + CLIP color classification.

Launch with gunicorn::

    GENEVAL_NUM_DEVICES=4 GPU_IDS="[0,1,2,3]" GENEVAL_PORT=8085 \
        gunicorn -c gunicorn.conf.py server:app
"""

import pickle
import traceback

import numpy as np
from flask import Flask, request, Blueprint
from PIL import Image
from io import BytesIO

from evaluator import load_geneval

root = Blueprint("root", __name__)


def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_geneval()
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "service": "geneval"}, 200


@root.route("/", methods=["POST"])
def inference():
    data = request.get_data()
    if not data:
        return {"error": "empty request body"}, 400
    try:
        data = pickle.loads(data)
        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
        meta_datas = data["meta_datas"]
        only_strict = data["only_strict"]

        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = (
            INFERENCE_FN(images, meta_datas, only_strict)
        )
        response = pickle.dumps({
            "scores": scores,
            "rewards": rewards,
            "strict_rewards": strict_rewards,
            "group_rewards": group_rewards,
            "group_strict_rewards": group_strict_rewards,
        })
        return response, 200
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb.encode("utf-8"), 500


app = create_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 8085)
