"""OCR reward server — word-recall scoring via PaddleOCR.

Scores generated images by comparing OCR-recognized text against quoted target
text from prompts using edit-distance reward.

Launch with gunicorn::

    NUM_DEVICES=1 PORT=8085 OCR_MODEL_DIR=/path/to/paddle_models \
        gunicorn -c gunicorn.conf.py server:app
"""

import pickle
import traceback

from flask import Flask, request, Blueprint
from PIL import Image
from io import BytesIO

from evaluator import load_ocr

root = Blueprint("root", __name__)


def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_ocr()
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "service": "ocr"}, 200


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
    app.run("0.0.0.0", 8085)
