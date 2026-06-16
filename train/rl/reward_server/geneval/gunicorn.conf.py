import os

NUM_DEVICES = int(os.environ.get("NUM_DEVICES", "1"))
GPU_IDS = eval(os.environ["GPU_IDS"]) if "GPU_IDS" in os.environ else list(range(NUM_DEVICES))
USED_DEVICES = set()
port = int(os.environ.get("PORT", "8085"))


def pre_fork(server, worker):
    global USED_DEVICES
    worker.device_id = next(gid for gid in GPU_IDS if gid not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)


def child_exit(server, worker):
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)


bind = f"0.0.0.0:{port}"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 3000
