"""Keep FAISS's OpenMP runtime out of the PyTorch inference process.

The worker imports NumPy/FAISS only. It receives fixed feature arrays over a
private subprocess pipe; no images, evaluation annotations or GPU models.
FAISS distances, clustering parameters and upstream FoundPose logic are retained.
"""
import atexit
import os
import pickle
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np


def send(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack('<Q', len(payload))); stream.write(payload); stream.flush()


def receive(stream):
    header = stream.read(8)
    if len(header) != 8: raise RuntimeError('FAISS worker exited; inspect its stderr log')
    count = struct.unpack('<Q', header)[0]
    if count > 1024**3: raise ValueError('Oversized feature message')
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count-len(chunks))
        if not chunk: raise RuntimeError('Incomplete FAISS feature message')
        chunks.extend(chunk)
    return pickle.loads(chunks)  # Trusted private child-process pipe only.


def worker_loop():
    import faiss
    if 'torch' in sys.modules: raise RuntimeError('FAISS worker must not import Torch')
    faiss.omp_set_num_threads(1)
    indices = {}
    while True:
        command = receive(sys.stdin.buffer)
        if command[0] == 'stop': break
        try:
            operation = command[0]
            if operation == 'fit':
                _, index_id, data, metric = command
                data = np.ascontiguousarray(data, dtype=np.float32)
                index = faiss.IndexFlatL2(data.shape[1]) if metric == 'l2' else faiss.IndexFlatIP(data.shape[1])
                index.add(data); indices[index_id] = index
                result = True
            elif operation == 'search':
                _, index_id, data, k = command
                result = indices[index_id].search(np.ascontiguousarray(data, dtype=np.float32), k)
            elif operation == 'delete':
                indices.pop(command[1], None); result = True
            elif operation == 'cluster':
                _, data, count, iterations = command
                data = np.ascontiguousarray(data, dtype=np.float32)
                model = faiss.Kmeans(data.shape[1], count, niter=iterations, gpu=False,
                                     verbose=False, seed=0, spherical=False)
                model.train(data); distances, ids = model.index.search(data, 1)
                result = (model.centroids, ids[:, 0].astype(np.int32), distances[:, 0])
            else: raise ValueError('Unknown feature operation')
            send(sys.stdout.buffer, ('ok', result))
        except Exception as error:
            send(sys.stdout.buffer, ('error', repr(error)))


class Client:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        logs = root/'.cache/model-quality/runtime'; logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs/f'faiss-worker-{os.getpid()}.log').open('wb')
        self.process = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'bench.quality_neighbors', '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=root)
        self.next_id = 0
        atexit.register(self.close)

    def request(self, command):
        send(self.process.stdin, command); status, result = receive(self.process.stdout)
        if status != 'ok': raise RuntimeError('FAISS worker: '+result)
        return result

    def close(self):
        if self.process.poll() is None:
            try: send(self.process.stdin, ('stop',)); self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired): self.process.kill(); self.process.wait(timeout=5)
        self.log.close()


_client = None


def client():
    global _client
    if _client is None: _client = Client()
    return _client


class IsolatedKNN:
    def __init__(self, k=1, metric='l2', radius=None, res=None):
        if radius is not None or metric not in ('l2', 'cosine'): raise ValueError('Unsupported FoundPose neighbor mode')
        self.k = k; self.metric = metric; self.index_id = None

    def fit(self, data):
        import torch
        values = data.detach().cpu().float()
        if self.metric == 'cosine': values = values/torch.linalg.norm(values, dim=1, keepdim=True)
        worker = client(); worker.next_id += 1; self.index_id = worker.next_id
        worker.request(('fit', self.index_id, values.numpy(), self.metric))

    def search(self, data):
        import torch
        values = data.detach().cpu().float()
        if self.metric == 'cosine': values = values/torch.linalg.norm(values, dim=1, keepdim=True)
        distances, ids = client().request(('search', self.index_id, values.numpy(), self.k))
        if self.metric == 'cosine': distances = 1.-distances
        return torch.from_numpy(distances).to(data.device), torch.from_numpy(ids).to(data.device)

    def __del__(self):
        if self.index_id is not None and _client is not None and _client.process.poll() is None:
            try: _client.request(('delete', self.index_id))
            except (RuntimeError, OSError, AttributeError): pass


def isolated_kmeans(samples, num_centroids, num_iter=50, verbose=False):
    import torch
    result = client().request(('cluster', samples.detach().cpu().numpy(), num_centroids, num_iter))
    return tuple(torch.from_numpy(value).to(samples.device) for value in result)


def install():
    from utils import knn_util, cluster_util
    knn_util.KNN = IsolatedKNN
    cluster_util.kmeans = isolated_kmeans


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']: raise SystemExit('Internal feature worker only')
    worker_loop()
