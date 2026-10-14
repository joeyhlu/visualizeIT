'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const CLIENT_PATH = path.join(__dirname, '..', 'prototype', 'client.html');
const HTML = fs.readFileSync(CLIENT_PATH, 'utf8');
const SCRIPT = HTML.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(SCRIPT, 'prototype client must contain its inline script');

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

function fakeCanvas() {
  const canvas = { width: 640, height: 480, drawCalls: [], clearCalls: [], strokes: 0 };
  const context = {
    clearRect(...args) { canvas.clearCalls.push(args); },
    drawImage(...args) { canvas.drawCalls.push(args); },
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() { canvas.strokes += 1; },
  };
  canvas.getContext = () => context;
  canvas.toDataURL = () => 'data:image/jpeg;base64,ZmFrZS1qcGVn';
  return canvas;
}

function makeStatus() {
  return {
    assets: [
      { objectId: 1, referenceId: 'ref-one', name: 'Object one', evaluationPose: { secret: true },
        evaluationMask: 'private-mask', depth: 'private-depth' },
      { objectId: 2, referenceId: 'ref-two', name: 'Object two' },
    ],
    calibrations: [
      { calibrationId: 'measured-32x24', kind: 'physical_measured', width: 32, height: 24 },
      { calibrationId: 'dataset-only', kind: 'dataset_reference', width: 32, height: 24 },
      { calibrationId: 'synthetic-only', kind: 'synthetic', width: 32, height: 24 },
    ],
  };
}

function makeHarness({ status = makeStatus(), secure = true, cameraError = null } = {}) {
  const ids = ['asset', 'calibration', 'camera', 'capture', 'continuous', 'stop', 'status', 'caption', 'video', 'result'];
  const elements = Object.fromEntries(ids.map(id => [id, {
    id, value: '', textContent: '', disabled: false, children: [], width: 640, height: 480,
    replaceChildren(...children) { this.children = children; },
  }]));
  const display = fakeCanvas();
  Object.assign(elements.result, display);
  elements.result.getContext = display.getContext;
  elements.result.drawCalls = display.drawCalls;
  elements.result.clearCalls = display.clearCalls;
  Object.defineProperty(elements.result, 'strokes', { get: () => display.strokes });

  const videoFrames = [];
  elements.video.videoWidth = 32;
  elements.video.videoHeight = 24;
  elements.video.play = () => Promise.resolve();
  elements.video.requestVideoFrameCallback = callback => { videoFrames.push(callback); return videoFrames.length; };
  const createdCanvases = [];
  const documentListeners = new Map();
  const windowListeners = new Map();
  const timers = new Map();
  const requests = [];
  const fetchCalls = [];
  const mediaCalls = [];
  let now = 1000;
  let uuid = 0;
  let timerId = 0;
  let stoppedTracks = 0;

  function response(body, ok = true) {
    return { ok, json: async () => body };
  }
  function fetch(url, options = {}) {
    fetchCalls.push({ url, options });
    if (url === '/v1/status') return Promise.resolve(response(status));
    const item = { url, options, packet: JSON.parse(options.body), ...deferred() };
    requests.push(item);
    return item.promise;
  }
  function setTimeoutFake(callback, delay = 0) {
    const id = ++timerId;
    timers.set(id, { callback, delay });
    return id;
  }
  function clearTimeoutFake(id) { timers.delete(id); }
  const video = elements.video;
  video.srcObject = null;
  const stream = { getTracks: () => [{ stop() { stoppedTracks += 1; } }] };
  const context = {
    document: {
      hidden: false,
      getElementById: id => elements[id],
      createElement: tag => {
        assert.equal(tag, 'canvas');
        const canvas = fakeCanvas();
        createdCanvases.push(canvas);
        return canvas;
      },
      addEventListener: (name, callback) => documentListeners.set(name, callback),
    },
    window: { isSecureContext: secure, addEventListener: (name, callback) => windowListeners.set(name, callback) },
    navigator: { mediaDevices: { getUserMedia: async constraints => {
      mediaCalls.push(constraints);
      if (cameraError) throw cameraError;
      return stream;
    } } },
    crypto: { randomUUID: () => `session-${++uuid}` },
    performance: { now: () => now },
    fetch,
    Option: function Option(text, value) { return { text, value }; },
    AbortController: class AbortController {
      constructor() { this.signal = { aborted: false }; }
      abort() { this.signal.aborted = true; }
    },
    setTimeout: setTimeoutFake,
    clearTimeout: clearTimeoutFake,
  };
  vm.runInNewContext(SCRIPT, context, { filename: CLIENT_PATH });

  return {
    elements, requests, fetchCalls, mediaCalls, videoFrames, createdCanvases,
    documentListeners, windowListeners, timers, stream,
    get stoppedTracks() { return stoppedTracks; },
    async ready() { await new Promise(resolve => setImmediate(resolve)); },
    now(value) { now = value; },
    present(frameId, mediaTime) {
      const callback = videoFrames.shift();
      assert.equal(typeof callback, 'function', 'video frame callback should be scheduled');
      callback(now, { presentedFrames: frameId, mediaTime });
    },
    runTimer(delay) {
      const entry = [...timers.entries()].find(([, item]) => item.delay === delay);
      if (!entry) return false;
      timers.delete(entry[0]);
      entry[1].callback();
      return true;
    },
  };
}

function trackedResult(packet, overrides = {}) {
  return {
    sessionId: packet.sessionId, frameId: packet.frameId, logicalIndex: packet.logicalIndex,
    objectId: packet.objectId, referenceId: packet.referenceId, calibrationId: packet.calibrationId,
    cameraFrameId: packet.cameraFrameId ?? null, cameraMediaTime: packet.cameraMediaTime ?? null,
    poseState: 'tracking', maskState: 'unavailable', renderState: 'diagnostic_wireframe',
    freshCameraFrameVerified: true, failureReason: null, processingSeconds: .01,
    neuralUpdatesPerSecond: 0, projectionSegments: [[1, 2, 3, 4]],
    ...overrides,
  };
}

async function startCamera(harness, { objectId = '1', calibrationId = 'measured-32x24' } = {}) {
  await harness.ready();
  harness.elements.asset.value = objectId;
  harness.elements.calibration.value = calibrationId;
  await harness.elements.camera.onclick();
}

async function settleRequest(request, value, ok = true) {
  request.resolve({ ok, json: async () => value });
  await new Promise(resolve => setImmediate(resolve));
}

test('status hides unmeasured calibration and camera errors are explicit', async () => {
  const h = makeHarness();
  await h.ready();
  assert.deepEqual(h.elements.calibration.children.map(option => option.value), ['', 'measured-32x24']);

  await startCamera(h, { calibrationId: '' });
  assert.equal(h.mediaCalls.length, 1);
  assert.equal('width' in h.mediaCalls[0].video, false);
  h.present(4, .12);
  const pending = h.elements.capture.onclick();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].packet.calibrationId, null);
  await settleRequest(h.requests[0], trackedResult(h.requests[0].packet, {
    poseState: 'lost', renderState: 'suppressed', freshCameraFrameVerified: false,
    failureReason: 'calibration_required', projectionSegments: null,
  }));
  await pending;
  assert.match(h.elements.status.textContent, /Reason: calibration_required/);
  assert.equal(h.elements.result.strokes, 0);

  const insecure = makeHarness({ secure: false });
  await insecure.ready();
  await insecure.elements.camera.onclick();
  assert.equal(insecure.mediaCalls.length, 0);
  assert.equal(insecure.elements.status.textContent, 'Camera access requires trusted HTTPS.');
});

test('every response identity mismatch suppresses the frame overlay', async () => {
  const mismatches = [
    ['sessionId', packet => `${packet.sessionId}-wrong`],
    ['frameId', packet => packet.frameId + 1],
    ['objectId', packet => packet.objectId + 1],
    ['referenceId', packet => `${packet.referenceId}-wrong`],
    ['calibrationId', () => 'another-calibration'],
  ];
  for (const [field, corrupt] of mismatches) {
    const h = makeHarness();
    await startCamera(h);
    h.present(10, .3);
    const operation = h.elements.capture.onclick();
    const request = h.requests[0];
    const result = trackedResult(request.packet, { [field]: corrupt(request.packet) });
    await settleRequest(request, result);
    await operation;
    assert.equal(h.elements.result.drawCalls.length, 0, `${field} mismatch must not draw a stale frame`);
    assert.ok(h.elements.result.clearCalls.length > 0, `${field} mismatch must clear prior pixels`);
    assert.match(h.elements.status.textContent, /Response identity mismatch/);
  }
});

test('a failed fetch clears the previously displayed snapshot and pose overlay', async () => {
  const h = makeHarness();
  await startCamera(h);
  h.present(1, .1);
  const first = h.elements.capture.onclick();
  await settleRequest(h.requests[0], trackedResult(h.requests[0].packet));
  await first;
  assert.equal(h.elements.result.drawCalls.length, 1);
  assert.equal(h.elements.result.strokes, 1);

  h.present(2, .2);
  const second = h.elements.capture.onclick();
  const clearsBeforeFailure = h.elements.result.clearCalls.length;
  h.requests[1].reject(new Error('offline')); // local fake; no network request is made
  await second;
  assert.ok(h.elements.result.clearCalls.length > clearsBeforeFailure);
  assert.equal(h.elements.result.drawCalls.length, 1, 'the failed request must not redraw the previous snapshot');
  assert.match(h.elements.caption.textContent, /pose overlay suppressed/);
  assert.match(h.elements.status.textContent, /offline/);
});

test('stop invalidates a pending result and stops the captured stream', async () => {
  const h = makeHarness();
  await startCamera(h);
  h.present(2, .2);
  const operation = h.elements.capture.onclick();
  const request = h.requests[0];
  h.elements.stop.onclick();
  const clearCount = h.elements.result.clearCalls.length;
  assert.equal(h.stoppedTracks, 1);
  await settleRequest(request, trackedResult(request.packet));
  await operation;
  assert.equal(h.elements.result.drawCalls.length, 0);
  assert.equal(h.elements.result.clearCalls.length, clearCount);
  assert.match(h.elements.caption.textContent, /Camera stopped/);
});

test('object changes invalidate an older in-flight response', async () => {
  const h = makeHarness();
  await startCamera(h);
  h.present(3, .3);
  const operation = h.elements.capture.onclick();
  const request = h.requests[0];
  h.elements.asset.value = '2';
  h.elements.asset.onchange();
  const clearCount = h.elements.result.clearCalls.length;
  await settleRequest(request, trackedResult(request.packet));
  await operation;
  assert.equal(h.elements.result.drawCalls.length, 0);
  assert.equal(h.elements.result.clearCalls.length, clearCount);
  assert.match(h.elements.caption.textContent, /Object changed/);
});

test('continuous capture sends each fresh camera frame once, drops overlap, and posts no evaluation data', async () => {
  const h = makeHarness();
  await startCamera(h);
  h.present(7, .7);
  h.elements.continuous.onclick();
  assert.equal(h.requests.length, 1);
  const overlapping = h.elements.capture.onclick();
  await overlapping;
  assert.equal(h.requests.length, 1, 'a second request must not overlap the in-flight frame');

  const allowed = [
    'cameraFrameId', 'cameraMediaTime', 'calibrationId', 'captureTimestampSeconds', 'frameId',
    'height', 'logicalIndex', 'objectId', 'referenceId', 'rgbJPEGBase64', 'sessionId',
    'timestampKind', 'width',
  ].sort();
  assert.deepEqual(Object.keys(h.requests[0].packet).sort(), allowed);
  assert.equal(h.requests[0].packet.cameraFrameId, 7);
  assert.equal(h.requests[0].packet.cameraMediaTime, .7);
  assert.equal(Object.keys(h.requests[0].packet).some(key => /evaluation|mask|depth|pose/i.test(key)), false);
  await settleRequest(h.requests[0], trackedResult(h.requests[0].packet));
  await new Promise(resolve => setImmediate(resolve));
  assert.match(h.elements.status.textContent, /Completed camera results: 1/);
  assert.match(h.elements.status.textContent, /Browser fresh valid poses: 1/);
  assert.match(h.elements.status.textContent, /Dropped capture requests: 1/);

  assert.equal(h.runTimer(0), true, 'continuous loop should schedule its next frame check');
  assert.equal(h.runTimer(10), true, 'same camera frame should be checked again without reposting');
  assert.equal(h.requests.length, 1, 'the same presented camera frame must not be sent twice');
  h.present(8, .8);
  assert.equal(h.runTimer(10), true);
  assert.equal(h.requests.length, 2, 'a new presented camera frame may be processed');
  await settleRequest(h.requests[1], trackedResult(h.requests[1].packet));
  await new Promise(resolve => setImmediate(resolve));
  assert.match(h.elements.status.textContent, /Completed camera results: 2/);
  assert.match(h.elements.status.textContent, /Browser fresh valid poses: 2/);
  h.elements.continuous.onclick();
});
