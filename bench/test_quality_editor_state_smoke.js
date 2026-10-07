const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// A small deterministic frozen-selection fixture is embedded here so the
// smoke test reads the real Python HTML template without generated editor data.
const frameRows = Array.from({length: 40}, (_, index) => {
  const frameId = 100 + index;
  const group = index % 5 === 1 ? 'hard_candidate' : 'uniform';
  return [frameId, group, `${frameId}.jpg`];
});
const frozen = {
  keyboard: {
    frames: frameRows,
    resolution: [1024, 1280],
    source_sha256: 'a'.repeat(64),
    pending_selection_status: 'hard candidates need image-only difficulty review',
    reviewed_selection_status: 'reviewed_image_only',
  },
};

const annotationSource = fs.readFileSync(path.join(__dirname, 'quality_annotations.py'), 'utf8');
const templateMatch = annotationSource.match(/^HTML = r'''([\s\S]*?)'''/m);
assert.ok(templateMatch, 'real annotation module embeds the editor HTML');
const html = templateMatch[1].replace('__FROZEN_CASES__', JSON.stringify(frozen));
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(scriptMatch, 'real editor template contains its application script');
const source = scriptMatch[1];

const alias = 'keyboard';
const fixture = frozen[alias];
const sourceDocument = {
  schema_version: 1,
  object: alias,
  resolution: fixture.resolution,
  selection_status: fixture.pending_selection_status,
  operator: null,
  source_sha256: fixture.source_sha256,
  frames: fixture.frames.map(([frame_id, group, image]) => ({
    frame_id, group, image, status: 'pending', visibility: null,
    visible_object: [], overlapping_hands: [], landmarks: [], review_notes: '',
  })),
};
const preservedSourceLabel = {
  status: 'pending',
  visibility: 'visible',
  visible_object: [{points: [[40, 40], [80, 40], [80, 90], [40, 90]], hole: false}],
  overlapping_hands: [],
  landmarks: [{
    landmark_id: 'preserved-center', object_point_m: [0, 0, 0],
    pixel: [60, 65], visible: true, reviewed: true,
  }],
  review_notes: 'Preserve this legacy v1 label during v2 export.',
};
Object.assign(sourceDocument.frames[1], preservedSourceLabel);

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      id, value: '', textContent: '', checked: false, files: [], children: [],
      replaceChildren(...children) { this.children = children; },
      click() {},
    });
  }
  return elements.get(id);
}
for (const id of ['object', 'frame', 'status', 'visibility', 'tool', 'hole', 'canvas',
  'landmarkId', 'xyz', 'correspondence', 'landmarkReason', 'operator', 'hardReview',
  'previous', 'next', 'close', 'undo', 'clear', 'reviewMask', 'reviewLandmarks',
  'markUnobservable', 'download', 'reload', 'file']) element(id);
element('object').value = alias;
element('frame').value = '0';
element('tool').value = 'visible_object';
element('canvas').width = fixture.resolution[0];
element('canvas').height = fixture.resolution[1];

const drawing = new Proxy({}, {
  get(target, key) { return key in target ? target[key] : () => {}; },
  set(target, key, value) { target[key] = value; return true; },
});
element('canvas').getContext = () => drawing;
element('canvas').getBoundingClientRect = () => ({
  left: 0, top: 0, width: element('canvas').width, height: element('canvas').height,
});
class FakeImage { async decode() {} }
const exported = {text: null};
class CapturedBlob {
  constructor(parts) { exported.text = parts.join(''); }
}
const context = {
  document: {
    getElementById: element,
    createElement: () => ({click() {}}),
  },
  Image: FakeImage,
  Blob: CapturedBlob,
  URL: {createObjectURL: () => 'blob:fixture', revokeObjectURL() {}},
  fetch: async () => ({ok: true, json: async () => sourceDocument}),
  setTimeout: () => 0,
  console,
  process,
};
vm.createContext(context);
vm.runInContext(source + '\nglobalThis.__e4 = {getData: () => data, validateDocument};', context);

function clickCanvas(x, y) {
  element('canvas').onclick({clientX: x, clientY: y});
}
function drawPolygon(tool, points) {
  element('tool').value = tool;
  for (const [x, y] of points) clickCanvas(x, y);
  element('close').onclick();
}

async function main() {
  await new Promise(resolve => setImmediate(resolve));
  const data = context.__e4.getData();
  assert.equal(sourceDocument.schema_version, 1, 'legacy source remains unchanged');
  assert.equal(data.schema_version, 2, 'legacy document is normalized in editor memory');
  assert.equal(data.frames[0].landmark_status, 'pending');
  assert.equal(data.frames[1].landmark_status, 'pending');
  assert.deepEqual(JSON.parse(JSON.stringify(data.frames[1].landmarks)), preservedSourceLabel.landmarks);

  const first = data.frames[0];
  first.visibility = 'visible';
  first.landmark_status = 'unobservable';
  first.landmark_review_reason = 'No independently measurable point';
  assert.throws(() => context.__e4.validateDocument(data, alias), /Landmark review needs a reviewer name/);
  first.visibility = null;
  first.landmark_status = 'pending';
  delete first.landmark_review_reason;

  element('visibility').value = 'visible';
  element('visibility').onchange();
  element('tool').value = 'landmark';
  element('landmarkId').value = 'pending-without-reviewer';
  element('xyz').value = '0,0,0';
  element('correspondence').checked = true;
  clickCanvas(16, 12);
  element('reviewLandmarks').onclick();
  assert.equal(first.landmark_status, 'pending', 'landmark review requires a reviewer before state mutation');
  assert.equal(first.landmarks.length, 1, 'failed review keeps the pending point');
  assert.match(element('status').textContent, /reviewer name/);
  element('landmarkReason').value = 'No independently measurable point';
  element('markUnobservable').onclick();
  assert.equal(first.landmark_status, 'pending', 'unobservable status also requires a reviewer');
  assert.equal(first.landmarks.length, 1, 'failed unobservable review does not erase the point');
  element('clear').onclick();

  element('visibility').value = 'visible';
  element('visibility').onchange();
  element('operator').value = 'Fixture reviewer';
  element('hardReview').checked = true;
  drawPolygon('visible_object', [[3, 3], [8, 3], [8, 9], [3, 9]]);
  element('reviewMask').onclick();
  assert.equal(first.status, 'reviewed');
  assert.equal(first.landmark_status, 'pending');

  element('tool').value = 'landmark';
  element('landmarkId').value = 'center';
  element('xyz').value = '0,0,0';
  element('correspondence').checked = true;
  clickCanvas(16, 12);
  assert.equal(first.status, 'reviewed', 'adding a point leaves mask review intact');
  assert.equal(first.landmark_status, 'pending', 'point edits invalidate landmark review');
  element('reviewLandmarks').onclick();
  assert.equal(first.landmark_status, 'reviewed');

  element('landmarkReason').value = 'No independently measurable surface point';
  element('markUnobservable').onclick();
  assert.equal(first.status, 'reviewed', 'unobservable choice leaves mask review intact');
  assert.equal(first.landmark_status, 'unobservable');
  assert.equal(first.landmark_review_reason, element('landmarkReason').value);
  assert.equal(first.landmarks.length, 0);

  drawPolygon('overlapping_hands', [[12, 3], [15, 3], [15, 9], [12, 9]]);
  assert.equal(first.status, 'pending', 'mask geometry invalidates only mask review');
  assert.equal(first.landmark_status, 'unobservable', 'mask edits preserve landmark review state');

  element('tool').value = 'landmark';
  element('landmarkId').value = 'temporary-visible-point';
  element('xyz').value = '0,0,0';
  element('correspondence').checked = true;
  clickCanvas(18, 12);
  const targetCountBeforeHidden = first.visible_object.length;
  const landmarkCountBeforeHidden = first.landmarks.length;

  element('visibility').value = 'fully_hidden';
  element('visibility').onchange();
  assert.equal(first.status, 'pending');
  assert.equal(element('visibility').value, 'visible', 'contradictory fully hidden edit is rejected');
  assert.equal(first.visibility, 'visible');
  assert.equal(first.landmark_status, 'pending');
  assert.equal(first.visible_object.length, targetCountBeforeHidden);
  assert.equal(first.landmarks.length, landmarkCountBeforeHidden);
  element('visibility').value = 'fully_hidden';
  element('reviewMask').onclick();
  assert.equal(first.visibility, 'visible', 'mask review rejects contradiction before changing labels');
  assert.match(element('status').textContent, /Clear target polygons and landmarks/);
  assert.equal(first.visible_object.length, targetCountBeforeHidden);
  assert.equal(first.landmarks.length, landmarkCountBeforeHidden);

  element('clear').onclick();
  assert.equal(first.visible_object.length, 0);
  assert.equal(first.landmarks.length, 0);
  element('visibility').value = 'fully_hidden';
  element('visibility').onchange();
  element('reviewMask').onclick();
  assert.equal(first.status, 'reviewed');
  assert.equal(first.landmark_status, 'not_applicable_fully_hidden');

  element('download').onclick();
  assert.ok(exported.text, 'the real editor download handler emitted an in-memory JSON export');
  const exportedDocument = JSON.parse(exported.text);
  assert.equal(exportedDocument.schema_version, 2, 'legacy input exports as v2');
  assert.equal(exportedDocument.frames[0].status, 'reviewed');
  assert.equal(exportedDocument.frames[0].landmark_status, 'not_applicable_fully_hidden');
  assert.equal(exportedDocument.frames[1].landmark_status, 'pending',
    'v1 pending review state is preserved even when editable labels are present');
  assert.deepEqual(exportedDocument.frames[1].visible_object, preservedSourceLabel.visible_object);
  assert.deepEqual(exportedDocument.frames[1].landmarks, preservedSourceLabel.landmarks);
  assert.equal(exportedDocument.frames[1].review_notes, preservedSourceLabel.review_notes);
  assert.equal(sourceDocument.schema_version, 1, 'editor transitions and export do not mutate v1 input');
  console.log('E4 editor state smoke passed (real HTML source, v1-to-v2 labels preserved, independent review transitions).');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
