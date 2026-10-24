/* Experimental, bounded CPU patch tracking. Thresholds are development gates,
 * not measured tracking accuracy. No image or feature history is accumulated. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PhoneTracker = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const R = 3, AREA = 49, MAX_FEATURES = 192;
  const squared = (x, y) => x * x + y * y;
  const typed = value => ArrayBuffer.isView(value) && !(value instanceof DataView);
  const dimensions = (w, h) => Number.isInteger(w) && Number.isInteger(h) && w > 0 && h > 0;
  function median(values) {
    const sorted = values.slice().sort((a, b) => a - b), middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  function grayFrame(input) {
    if (!input || !dimensions(input.width, input.height) || Math.max(input.width, input.height) > 192 ||
        !typed(input.rgba) || input.rgba.length !== input.width * input.height * 4 ||
        !Number.isFinite(input.timestamp)) return null;
    const gray = new Uint8Array(input.width * input.height);
    for (let i = 0; i < gray.length; i++) {
      const j = i * 4;
      if (!Number.isFinite(input.rgba[j]) || !Number.isFinite(input.rgba[j + 1]) ||
          !Number.isFinite(input.rgba[j + 2]) || !Number.isFinite(input.rgba[j + 3])) return null;
      gray[i] = (77 * input.rgba[j] + 150 * input.rgba[j + 1] + 29 * input.rgba[j + 2]) >> 8;
    }
    return gray;
  }
  function patch(gray, width, x, y) {
    const result = new Uint8Array(AREA);
    let k = 0;
    for (let dy = -R; dy <= R; dy++) {
      for (let dx = -R; dx <= R; dx++) result[k++] = gray[(y + dy) * width + x + dx];
    }
    return result;
  }
  function featureScore(gray, width, x, y) {
    let sum = 0, sum2 = 0, xx = 0, yy = 0, xy = 0;
    for (let dy = -R; dy <= R; dy++) {
      for (let dx = -R; dx <= R; dx++) {
        const i = (y + dy) * width + x + dx, v = gray[i];
        const gx = (gray[i + 1] - gray[i - 1]) / 2;
        const gy = (gray[i + width] - gray[i - width]) / 2;
        sum += v; sum2 += v * v; xx += gx * gx; yy += gy * gy; xy += gx * gy;
      }
    }
    if (sum2 / AREA - squared(sum / AREA, 0) < 100) return -1;
    xx /= AREA; yy /= AREA; xy /= AREA;
    const eigen = (xx + yy - Math.sqrt(squared(xx - yy, 2 * xy))) / 2;
    return eigen >= 25 ? eigen : -1;
  }
  function candidates(gray, width, height, diagnostics) {
    const scored = [];
    for (let y = 10; y < height - 10; y += 5) {
      for (let x = 10; x < width - 10; x += 5) {
        const score = featureScore(gray, width, x, y);
        if (score >= 0) scored.push({ x0: x, y0: y, x, y, score, alive: true, masked: false });
      }
    }
    scored.sort((a, b) => b.score - a.score || a.y0 - b.y0 || a.x0 - b.x0);
    const selected = [], chosen = new Set(), quotas = new Uint8Array(48);
    const cell = candidate => Math.min(5, Math.floor(6 * candidate.y0 / height)) * 8 +
      Math.min(7, Math.floor(8 * candidate.x0 / width));
    const accept = candidate => {
      if (chosen.has(candidate) || !selected.every(other =>
        squared(candidate.x0 - other.x0, candidate.y0 - other.y0) >= 25)) return false;
      candidate.reference = patch(gray, width, candidate.x0, candidate.y0);
      selected.push(candidate); chosen.add(candidate); return true;
    };
    // Reserve coverage before stronger background texture consumes the cap.
    // Every selected feature still passes the original quality/spacing gates.
    for (const candidate of scored) {
      const index = cell(candidate);
      if (quotas[index] < 4 && accept(candidate)) quotas[index]++;
      if (selected.length === MAX_FEATURES) break;
    }
    const quotaCount = selected.length;
    // Sparse cells return their unused slots to the same deterministic ranking.
    for (const candidate of scored) {
      if (selected.length === MAX_FEATURES) break;
      accept(candidate);
    }
    if (diagnostics) {
      diagnostics.eligibleCandidates = scored.length;
      diagnostics.quotaSelectedFeatures = quotaCount;
      diagnostics.backfillFeatures = selected.length - quotaCount;
      diagnostics.initialOccupiedCells = new Set(selected.map(cell)).size;
    }
    return selected;
  }
  // Mean-subtracted SSD is computed without allocating a patch per candidate.
  // The second candidate must be at least three pixels from the best position.
  function match(template, gray, width, height, cx, cy, radius) {
    const scores = [];
    let best = null;
    for (let y = cy - radius; y <= cy + radius; y++) {
      if (y < R || y >= height - R) continue;
      for (let x = cx - radius; x <= cx + radius; x++) {
        if (x < R || x >= width - R) continue;
        let sum = 0, sum2 = 0, k = 0;
        for (let dy = -R; dy <= R; dy++) {
          for (let dx = -R; dx <= R; dx++) {
            const difference = template[k++] - gray[(y + dy) * width + x + dx];
            sum += difference; sum2 += difference * difference;
          }
        }
        const score = Math.max(0, (sum2 - sum * sum / AREA) / AREA);
        const candidate = { x, y, score };
        scores.push(candidate);
        if (!best || score < best.score) best = candidate;
      }
    }
    if (!best || best.score > 225) return null;
    let second = Infinity;
    for (const candidate of scores) {
      if (squared(candidate.x - best.x, candidate.y - best.y) >= 9) second = Math.min(second, candidate.score);
    }
    if (!Number.isFinite(second) || best.score > .75 * second || second - best.score < 25) return null;
    return best;
  }
  class Tracker {
    constructor() { this.reset(); }
    reset() {
      this.width = 0; this.height = 0; this.timestamp = null;
      this.epoch = null; this.frameId = null;
      this.referenceGray = null; this.previousGray = null; this.currentGray = null;
      this.features = []; this.maskBounds = null; this.hasMask = false;
      this.lost = false; this.valid = false; this.reason = 'not-started';
      this.dx = null; this.dy = null;
      this.diagnostics = { initialFeatures: 0, survivingFeatures: 0, maskedInitialFeatures: 0,
        inliers: 0, occupiedQuadrants: 0, medianResidual: null, eligibleCandidates: 0,
        quotaSelectedFeatures: 0, backfillFeatures: 0, initialOccupiedCells: 0 };
      return this.result();
    }
    fail(reason) {
      this.lost = true; this.valid = false; this.reason = reason;
      this.dx = null; this.dy = null;
      return this.result();
    }
    start(input) {
      this.reset();
      const gray = grayFrame(input);
      if (!gray) return this.fail('invalid-input');
      this.width = input.width; this.height = input.height; this.timestamp = input.timestamp;
      this.epoch = input.epoch; this.frameId = input.frameId;
      this.referenceGray = gray;
      this.currentGray = gray.slice();
      this.previousGray = this.currentGray;
      this.features = candidates(gray, this.width, this.height, this.diagnostics);
      this.diagnostics.initialFeatures = this.features.length;
      this.diagnostics.survivingFeatures = this.features.length;
      if (this.features.length < 8) return this.fail('insufficient-features');
      this.reason = 'awaiting-mask';
      return this.result();
    }
    update(input) {
      if (this.lost) return this.result();
      if (!this.referenceGray) return this.fail('not-started');
      if (!input || input.width !== this.width || input.height !== this.height) return this.fail('dimension-change');
      if (!Number.isFinite(input.timestamp)) return this.fail('invalid-input');
      if (input.timestamp <= this.timestamp) return this.fail('nonincreasing-time');
      if (input.timestamp - this.timestamp > 750) return this.fail('frame-gap');
      const gray = grayFrame(input);
      if (!gray) return this.fail('invalid-input');
      this.previousGray = this.currentGray;
      this.currentGray = gray;
      this.timestamp = input.timestamp;
      // Every feature is tracked independently, including before a delayed mask
      // arrives. Background consensus never chooses the object's translation.
      for (const feature of this.features) {
        if (!feature.alive) continue;
        const previous = patch(this.previousGray, this.width, feature.x, feature.y);
        const forward = match(previous, gray, this.width, this.height, feature.x, feature.y, 6);
        if (!forward) { feature.alive = false; continue; }
        const backward = match(patch(gray, this.width, forward.x, forward.y), this.previousGray,
          this.width, this.height, forward.x, forward.y, 6);
        if (!backward || squared(backward.x - feature.x, backward.y - feature.y) > 2.25) {
          feature.alive = false; continue;
        }
        const anchored = match(feature.reference, gray, this.width, this.height, forward.x, forward.y, 4);
        if (!anchored || squared(anchored.x - forward.x, anchored.y - forward.y) > 4) {
          feature.alive = false; continue;
        }
        feature.x = anchored.x; feature.y = anchored.y;
      }
      this.diagnostics.survivingFeatures = this.features.filter(feature => feature.alive).length;
      if (this.diagnostics.survivingFeatures < 8) return this.fail('insufficient-survivors');
      if (this.hasMask) return this.evaluate();
      this.valid = false; this.reason = 'awaiting-mask';
      return this.result();
    }
    setMask(input) {
      // A delayed mask belongs solely to the immutable tap frame. Reject stale
      // identities before inspecting its content, without changing any state.
      if (!input || !this.referenceGray || input.epoch !== this.epoch || input.frameId !== this.frameId || this.lost) return false;
      const { mask, width, height } = input;
      if (!dimensions(width, height) || !typed(mask) || mask.length !== width * height) {
        this.fail('invalid-mask'); return false;
      }
      let left = width, top = height, right = -1, bottom = -1;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const value = mask[y * width + x];
          if (!Number.isFinite(value)) { this.fail('invalid-mask'); return false; }
          if (value > 0) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); }
        }
      }
      this.hasMask = true;
      if (right < left) { this.fail('empty-mask'); return true; }
      this.maskBounds = { left, top, right, bottom, width, height };
      let initial = 0;
      for (const feature of this.features) {
        let inside = true;
        for (let dy = -R; dy <= R && inside; dy++) {
          for (let dx = -R; dx <= R; dx++) {
            // Pixel-center nearest-neighbor mapping preserves the whole 7x7
            // reference footprint, rather than checking only its center.
            const mx = Math.min(width - 1, Math.floor((feature.x0 + dx + .5) * width / this.width));
            const my = Math.min(height - 1, Math.floor((feature.y0 + dy + .5) * height / this.height));
            if (mask[my * width + mx] < 128) { inside = false; break; }
          }
        }
        feature.masked = inside;
        if (inside) initial++;
      }
      // Includes original features that died before the delayed mask arrived.
      this.diagnostics.maskedInitialFeatures = initial;
      this.evaluate();
      return true;
    }
    evaluate() {
      const selected = this.features.filter(feature => feature.masked && feature.alive);
      this.diagnostics.inliers = 0; this.diagnostics.occupiedQuadrants = 0;
      this.diagnostics.medianResidual = null;
      if (!selected.length) return this.fail('insufficient-mask-support');
      let dx = median(selected.map(feature => feature.x - feature.x0));
      let dy = median(selected.map(feature => feature.y - feature.y0));
      const inliers = selected.filter(feature => squared(feature.x - feature.x0 - dx, feature.y - feature.y0 - dy) <= 2.25);
      if (!inliers.length) return this.fail('insufficient-inliers');
      dx = median(inliers.map(feature => feature.x - feature.x0));
      dy = median(inliers.map(feature => feature.y - feature.y0));
      const bounds = this.maskBounds, quadrants = [0, 0, 0, 0];
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const feature of inliers) {
        const x = (feature.x0 + .5) * bounds.width / this.width - .5;
        const y = (feature.y0 + .5) * bounds.height / this.height - .5;
        quadrants[(x >= (bounds.left + bounds.right) / 2 ? 1 : 0) +
          (y >= (bounds.top + bounds.bottom) / 2 ? 2 : 0)]++;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      this.diagnostics.inliers = inliers.length;
      this.diagnostics.occupiedQuadrants = quadrants.filter(count => count >= 2).length;
      this.diagnostics.medianResidual = median(inliers.map(feature =>
        Math.sqrt(squared(feature.x - feature.x0 - dx, feature.y - feature.y0 - dy))));
      if (inliers.length < Math.max(8, Math.ceil(.5 * this.diagnostics.maskedInitialFeatures))) return this.fail('insufficient-inliers');
      if (this.diagnostics.occupiedQuadrants < 3 || maxX - minX < .25 * (bounds.right - bounds.left + 1) ||
          maxY - minY < .25 * (bounds.bottom - bounds.top + 1)) return this.fail('insufficient-spatial-support');
      this.valid = true; this.reason = 'tracking'; this.dx = dx; this.dy = dy;
      return this.result();
    }
    result() {
      return { valid: this.valid, dx: this.dx, dy: this.dy, timestamp: this.timestamp,
        reason: this.reason, diagnostics: { ...this.diagnostics } };
    }
  }
  return { Tracker };
});
