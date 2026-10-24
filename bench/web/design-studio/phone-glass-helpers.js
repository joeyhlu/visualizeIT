/*
 * VisualizeIT glass camera helpers (staged; NOT a replacement for phone-camera.js).
 *
 * Self-contained, dependency-free pieces of the approved glass camera spec that do
 * not depend on the existing PhoneCamera / PhoneSegmenterClient / tracker / studio
 * internals. phone-camera.js must call these at the integration points listed in
 * glass-implementation-notes.md. Nothing here touches segmentation, tracking math,
 * model runtimes, network, or browser storage.
 *
 * Exposed as globalThis.VisualizeITGlass (classic script) and module.exports (Node tests).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.VisualizeITGlass = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const IMPORT_LIMITS = Object.freeze({
    maxBytes: 12 * 1024 * 1024,
    maxPixels: 24000000,
    maxEdge: 1024,
    headerBytes: 512 * 1024,
  });
  const ENTRANCE_MS = 900;
  const REDUCED_MS = 160;

  class ImportError extends Error {
    constructor(code, message) {
      super(message);
      this.name = 'ImportError';
      this.code = code;
    }
  }

  // ---------------------------------------------------------------- headers
  const u16be = (b, i) => (b[i] << 8) | b[i + 1];
  const u32be = (b, i) => ((b[i] << 24) >>> 0) + (b[i + 1] << 16) + (b[i + 2] << 8) + b[i + 3];
  const u16le = (b, i) => b[i] | (b[i + 1] << 8);
  const u24le = (b, i) => b[i] | (b[i + 1] << 8) | (b[i + 2] << 16);
  const u32le = (b, i) => (b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)) + ((b[i + 3] << 24) >>> 0);
  const ascii = (b, i, n) => String.fromCharCode.apply(null, Array.from(b.subarray(i, i + n)));

  function parsePng(b) {
    const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (b.length < 24 || !sig.every((v, i) => b[i] === v)) return null;
    if (u32be(b, 8) !== 13 || ascii(b, 12, 4) !== 'IHDR') throw new ImportError('corrupt', 'PNG header is malformed.');
    return { type: 'image/png', width: u32be(b, 16), height: u32be(b, 20) };
  }

  function parseJpeg(b) {
    if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
    // Scan every metadata segment up to SOS/EOI: an APP1 EXIF orientation may legally
    // appear after the SOF segment, so SOF dimensions are retained, not returned early.
    let i = 2;
    let orientation = 1;
    let size = null;
    while (i + 3 < b.length) {
      if (b[i] !== 0xff) {
        if (size) break; // trailing bytes after a complete SOF are outside the scanned metadata
        throw new ImportError('corrupt', 'JPEG marker stream is malformed.');
      }
      while (b[i] === 0xff && i < b.length) i += 1; // fill bytes
      const marker = b[i];
      i += 1;
      if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (marker === 0xd9 || marker === 0xda) break; // EOI / SOS ends metadata
      if (i + 1 >= b.length) break;
      const length = u16be(b, i);
      if (length < 2) throw new ImportError('corrupt', 'JPEG segment length is invalid.');
      if (marker === 0xe1) orientation = exifOrientation(b, i + 2, Math.min(b.length, i + length)) || orientation;
      const sof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
      if (sof && !size && i + 7 <= b.length) size = { width: u16be(b, i + 5), height: u16be(b, i + 3) };
      i += length;
    }
    if (!size) throw new ImportError('corrupt', 'JPEG dimensions were not found in the readable header.');
    return { type: 'image/jpeg', width: size.width, height: size.height, orientation };
  }

  /** Reads EXIF orientation (tag 0x0112) from an APP1 payload; returns 1..8 or 0. Bounded, never throws. */
  function exifOrientation(b, start, end) {
    try {
      if (end - start < 14 || ascii(b, start, 4) !== 'Exif' || b[start + 4] !== 0 || b[start + 5] !== 0) return 0;
      const tiff = start + 6;
      const little = b[tiff] === 0x49 && b[tiff + 1] === 0x49;
      if (!little && !(b[tiff] === 0x4d && b[tiff + 1] === 0x4d)) return 0;
      const r16 = (k) => (little ? u16le(b, k) : u16be(b, k));
      const r32 = (k) => (little ? u32le(b, k) : u32be(b, k));
      if (r16(tiff + 2) !== 42) return 0;
      const ifd = tiff + r32(tiff + 4);
      if (ifd + 2 > end) return 0;
      const count = Math.min(r16(ifd), 256);
      for (let n = 0; n < count; n += 1) {
        const entry = ifd + 2 + n * 12;
        if (entry + 12 > end) return 0;
        if (r16(entry) === 0x0112) {
          const value = r16(entry + 8);
          return value >= 1 && value <= 8 ? value : 0;
        }
      }
    } catch (error) {
      return 0;
    }
    return 0;
  }

  function parseWebp(b, fileSize) {
    if (b.length < 30 || ascii(b, 0, 4) !== 'RIFF' || ascii(b, 8, 4) !== 'WEBP') return null;
    const riffSize = u32le(b, 4) + 8;
    if (riffSize > fileSize + 1 || riffSize < 30) throw new ImportError('corrupt', 'WebP container size is inconsistent.');
    const chunk = ascii(b, 12, 4);
    if (chunk === 'VP8 ') {
      if (b[23] !== 0x9d || b[24] !== 0x01 || b[25] !== 0x2a) throw new ImportError('corrupt', 'WebP VP8 frame is malformed.');
      return { type: 'image/webp', width: u16le(b, 26) & 0x3fff, height: u16le(b, 28) & 0x3fff };
    }
    if (chunk === 'VP8L') {
      if (b[20] !== 0x2f) throw new ImportError('corrupt', 'WebP lossless header is malformed.');
      const w = 1 + (b[21] | ((b[22] & 0x3f) << 8));
      const h = 1 + ((b[22] >> 6) | (b[23] << 2) | ((b[24] & 0x0f) << 10));
      return { type: 'image/webp', width: w, height: h };
    }
    if (chunk === 'VP8X') {
      return { type: 'image/webp', width: 1 + u24le(b, 24), height: 1 + u24le(b, 27) };
    }
    throw new ImportError('unsupported', 'Unsupported WebP variant.');
  }

  /** Sniff PNG/JPEG/WebP from bytes only. Throws ImportError; never trusts file.type. */
  function probeImageHeader(bytes, fileSize = bytes.length) {
    const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const header = parsePng(b) || parseJpeg(b) || parseWebp(b, fileSize);
    if (!header) throw new ImportError('unsupported', 'Only PNG, JPEG or WebP images can be imported.');
    const { width, height } = header;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
      throw new ImportError('corrupt', 'Image dimensions are missing or invalid.');
    }
    if (width * height > IMPORT_LIMITS.maxPixels) {
      throw new ImportError('too-large', 'Image exceeds 24 megapixels.');
    }
    return header;
  }

  async function readHeaderBytes(file, limit) {
    const slice = file.slice(0, Math.min(file.size, limit));
    if (typeof slice.arrayBuffer === 'function') return new Uint8Array(await slice.arrayBuffer());
    return new Uint8Array(await new Response(slice).arrayBuffer());
  }

  /** Validate size, real content signature and dimensions before any full decode. */
  async function validateImageFile(file) {
    if (!file || typeof file.size !== 'number' || typeof file.slice !== 'function') {
      throw new ImportError('unsupported', 'No image file was provided.');
    }
    if (/svg/i.test(file.type || '') || /\.svg$/i.test(file.name || '')) {
      throw new ImportError('unsupported', 'SVG images cannot be imported.');
    }
    if (file.size <= 0) throw new ImportError('corrupt', 'The image file is empty.');
    if (file.size > IMPORT_LIMITS.maxBytes) throw new ImportError('too-large', 'Image exceeds 12 MB.');
    return probeImageHeader(await readHeaderBytes(file, IMPORT_LIMITS.headerBytes), file.size);
  }

  function fitSize(width, height, edge) {
    const scale = Math.min(1, edge / Math.max(width, height));
    return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
  }

  function makeCanvas(size, doc) {
    if (doc && typeof doc.createElement === 'function') {
      const canvas = doc.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      return canvas;
    }
    if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(size, size);
    throw new ImportError('unsupported', 'Canvas is unavailable.');
  }

  /**
   * Decode on the phone into one transparent maxEdge-square canvas (letterboxed, alpha kept).
   * Only one ImageBitmap is alive at a time and it is always closed.
   *
   * EXIF orientation 5-8 swaps the displayed axes. Browsers differ in whether resize
   * options apply before or after orientation, so for those files only resizeWidth is
   * passed with the SHORT side's target: the aspect ratio is preserved either way and
   * the longest decoded edge never exceeds maxEdge. The drawn size comes from the
   * decoded bitmap itself, so a rotated result is letterboxed without distortion.
   */
  async function decodeToSquare(file, header, options = {}) {
    const edge = options.edge || IMPORT_LIMITS.maxEdge;
    const create = options.createImageBitmap || (typeof createImageBitmap === 'function' ? createImageBitmap : null);
    if (!create) throw new ImportError('unsupported', 'This browser cannot decode images.');
    const swapped = header.orientation >= 5 && header.orientation <= 8;
    const scale = Math.min(1, edge / Math.max(header.width, header.height));
    const resize = swapped
      ? { resizeWidth: Math.max(1, Math.round(Math.min(header.width, header.height) * scale)) }
      : { resizeWidth: Math.max(1, Math.round(header.width * scale)), resizeHeight: Math.max(1, Math.round(header.height * scale)) };
    let bitmap = null;
    try {
      try {
        bitmap = await create(file, { ...resize, resizeQuality: 'high', imageOrientation: 'from-image',
          premultiplyAlpha: 'default', colorSpaceConversion: 'default' });
      } catch (resizeError) {
        // Header pixel budget was already enforced, so a full decode is bounded.
        bitmap = await create(file);
      }
      if (!bitmap || !(bitmap.width > 0) || !(bitmap.height > 0)) throw new ImportError('corrupt', 'Image could not be decoded.');
      const decodedRatio = bitmap.width / bitmap.height;
      const rawRatio = header.width / header.height;
      const near = (a, b) => Math.abs(a - b) / b <= 0.02;
      if (!near(decodedRatio, rawRatio) && !near(decodedRatio, 1 / rawRatio)) {
        throw new ImportError('corrupt', 'Decoded image does not match its header dimensions.');
      }
      if (options.isCurrent && !options.isCurrent()) return null;
      const fit = fitSize(bitmap.width, bitmap.height, edge);
      const canvas = makeCanvas(edge, options.document);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, edge, edge);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(bitmap, Math.round((edge - fit.width) / 2), Math.round((edge - fit.height) / 2), fit.width, fit.height);
      return { canvas, width: fit.width, height: fit.height, sourceWidth: header.width, sourceHeight: header.height };
    } catch (error) {
      if (error instanceof ImportError) throw error;
      throw new ImportError('corrupt', 'Image could not be decoded: ' + (error && error.message || error));
    } finally {
      if (bitmap && typeof bitmap.close === 'function') bitmap.close();
    }
  }

  /** Generation token: every import/generate/stop/reset/pagehide invalidates older async work. */
  class WorkToken {
    constructor() { this.value = 0; }
    next() { this.value += 1; return this.value; }
    invalidate() { this.value += 1; }
    isCurrent(token) { return token === this.value; }
  }

  const SUPERSEDED = Object.freeze({ ok: false, code: 'superseded', message: 'Import was superseded.' });

  /** One file: validate header and budget first, then decode. Resolves; never rejects. */
  async function importImage(file, isCurrent, options = {}) {
    try {
      const header = await validateImageFile(file);
      if (!isCurrent()) return SUPERSEDED;
      const raster = await decodeToSquare(file, header, { ...options, isCurrent });
      if (!raster || !isCurrent()) {
        if (raster && raster.canvas) { raster.canvas.width = 1; raster.canvas.height = 1; }
        return SUPERSEDED;
      }
      return { ok: true, raster: { ...raster, type: header.type, name: String(file.name || 'imported image') } };
    } catch (error) {
      const code = error instanceof ImportError ? error.code : 'failed';
      return { ok: false, code, message: error && error.message ? error.message : String(error) };
    }
  }

  /**
   * At most ONE decode in flight and at most ONE newest file waiting. A newer submit
   * or cancel() supersedes older work through the shared WorkToken; nothing queues
   * beyond one file, and superseded promises resolve with code 'superseded'.
   */
  class ImportGate {
    constructor(tokens, options = {}) {
      this.tokens = tokens;
      this.options = options;
      this.active = null;
      this.queued = null;
    }

    submit(file) {
      const token = this.tokens.next();
      if (this.queued) { this.queued.resolve(SUPERSEDED); this.queued = null; }
      return new Promise((resolve) => {
        const job = { file, token, resolve };
        if (this.active) this.queued = job;
        else this.run(job);
      });
    }

    async run(job) {
      this.active = job;
      const isCurrent = () => this.tokens.isCurrent(job.token);
      let result = SUPERSEDED;
      try {
        if (isCurrent()) result = await importImage(job.file, isCurrent, this.options);
      } finally {
        this.active = null;
      }
      // ANY outdated outcome (success or failure) is superseded; only the latest is reported.
      if (!isCurrent()) {
        if (result.ok && result.raster && result.raster.canvas) { result.raster.canvas.width = 1; result.raster.canvas.height = 1; }
        result = SUPERSEDED;
      }
      job.resolve(result);
      const next = this.queued;
      this.queued = null;
      if (next) {
        if (this.tokens.isCurrent(next.token)) this.run(next);
        else next.resolve(SUPERSEDED);
      }
    }

    cancel() {
      this.tokens.invalidate();
      if (this.queued) { this.queued.resolve(SUPERSEDED); this.queued = null; }
    }

    get busy() { return Boolean(this.active); }
  }

  // ---------------------------------------------------------- artwork source
  function recipeSource(recipe) {
    return Object.freeze({ kind: 'recipe', recipe });
  }

  function rasterSource(raster) {
    return Object.freeze({ kind: 'raster', name: raster.name, type: raster.type,
      width: raster.width, height: raster.height });
  }

  /** Studio round-trip only exists for procedural recipes; rasters export PNG instead. */
  function studioHandoffFor(source) {
    if (source && source.kind === 'recipe') return { supported: true };
    return { supported: false,
      message: 'Imported images cannot be edited in the studio. Use Export artwork to save a PNG.' };
  }

  // --------------------------------------------------------- candidate gating
  /** Confirm is enabled only with an observed mask, valid tracking and a fresh timestamp. */
  function canConfirm(state) {
    if (!state || !state.candidateMask || state.trackingValid !== true) return false;
    const age = Number(state.now) - Number(state.maskTimestamp);
    return Number.isFinite(age) && age >= 0 && age <= Number(state.maxAgeMs);
  }

  // --------------------------------------------------------- entrance motion
  function prefersReducedMotion(win) {
    try {
      return Boolean(win && win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (error) {
      return false;
    }
  }

  /** fromY: pixels from settled centre down to just below the bottom edge near the composer. */
  function entranceKeyframes(fromY) {
    const y = Math.max(0, Math.round(Number(fromY) || 0));
    return [
      { transform: `translate3d(0, ${y}px, 0) scale(.35) rotateY(-360deg) rotateX(50deg) rotateZ(-12deg)`, opacity: 0.2, offset: 0 },
      { opacity: 1, offset: 0.25 },
      { transform: 'translate3d(0, 0, 0) scale(1) rotateY(0deg) rotateX(0deg) rotateZ(0deg)', opacity: 1, offset: 1 },
    ];
  }

  /**
   * Animates the entrance wrapper only (never the interactive transform element), so
   * gestures never jump. Exposes `entering`, `startedAt`, `generation` for tests.
   */
  class EntranceAnimator {
    constructor(element, options = {}) {
      this.element = element;
      this.window = options.window || (typeof window !== 'undefined' ? window : null);
      this.onSettled = options.onSettled || null;
      this.animation = null;
      this.timer = null;
      this.entering = false;
      this.generation = 0;
      this.startedAt = null;
      this.reduced = false;
    }

    play() {
      this.cancel();
      const generation = ++this.generation;
      const el = this.element;
      this.reduced = prefersReducedMotion(this.window);
      this.entering = true;
      this.startedAt = Date.now();
      if (el && el.dataset) el.dataset.entering = 'true';
      const duration = this.reduced ? REDUCED_MS : ENTRANCE_MS;
      const viewport = this.window && this.window.innerHeight ? this.window.innerHeight : 800;
      const frames = this.reduced ? [{ opacity: 0 }, { opacity: 1 }] : entranceKeyframes(viewport * 0.6);
      if (el && typeof el.animate === 'function') {
        this.animation = el.animate(frames, { duration, easing: 'cubic-bezier(.16, 1, .3, 1)', fill: 'none' });
        this.animation.onfinish = () => this.settle(generation);
      }
      // Timer fallback guarantees settlement even without Web Animations.
      const setT = (this.window && this.window.setTimeout) ? this.window.setTimeout.bind(this.window) : setTimeout;
      this.timer = setT(() => this.settle(generation), duration + 50);
      return generation;
    }

    settle(generation) {
      if (generation !== this.generation || !this.entering) return;
      this.clearTimer();
      this.animation = null;
      this.entering = false;
      if (this.element && this.element.dataset) delete this.element.dataset.entering;
      if (this.onSettled) this.onSettled(generation);
    }

    clearTimer() {
      if (this.timer !== null) {
        const clearT = (this.window && this.window.clearTimeout) ? this.window.clearTimeout.bind(this.window) : clearTimeout;
        clearT(this.timer);
        this.timer = null;
      }
    }

    cancel() {
      this.generation += 1;
      this.clearTimer();
      if (this.animation) {
        this.animation.onfinish = null;
        try { this.animation.cancel(); } catch (error) { /* already finished */ }
        this.animation = null;
      }
      this.entering = false;
      if (this.element && this.element.dataset) delete this.element.dataset.entering;
    }

    state() {
      return { entering: this.entering, generation: this.generation, startedAt: this.startedAt,
        durationMs: this.reduced ? REDUCED_MS : ENTRANCE_MS, reducedMotion: this.reduced };
    }
  }

  // ------------------------------------------------------- keyboard viewport
  /**
   * Keeps the Ask composer above the iOS keyboard (Safari and standalone) by writing
   * --keyboard-inset on the root element from visualViewport. Returns cleanup.
   */
  function bindKeyboardInset(win, rootElement) {
    const vv = win && win.visualViewport;
    const rootEl = rootElement || (win && win.document && win.document.documentElement);
    if (!vv || !rootEl) return () => {};
    let frame = 0;
    const update = () => {
      frame = 0;
      const inset = Math.max(0, Math.round(win.innerHeight - (vv.height + vv.offsetTop)));
      rootEl.style.setProperty('--keyboard-inset', inset + 'px');
    };
    const schedule = () => {
      if (!frame) frame = (win.requestAnimationFrame || ((cb) => win.setTimeout(cb, 16)))(update);
    };
    vv.addEventListener('resize', schedule);
    vv.addEventListener('scroll', schedule);
    update();
    return () => {
      vv.removeEventListener('resize', schedule);
      vv.removeEventListener('scroll', schedule);
      if (frame && win.cancelAnimationFrame) win.cancelAnimationFrame(frame);
      rootEl.style.removeProperty('--keyboard-inset');
    };
  }

  // -------------------------------------------------------- install helper
  /** Non-blocking Add to Home Screen hint: iOS Safari, not already standalone. */
  function shouldShowHomeScreenHint(win) {
    if (!win || !win.navigator) return false;
    const ua = String(win.navigator.userAgent || '');
    const iOS = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && win.navigator.maxTouchPoints > 1);
    const safari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS/.test(ua);
    const standalone = win.navigator.standalone === true ||
      Boolean(win.matchMedia && win.matchMedia('(display-mode: standalone)').matches);
    return iOS && safari && !standalone;
  }

  return Object.freeze({
    IMPORT_LIMITS, ENTRANCE_MS, REDUCED_MS, ImportError, probeImageHeader, validateImageFile,
    decodeToSquare, importImage, ImportGate, fitSize, WorkToken, recipeSource, rasterSource, studioHandoffFor,
    canConfirm, prefersReducedMotion, EntranceAnimator, entranceKeyframes, bindKeyboardInset,
    shouldShowHomeScreenHint,
  });
});
