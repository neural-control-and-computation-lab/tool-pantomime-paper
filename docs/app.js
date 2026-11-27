/* ==========================================================================
 * Interactive Companion & Live Two-Pathway Ridge Sandbox
 * Automated Kinematic Scoring of Tool-Use Pantomime in Limb Apraxia
 * ========================================================================== */

(function () {
  "use strict";

  const ALPHA_STEPS = [0.5, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0];

  // High-yield 12-item compact battery combining strong apraxia discrimination,
  // reliable kinematic predictability, and diverse mechanical demands (Discussion)
  const COMPACT_12_ITEM_NAMES = new Set([
    "Lipstick",
    "Syringe",
    "Nail Clipper",
    "Fork",
    "Corkscrew",
    "Iron",
    "Toothbrush",
    "Lighter",
    "Clamp",
    "Scissors",
    "Screwdriver",
    "Hammer",
  ]);

  // Lowest 6 standalone apraxia AUC items
  const LOWEST_6_ITEM_NAMES = new Set([
    "Beer Mug",
    "Paper Clip",
    "Clothing Pin",
    "Key",
    "Remote Control",
    "Soap Dispenser",
  ]);

  const STATE = {
    data: null,
    split: "full53", // "full53" | "train41" | "test12"
    blendItemWeight: 0.5, // 0 = 100% Participant, 0.5 = 50/50, 1 = 100% Item
    alpha: 10.0,
    cutoff: 2.52,
    activeFeatures: new Set(),
    activeItems: new Set(),
    selectedParticipant: "ST028",
    featureSearch: "",
    itemSearch: "",
    traceItem: "Corkscrew",
    showWrist: true,
    showIndex: false,
    showThumb: false,
    traceTimePct: 100,
    playingTrace: false,
    rotElev: 18,
    rotAzim: -62,
    atlasXMode: "manual_auc", // "manual_auc" | "kinematic_auc"
    atlasSortKey: "manual_auc",
    atlasSortAsc: false,
    atlasSearch: "",
    baselineMetrics: {},
    pathwayCache: new Map(),
    lastResult: null,
  };

  // ========================================================================
  // 1. EXACT LINEAR ALGEBRA & STATISTICAL HELPERS (MATCHING MODEL.PY)
  // ========================================================================

  function clip(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // Pre-allocated scratch buffers for zero-GC inner Ridge solves (max dim 128)
  const _SCRATCH_G = new Float64Array(128 * 128);
  const _SCRATCH_L = new Float64Array(128 * 128);
  const _SCRATCH_Z = new Float64Array(128);
  const _SCRATCH_B = new Float64Array(128);

  /** Cholesky solve G u = b in-place using pre-allocated scratch when n <= 128. */
  function solveSymPosDef(G, b, n) {
    const L = n <= 128 ? _SCRATCH_L : new Float64Array(n * n);
    const z = n <= 128 ? _SCRATCH_Z : new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const iOff = i * n;
      for (let j = 0; j <= i; j++) {
        let sum = G[iOff + j];
        const jOff = j * n;
        for (let k = 0; k < j; k++) {
          sum -= L[iOff + k] * L[jOff + k];
        }
        if (i === j) {
          L[iOff + i] = Math.sqrt(sum > 1e-12 ? sum : 1e-12);
        } else {
          L[iOff + j] = sum / L[jOff + j];
        }
      }
    }
    for (let i = 0; i < n; i++) {
      const iOff = i * n;
      let sum = b[i];
      for (let k = 0; k < i; k++) sum -= L[iOff + k] * z[k];
      z[i] = sum / L[iOff + i];
    }
    const u = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = z[i];
      for (let k = i + 1; k < n; k++) sum -= L[k * n + i] * u[k];
      u[i] = sum / L[i * n + i];
    }
    return u;
  }

  /**
   * Fit Ridge regression matching `sklearn.linear_model.Ridge(alpha=alpha)`:
   * Dispatches primal (p x p) when p < n and dual Woodbury (n x n) when n <= p.
   */
  function fitRidge(X, y, n, p, alpha) {
    let yMean = 0.0;
    for (let i = 0; i < n; i++) yMean += y[i];
    yMean /= n;

    const xMean = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) xMean[j] += X[off + j];
    }
    for (let j = 0; j < p; j++) xMean[j] /= n;

    const Xc = new Float64Array(n * p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) Xc[off + j] = X[off + j] - xMean[j];
    }

    const yc = new Float64Array(n);
    for (let i = 0; i < n; i++) yc[i] = y[i] - yMean;

    let coef;
    if (p < n) {
      // Primal solve: (Xc^T Xc + alpha I_p) coef = Xc^T yc
      const G = p <= 128 ? _SCRATCH_G : new Float64Array(p * p);
      const rhs = p <= 128 ? _SCRATCH_B : new Float64Array(p);
      for (let j = 0; j < p; j++) {
        rhs[j] = 0.0;
        for (let k = 0; k <= j; k++) G[j * p + k] = 0.0;
      }
      for (let i = 0; i < n; i++) {
        const off = i * p;
        const yci = yc[i];
        for (let j = 0; j < p; j++) {
          const xij = Xc[off + j];
          rhs[j] += xij * yci;
          const jOff = j * p;
          for (let k = 0; k <= j; k++) {
            G[jOff + k] += xij * Xc[off + k];
          }
        }
      }
      for (let j = 0; j < p; j++) {
        G[j * p + j] += alpha;
        for (let k = 0; k < j; k++) G[k * p + j] = G[j * p + k];
      }
      coef = solveSymPosDef(G, rhs, p);
    } else {
      // Dual Woodbury solve: (Xc Xc^T + alpha I_n) u = yc, coef = Xc^T u
      const G = n <= 128 ? _SCRATCH_G : new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        const offI = i * p;
        for (let j = 0; j <= i; j++) {
          const offJ = j * p;
          let dot = 0.0;
          for (let k = 0; k < p; k++) dot += Xc[offI + k] * Xc[offJ + k];
          if (i === j) dot += alpha;
          G[i * n + j] = dot;
          G[j * n + i] = dot;
        }
      }
      const u = solveSymPosDef(G, yc, n);
      coef = new Float64Array(p);
      for (let i = 0; i < n; i++) {
        const ui = u[i];
        const off = i * p;
        for (let j = 0; j < p; j++) coef[j] += Xc[off + j] * ui;
      }
    }

    let intercept = yMean;
    for (let j = 0; j < p; j++) intercept -= xMean[j] * coef[j];
    return { coef, intercept };
  }

  /** Standardize matrix X (n x p) with population mean/std (matching sklearn StandardScaler). */
  function fitStandardScaler(X, n, p) {
    const mean = new Float64Array(p);
    const scale = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) mean[j] += X[off + j];
    }
    for (let j = 0; j < p; j++) mean[j] /= n;

    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) {
        const d = X[off + j] - mean[j];
        scale[j] += d * d;
      }
    }
    for (let j = 0; j < p; j++) {
      const s = Math.sqrt(scale[j] / n);
      scale[j] = s > 1e-12 ? s : 1.0;
    }

    const Xs = new Float64Array(n * p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) Xs[off + j] = (X[off + j] - mean[j]) / scale[j];
    }
    return { mean, scale, Xs };
  }

  /** Contiguous K-fold cross_val_predict MSE matching `sklearn.model_selection.KFold(n_splits=k)`. */
  function kfoldRidgeMse(Xs, y, n, p, alpha, kSplits) {
    const k = Math.min(kSplits, n);
    const foldSizes = new Int32Array(k);
    const base = Math.floor(n / k);
    const rem = n % k;
    for (let f = 0; f < k; f++) foldSizes[f] = base + (f < rem ? 1 : 0);

    let mseSum = 0.0;
    let start = 0;
    for (let f = 0; f < k; f++) {
      const end = start + foldSizes[f];
      const nTr = n - (end - start);
      const Xtr = new Float64Array(nTr * p);
      const ytr = new Float64Array(nTr);
      let dst = 0;
      for (let i = 0; i < n; i++) {
        if (i >= start && i < end) continue;
        ytr[dst] = y[i];
        Xtr.set(Xs.subarray(i * p, (i + 1) * p), dst * p);
        dst++;
      }
      const mdl = fitRidge(Xtr, ytr, nTr, p, alpha);
      for (let i = start; i < end; i++) {
        const off = i * p;
        let pred = mdl.intercept;
        for (let j = 0; j < p; j++) pred += Xs[off + j] * mdl.coef[j];
        const diff = pred - y[i];
        mseSum += diff * diff;
      }
      start = end;
    }
    return mseSum / n;
  }

  /** Golden-section + parabolic bounded 1D minimization on [a, b] matching scipy minimize_scalar. */
  function minimizeScalarBounded(fn, a, b) {
    const phi = 0.3819660112501051;
    let x = a + phi * (b - a);
    let w = x;
    let v = x;
    let fx = fn(x);
    let fw = fx;
    let fv = fx;
    let d = 0.0;
    let e = 0.0;

    for (let iter = 0; iter < 45; iter++) {
      const xm = 0.5 * (a + b);
      const tol1 = 1e-6 * Math.abs(x) + 1e-9;
      const tol2 = 2.0 * tol1;
      if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) break;

      let useGolden = true;
      if (Math.abs(e) > tol1) {
        const r = (x - w) * (fx - fv);
        let q = (x - v) * (fx - fw);
        let p = (x - v) * q - (x - w) * r;
        q = 2.0 * (q - r);
        if (q > 0.0) p = -p;
        q = Math.abs(q);
        const etemp = e;
        e = d;
        if (Math.abs(p) < Math.abs(0.5 * q * etemp) && p > q * (a - x) && p < q * (b - x)) {
          d = p / q;
          const u = x + d;
          if (u - a < tol2 || b - u < tol2) d = x < xm ? tol1 : -tol1;
          useGolden = false;
        }
      }
      if (useGolden) {
        e = x >= xm ? a - x : b - x;
        d = phi * e;
      }
      const u = Math.abs(d) >= tol1 ? x + d : x + (d > 0 ? tol1 : -tol1);
      const fu = fn(u);
      if (fu <= fx) {
        if (u >= x) a = x;
        else b = x;
        v = w;
        fv = fw;
        w = x;
        fw = fx;
        x = u;
        fx = fu;
      } else {
        if (u < x) a = u;
        else b = u;
        if (fu <= fw || w === x) {
          v = w;
          fv = fw;
          w = u;
          fw = fu;
        } else if (fu <= fv || v === x || v === w) {
          v = u;
          fv = fu;
        }
      }
    }
    return x;
  }

  /** Fit a single BlendModel fold on active trials/features and return model parameters. */
  function fitBlendFold(trainIdx, allTrials, featIdx, alpha) {
    const p = featIdx.length;
    const nTr = trainIdx.length;
    let ySum = 0.0;

    const byItem = new Map();
    const byPart = new Map();

    for (let t = 0; t < nTr; t++) {
      const rowIdx = trainIdx[t];
      const yVal = allTrials.total[rowIdx];
      const itemId = allTrials.item[rowIdx];
      const pid = allTrials.participant[rowIdx];
      ySum += yVal;

      if (!byItem.has(itemId)) byItem.set(itemId, []);
      byItem.get(itemId).push(rowIdx);

      if (!byPart.has(pid)) byPart.set(pid, []);
      byPart.get(pid).push(rowIdx);
    }

    const globalMean = ySum / nTr;
    const itemModels = new Map();
    const itemScalers = new Map();
    const itemMeans = new Map();
    const itemWeights = new Map();

    for (const [itemId, rList] of byItem.entries()) {
      const ni = rList.length;
      let s = 0.0;
      const uniq = new Set();
      for (let i = 0; i < ni; i++) {
        const yv = allTrials.total[rList[i]];
        s += yv;
        uniq.add(yv);
      }
      itemMeans.set(itemId, s / ni);

      if (ni < 10 || uniq.size < 2) {
        itemWeights.set(itemId, 0.1);
        continue;
      }

      const Xi = new Float64Array(ni * p);
      const yi = new Float64Array(ni);
      for (let i = 0; i < ni; i++) {
        const r = rList[i];
        yi[i] = allTrials.total[r];
        const srcRow = allTrials.X[r];
        const off = i * p;
        for (let j = 0; j < p; j++) Xi[off + j] = srcRow[featIdx[j]];
      }

      const sc = fitStandardScaler(Xi, ni, p);
      itemScalers.set(itemId, sc);
      const mdl = fitRidge(sc.Xs, yi, ni, p, alpha);
      itemModels.set(itemId, mdl);

      const mse = kfoldRidgeMse(sc.Xs, yi, ni, p, alpha, 5);
      itemWeights.set(itemId, 1.0 / (mse + 0.01));
    }

    let wTot = 0.0;
    for (const v of itemWeights.values()) wTot += v;
    if (wTot > 0) {
      for (const [k, v] of itemWeights.entries()) itemWeights.set(k, v / wTot);
    }

    // Predict training trials for bias_factor optimization
    const partLoio = [];
    for (const [pid, rList] of byPart.entries()) {
      let wSum = 0.0;
      let wpSum = 0.0;
      let tSum = 0.0;
      for (let i = 0; i < rList.length; i++) {
        const r = rList[i];
        const itemId = allTrials.item[r];
        let pred = itemMeans.has(itemId) ? itemMeans.get(itemId) : globalMean;
        if (itemModels.has(itemId)) {
          const mdl = itemModels.get(itemId);
          const sc = itemScalers.get(itemId);
          const srcRow = allTrials.X[r];
          pred = mdl.intercept;
          for (let j = 0; j < p; j++) {
            pred += ((srcRow[featIdx[j]] - sc.mean[j]) / sc.scale[j]) * mdl.coef[j];
          }
        }
        pred = clip(pred, 0.0, 3.0);
        const w = itemWeights.has(itemId) ? itemWeights.get(itemId) : 0.1;
        wpSum += pred * w;
        wSum += w;
        tSum += allTrials.total[r];
      }
      partLoio.push({
        pid,
        weightedPred: wpSum / (wSum || 1.0),
        trueMean: tSum / rList.length,
      });
    }

    const biasFactor = minimizeScalarBounded((bias) => {
      let errSq = 0.0;
      for (let i = 0; i < partLoio.length; i++) {
        const d = partLoio[i];
        const pr = clip(globalMean + (d.weightedPred - globalMean) * (1.0 + bias), 0.0, 3.0);
        const df = pr - d.trueMean;
        errSq += df * df;
      }
      return errSq / partLoio.length;
    }, -1.0, 3.0);

    // Participant-level Ridge on [mean, std] (2*p inputs)
    const pids = Array.from(byPart.keys()).sort();
    const nParts = pids.length;
    const p2 = 2 * p;
    const Xparts = new Float64Array(nParts * p2);
    const yparts = new Float64Array(nParts);

    for (let idxP = 0; idxP < nParts; idxP++) {
      const pid = pids[idxP];
      const rList = byPart.get(pid);
      const m = rList.length;
      const off = idxP * p2;

      let yAcc = 0.0;
      for (let i = 0; i < m; i++) {
        const r = rList[i];
        yAcc += allTrials.total[r];
        const srcRow = allTrials.X[r];
        for (let j = 0; j < p; j++) Xparts[off + j] += srcRow[featIdx[j]];
      }
      yparts[idxP] = yAcc / m;
      for (let j = 0; j < p; j++) Xparts[off + j] /= m;

      for (let i = 0; i < m; i++) {
        const srcRow = allTrials.X[rList[i]];
        for (let j = 0; j < p; j++) {
          const diff = srcRow[featIdx[j]] - Xparts[off + j];
          Xparts[off + p + j] += diff * diff;
        }
      }
      for (let j = 0; j < p; j++) {
        Xparts[off + p + j] = Math.sqrt(Xparts[off + p + j] / m);
      }
    }

    const directScaler = fitStandardScaler(Xparts, nParts, p2);
    const directModel = fitRidge(directScaler.Xs, yparts, nParts, p2, alpha);

    // Store training-fold loio & direct predictions so calibration can be evaluated instantly for ANY blend weight!
    const trainLoioByPid = new Map();
    for (let i = 0; i < partLoio.length; i++) {
      const d = partLoio[i];
      trainLoioByPid.set(
        d.pid,
        clip(globalMean + (d.weightedPred - globalMean) * (1.0 + biasFactor), 0.0, 3.0)
      );
    }

    const trainDirectArr = new Float64Array(nParts);
    const trainLoioArr = new Float64Array(nParts);
    for (let idxP = 0; idxP < nParts; idxP++) {
      const off = idxP * p2;
      let pr = directModel.intercept;
      for (let j = 0; j < p2; j++) pr += directScaler.Xs[off + j] * directModel.coef[j];
      trainDirectArr[idxP] = clip(pr, 0.0, 3.0);
      trainLoioArr[idxP] = trainLoioByPid.get(pids[idxP]);
    }

    return {
      p,
      featIdx,
      globalMean,
      biasFactor,
      itemModels,
      itemScalers,
      itemMeans,
      itemWeights,
      directScaler,
      directModel,
      trainDirectArr,
      trainLoioArr,
      trainTrueArr: yparts,
    };
  }

  /** Predict held-out trials for one or more test participants using a fitted fold model. */
  function predictFoldParticipants(mdl, testIdx, allTrials) {
    const p = mdl.p;
    const p2 = 2 * p;
    const featIdx = mdl.featIdx;
    const byPart = new Map();
    for (let t = 0; t < testIdx.length; t++) {
      const r = testIdx[t];
      const pid = allTrials.participant[r];
      if (!byPart.has(pid)) byPart.set(pid, []);
      byPart.get(pid).push(r);
    }

    const out = {};
    for (const [pid, rList] of byPart.entries()) {
      const m = rList.length;
      let wSum = 0.0;
      let wpSum = 0.0;
      const itemDetail = [];

      const agg = new Float64Array(p2);
      for (let i = 0; i < m; i++) {
        const r = rList[i];
        const itemId = allTrials.item[r];
        const srcRow = allTrials.X[r];
        for (let j = 0; j < p; j++) agg[j] += srcRow[featIdx[j]];

        let pred = mdl.itemMeans.has(itemId) ? mdl.itemMeans.get(itemId) : mdl.globalMean;
        if (mdl.itemModels.has(itemId)) {
          const im = mdl.itemModels.get(itemId);
          const sc = mdl.itemScalers.get(itemId);
          pred = im.intercept;
          for (let j = 0; j < p; j++) {
            pred += ((srcRow[featIdx[j]] - sc.mean[j]) / sc.scale[j]) * im.coef[j];
          }
        }
        pred = clip(pred, 0.0, 3.0);
        const w = mdl.itemWeights.has(itemId) ? mdl.itemWeights.get(itemId) : 0.1;
        wpSum += pred * w;
        wSum += w;
        itemDetail.push({
          item: itemId,
          pred,
          trueScore: allTrials.total[r],
          weight: w,
        });
      }

      for (let j = 0; j < p; j++) agg[j] /= m;
      for (let i = 0; i < m; i++) {
        const srcRow = allTrials.X[rList[i]];
        for (let j = 0; j < p; j++) {
          const diff = srcRow[featIdx[j]] - agg[j];
          agg[p + j] += diff * diff;
        }
      }
      for (let j = 0; j < p; j++) agg[p + j] = Math.sqrt(agg[p + j] / m);

      const rawItem = wpSum / (wSum || 1.0);
      const loioPred = clip(mdl.globalMean + (rawItem - mdl.globalMean) * (1.0 + mdl.biasFactor), 0.0, 3.0);

      let dirRaw = mdl.directModel.intercept;
      const featContrib = new Float64Array(46);
      for (let j = 0; j < p; j++) {
        const zMean = (agg[j] - mdl.directScaler.mean[j]) / mdl.directScaler.scale[j];
        const zSd = (agg[p + j] - mdl.directScaler.mean[p + j]) / mdl.directScaler.scale[p + j];
        const cMean = zMean * mdl.directModel.coef[j];
        const cSd = zSd * mdl.directModel.coef[p + j];
        dirRaw += cMean + cSd;
        featContrib[featIdx[j]] = cMean + cSd;
      }
      const directPred = clip(dirRaw, 0.0, 3.0);

      out[pid] = {
        direct: directPred,
        loio: loioPred,
        featContrib,
        itemDetail,
        trainDirectArr: mdl.trainDirectArr,
        trainLoioArr: mdl.trainLoioArr,
        trainTrueArr: mdl.trainTrueArr,
      };
    }
    return out;
  }

  /** Apply linear variance-matching calibration for any blend weight `bw` (0..1) in < 0.2 ms. */
  function combineWithCalibration(foldOutputs, trueByPid, bw) {
    const partPred = {};
    const itemPred = {};
    const blendPred = {};

    for (const pid of Object.keys(foldOutputs)) {
      const fo = foldOutputs[pid];
      partPred[pid] = fo.direct;
      itemPred[pid] = fo.loio;

      const n = fo.trainTrueArr.length;
      let mTrue = 0.0;
      let mBlend = 0.0;
      const trBlended = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const b = bw * fo.trainLoioArr[i] + (1.0 - bw) * fo.trainDirectArr[i];
        trBlended[i] = b;
        mBlend += b;
        mTrue += fo.trainTrueArr[i];
      }
      mBlend /= n;
      mTrue /= n;

      let vBlend = 0.0;
      let vTrue = 0.0;
      for (let i = 0; i < n; i++) {
        const db = trBlended[i] - mBlend;
        const dt = fo.trainTrueArr[i] - mTrue;
        vBlend += db * db;
        vTrue += dt * dt;
      }
      const sBlend = Math.sqrt(vBlend / n);
      const sTrue = Math.sqrt(vTrue / n);
      let slope = 1.0;
      let intercept = 0.0;
      if (sBlend > 1e-8) {
        slope = sTrue / sBlend;
        intercept = mTrue - slope * mBlend;
      }
      const raw = bw * fo.loio + (1.0 - bw) * fo.direct;
      blendPred[pid] = clip(slope * raw + intercept, 0.0, 3.0);
    }
    return { partPred, itemPred, blendPred, trueByPid };
  }

  /** Run or retrieve cached pathways for current (split, alpha, activeFeatures, activeItems). */
  function evaluateActivePipeline() {
    const t0 = performance.now();
    const featIdx = Array.from(STATE.activeFeatures).sort((a, b) => a - b);
    const itemSet = STATE.activeItems;
    const key = `${STATE.split}|${STATE.alpha}|${featIdx.join(",")}|${Array.from(itemSet).sort((a, b) => a - b).join(",")}`;

    let cached = STATE.pathwayCache.get(key);
    if (!cached) {
      const allTrials = STATE.data.trials;
      const testSet = new Set(STATE.data.test_participants);
      const nRows = allTrials.total.length;

      // True score per participant is defined on the selected tool battery (or full battery if participant missed all selected tools)
      const fullTrueByPid = {};
      for (const p of STATE.data.participants) fullTrueByPid[p.id] = p.true_score;

      const filteredIdx = [];
      const partTrialScores = new Map();
      for (let r = 0; r < nRows; r++) {
        if (!itemSet.has(allTrials.item[r])) continue;
        const pid = allTrials.participant[r];
        if (STATE.split === "train41" && testSet.has(pid)) continue;
        filteredIdx.push(r);
        if (!partTrialScores.has(pid)) partTrialScores.set(pid, []);
        partTrialScores.get(pid).push(allTrials.total[r]);
      }

      const trueByPid = {};
      for (const [pid, arr] of partTrialScores.entries()) {
        if (STATE.split === "test12" && !testSet.has(pid)) continue;
        trueByPid[pid] = arr.reduce((a, b) => a + b, 0) / arr.length;
      }

      const foldOutputs = {};
      const accumAbsCoef = new Float64Array(92);
      const accumSignedCoef = new Float64Array(92);
      let nModels = 0;

      if (STATE.split === "test12") {
        const trIdx = [];
        const teIdx = [];
        for (let i = 0; i < filteredIdx.length; i++) {
          const r = filteredIdx[i];
          if (testSet.has(allTrials.participant[r])) teIdx.push(r);
          else trIdx.push(r);
        }
        const mdl = fitBlendFold(trIdx, allTrials, featIdx, STATE.alpha);
        Object.assign(foldOutputs, predictFoldParticipants(mdl, teIdx, allTrials));
        for (let j = 0; j < featIdx.length; j++) {
          const f = featIdx[j];
          accumAbsCoef[f] += Math.abs(mdl.directModel.coef[j]);
          accumAbsCoef[46 + f] += Math.abs(mdl.directModel.coef[featIdx.length + j]);
          accumSignedCoef[f] += mdl.directModel.coef[j];
          accumSignedCoef[46 + f] += mdl.directModel.coef[featIdx.length + j];
        }
        nModels = 1;
      } else {
        const evalPids = Object.keys(trueByPid).sort();
        for (let fIdx = 0; fIdx < evalPids.length; fIdx++) {
          const holdPid = evalPids[fIdx];
          const trIdx = [];
          const hoIdx = [];
          for (let i = 0; i < filteredIdx.length; i++) {
            const r = filteredIdx[i];
            if (allTrials.participant[r] === holdPid) hoIdx.push(r);
            else trIdx.push(r);
          }
          if (hoIdx.length === 0 || trIdx.length < 15) continue;
          const mdl = fitBlendFold(trIdx, allTrials, featIdx, STATE.alpha);
          Object.assign(foldOutputs, predictFoldParticipants(mdl, hoIdx, allTrials));
          for (let j = 0; j < featIdx.length; j++) {
            const f = featIdx[j];
            accumAbsCoef[f] += Math.abs(mdl.directModel.coef[j]);
            accumAbsCoef[46 + f] += Math.abs(mdl.directModel.coef[featIdx.length + j]);
            accumSignedCoef[f] += mdl.directModel.coef[j];
            accumSignedCoef[46 + f] += mdl.directModel.coef[featIdx.length + j];
          }
          nModels++;
        }
      }

      const relWeightPct = new Float64Array(46);
      const signedMeanCoef = new Float64Array(46);
      let totMag = 0.0;
      for (let f = 0; f < 46; f++) {
        const mag = (accumAbsCoef[f] + accumAbsCoef[46 + f]) / Math.max(1, nModels);
        relWeightPct[f] = mag;
        signedMeanCoef[f] = accumSignedCoef[f] / Math.max(1, nModels);
        totMag += mag;
      }
      if (totMag > 0) {
        for (let f = 0; f < 46; f++) relWeightPct[f] = (100.0 * relWeightPct[f]) / totMag;
      }

      cached = { foldOutputs, trueByPid, fullTrueByPid, relWeightPct, signedMeanCoef };
      STATE.pathwayCache.set(key, cached);
    }

    // Note: keep canonical manual apraxia score (`fullTrueByPid`) when all 39 items are active,
    // and battery-specific manual score when testing reduced tool batteries (matching clinical screening!).
    const combined = combineWithCalibration(cached.foldOutputs, cached.trueByPid, STATE.blendItemWeight);
    const elapsedMs = performance.now() - t0;

    return {
      ...combined,
      foldOutputs: cached.foldOutputs,
      relWeightPct: cached.relWeightPct,
      signedMeanCoef: cached.signedMeanCoef,
      elapsedMs,
    };
  }

  // ========================================================================
  // 2. METRICS: PEARSON r, SPEARMAN rho, MAE, RMSE, ROC AUC & CURVE
  // ========================================================================

  function pearsonR(x, y) {
    const n = x.length;
    if (n < 2) return 0;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    let num = 0, dx2 = 0, dy2 = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      num += dx * dy;
      dx2 += dx * dx;
      dy2 += dy * dy;
    }
    return dx2 > 1e-12 && dy2 > 1e-12 ? num / Math.sqrt(dx2 * dy2) : 0;
  }

  function rankArray(arr) {
    const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Float64Array(arr.length);
    let i = 0;
    while (i < indexed.length) {
      let j = i;
      while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) j++;
      const avgRank = 0.5 * (i + j) + 1;
      for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
      i = j + 1;
    }
    return ranks;
  }

  function computeRocAndAuc(labels, scores) {
    // labels: 1 = apraxic, 0 = non-apraxic; higher `scores` = more likely apraxic (-pred)
    const n = labels.length;
    let nPos = 0, nNeg = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] === 1) nPos++;
      else nNeg++;
    }
    if (nPos === 0 || nNeg === 0) return { auc: 0.5, curve: [[0, 0], [1, 1]] };

    const pairs = scores.map((s, i) => ({ s, y: labels[i] })).sort((a, b) => b.s - a.s);
    const curve = [[0, 0]];
    let tp = 0, fp = 0, prevTp = 0, prevFp = 0, auc = 0;
    let idx = 0;
    while (idx < n) {
      let end = idx;
      while (end + 1 < n && pairs[end + 1].s === pairs[idx].s) end++;
      for (let k = idx; k <= end; k++) {
        if (pairs[k].y === 1) tp++;
        else fp++;
      }
      auc += ((fp - prevFp) / nNeg) * (((tp + prevTp) * 0.5) / nPos);
      curve.push([fp / nNeg, tp / nPos]);
      prevTp = tp;
      prevFp = fp;
      idx = end + 1;
    }
    return { auc, curve };
  }

  function computeMetrics(predMap, trueMap, cutoff) {
    const pids = Object.keys(predMap).sort();
    const n = pids.length;
    const pArr = pids.map((k) => predMap[k]);
    const tArr = pids.map((k) => trueMap[k]);

    let absErr = 0, sqErr = 0;
    for (let i = 0; i < n; i++) {
      const e = pArr[i] - tArr[i];
      absErr += Math.abs(e);
      sqErr += e * e;
    }
    const r = pearsonR(pArr, tArr);
    const rho = pearsonR(rankArray(pArr), rankArray(tArr));
    const mae = absErr / Math.max(1, n);
    const rmse = Math.sqrt(sqErr / Math.max(1, n));

    const strokePids = pids.filter((k) => k.startsWith("ST"));
    const labels = strokePids.map((k) => (trueMap[k] < cutoff ? 1 : 0));
    const negPreds = strokePids.map((k) => -predMap[k]);
    const { auc, curve } = computeRocAndAuc(labels, negPreds);

    let tp = 0, fn = 0, tn = 0, fp = 0;
    for (let i = 0; i < strokePids.length; i++) {
      const p = strokePids[i];
      const tAp = trueMap[p] < cutoff;
      const pAp = predMap[p] < cutoff;
      if (tAp && pAp) tp++;
      else if (tAp && !pAp) fn++;
      else if (!tAp && !pAp) tn++;
      else fp++;
    }
    const sens = tp + fn > 0 ? tp / (tp + fn) : 0;
    const spec = tn + fp > 0 ? tn / (tn + fp) : 0;

    return { n, r, rho, mae, rmse, auc, curve, tp, fn, tn, fp, sens, spec };
  }

  // ========================================================================
  // 3. TOOLTIP & SVG RENDERING UTILITIES
  // ========================================================================

  const tooltipEl = typeof document !== "undefined" ? document.getElementById("tooltip") : null;
  function showTooltip(evt, html) {
    if (!tooltipEl) return;
    tooltipEl.innerHTML = html;
    tooltipEl.style.opacity = "1";
    const x = Math.min(window.innerWidth - 315, evt.clientX + 14);
    const y = Math.min(window.innerHeight - 110, evt.clientY + 14);
    tooltipEl.style.left = `${Math.max(8, x)}px`;
    tooltipEl.style.top = `${Math.max(8, y)}px`;
  }
  function hideTooltip() {
    if (tooltipEl) tooltipEl.style.opacity = "0";
  }

  function formatDeltaBadge(el, val, baseVal, higherIsBetter, digits = 3) {
    const diff = val - baseVal;
    if (Math.abs(diff) < 0.0005) {
      el.className = "kpi-delta neutral";
      el.textContent = "baseline";
      return;
    }
    const good = higherIsBetter ? diff > 0 : diff < 0;
    el.className = `kpi-delta ${good ? "good" : "warn"}`;
    const sign = diff > 0 ? "+" : "";
    el.textContent = `${sign}${diff.toFixed(digits)}`;
  }

  // ========================================================================
  // 4. SANDBOX CHARTS: SCATTER, PATHWAY BARS, CONFUSION MATRIX, ROC
  // ========================================================================

  function renderScatter(res) {
    const container = document.getElementById("scatter-container");
    const pids = Object.keys(res.blendPred).sort();
    const cutoff = STATE.cutoff;
    const W = 470, H = 430;
    const pad = { l: 54, r: 20, t: 18, b: 48 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const vals = [];
    for (const p of pids) {
      vals.push(res.blendPred[p], res.trueByPid[p]);
    }
    const mn = Math.min(1.65, Math.floor((Math.min(...vals) - 0.06) * 20) / 20);
    const mx = Math.max(3.08, Math.ceil((Math.max(...vals) + 0.04) * 20) / 20);
    const sx = (v) => pad.l + ((v - mn) / (mx - mn)) * pw;
    const sy = (v) => pad.t + (1 - (v - mn) / (mx - mn)) * ph;

    const cx = sx(clip(cutoff, mn, mx));
    const cy = sy(clip(cutoff, mn, mx));

    const ticks = [];
    for (let v = Math.ceil(mn * 5) / 5; v <= mx + 1e-6; v += 0.2) {
      ticks.push(Number(v.toFixed(2)));
    }

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="True versus predicted praxis score scatter plot">`;
    // Quadrant fills
    svg += `<rect x="${pad.l}" y="${cy}" width="${Math.max(0, cx - pad.l)}" height="${Math.max(0, pad.t + ph - cy)}" fill="#d32f2f" fill-opacity="0.085"/>`;
    svg += `<rect x="${cx}" y="${pad.t}" width="${Math.max(0, pad.l + pw - cx)}" height="${Math.max(0, cy - pad.t)}" fill="#1976d2" fill-opacity="0.085"/>`;

    // Grid & identity & cutoff lines
    for (const tv of ticks) {
      const tx = sx(tv), ty = sy(tv);
      svg += `<line x1="${tx}" y1="${pad.t}" x2="${tx}" y2="${pad.t + ph}" stroke="#ece9e1" stroke-width="1"/>`;
      svg += `<line x1="${pad.l}" y1="${ty}" x2="${pad.l + pw}" y2="${ty}" stroke="#ece9e1" stroke-width="1"/>`;
      svg += `<text x="${tx}" y="${pad.t + ph + 17}" text-anchor="middle" font-size="11" fill="#494e59">${tv.toFixed(1)}</text>`;
      svg += `<text x="${pad.l - 8}" y="${ty + 4}" text-anchor="end" font-size="11" fill="#494e59">${tv.toFixed(1)}</text>`;
    }

    svg += `<line x1="${sx(mn)}" y1="${sy(mn)}" x2="${sx(mx)}" y2="${sy(mx)}" stroke="#8c8f99" stroke-width="1.4" stroke-dasharray="5,4"/>`;
    svg += `<line x1="${cx}" y1="${pad.t}" x2="${cx}" y2="${pad.t + ph}" stroke="#595e6b" stroke-width="1.2" stroke-dasharray="4,3"/>`;
    svg += `<line x1="${pad.l}" y1="${cy}" x2="${pad.l + pw}" y2="${cy}" stroke="#595e6b" stroke-width="1.2" stroke-dasharray="4,3"/>`;

    svg += `<text x="${pad.l + 8}" y="${pad.t + ph - 10}" font-size="11.5" font-style="italic" font-weight="700" fill="#7f0000">With Apraxia</text>`;
    svg += `<text x="${pad.l + pw - 8}" y="${pad.t + 18}" text-anchor="end" font-size="11.5" font-style="italic" font-weight="700" fill="#0a2e5c">Without Apraxia</text>`;

    // Border & Axis labels
    svg += `<rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" fill="none" stroke="#333740" stroke-width="1.2"/>`;
    svg += `<text x="${pad.l + pw / 2}" y="${H - 8}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#181a1f">Manually Coded Praxis Score (0–3)</text>`;
    svg += `<text x="15" y="${pad.t + ph / 2}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#181a1f" transform="rotate(-90 15 ${pad.t + ph / 2})">Predicted Praxis Score (0–3)</text>`;

    // Points
    for (const pid of pids) {
      const tr = res.trueByPid[pid];
      const pr = res.blendPred[pid];
      const px = sx(tr);
      const py = sy(pr);
      const isCtrl = pid.startsWith("CL");
      const color = isCtrl ? "#0a1f4d" : tr < cutoff ? "#d32f2f" : "#1976d2";
      const mis = !isCtrl && (tr < cutoff) !== (pr < cutoff);
      const isSel = pid === STATE.selectedParticipant;

      if (isSel) {
        svg += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="10.5" fill="none" stroke="#181a1f" stroke-width="2" stroke-dasharray="2,2"/>`;
      }
      if (mis) {
        const d = 5.4;
        svg += `<g class="pt-node" data-pid="${pid}" style="cursor:pointer;">
          <line x1="${(px - d).toFixed(1)}" y1="${(py - d).toFixed(1)}" x2="${(px + d).toFixed(1)}" y2="${(py + d).toFixed(1)}" stroke="#181a1f" stroke-width="4.2" stroke-linecap="round"/>
          <line x1="${(px - d).toFixed(1)}" y1="${(py + d).toFixed(1)}" x2="${(px + d).toFixed(1)}" y2="${(py - d).toFixed(1)}" stroke="#181a1f" stroke-width="4.2" stroke-linecap="round"/>
          <line x1="${(px - d).toFixed(1)}" y1="${(py - d).toFixed(1)}" x2="${(px + d).toFixed(1)}" y2="${(py + d).toFixed(1)}" stroke="${color}" stroke-width="2.6" stroke-linecap="round"/>
          <line x1="${(px - d).toFixed(1)}" y1="${(py + d).toFixed(1)}" x2="${(px + d).toFixed(1)}" y2="${(py - d).toFixed(1)}" stroke="${color}" stroke-width="2.6" stroke-linecap="round"/>
        </g>`;
      } else {
        svg += `<circle class="pt-node" data-pid="${pid}" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5.8" fill="${color}" fill-opacity="0.9" stroke="#181a1f" stroke-width="1.1" style="cursor:pointer;"/>`;
      }
    }
    svg += `</svg>`;
    container.innerHTML = svg;

    container.querySelectorAll(".pt-node").forEach((node) => {
      const pid = node.getAttribute("data-pid");
      node.addEventListener("mousemove", (evt) => {
        const tr = res.trueByPid[pid];
        const pr = res.blendPred[pid];
        const pt = res.partPred[pid];
        const it = res.itemPred[pid];
        const grp = pid.startsWith("CL")
          ? "Healthy control"
          : tr < cutoff
          ? "Stroke: with apraxia"
          : "Stroke: without apraxia";
        showTooltip(
          evt,
          `<strong>${pid}</strong> (${grp})<br/>` +
            `Manual score: <strong>${tr.toFixed(3)}</strong><br/>` +
            `Blended pred: <strong>${pr.toFixed(3)}</strong> (err ${(pr - tr >= 0 ? "+" : "") + (pr - tr).toFixed(3)})<br/>` +
            `Part. pathway: ${pt.toFixed(3)} &middot; Item pathway: ${it.toFixed(3)}<br/>` +
            `<span style="color:#cbd5e1;">Click to inspect feature &amp; tool contributions</span>`
        );
      });
      node.addEventListener("mouseleave", hideTooltip);
      node.addEventListener("click", () => {
        STATE.selectedParticipant = pid;
        renderScatter(res);
        renderParticipantInspector(res);
        document.getElementById("inspector-section").scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    });
  }

  function renderPathwayBars(mPart, mItem, mBlend) {
    const container = document.getElementById("pathway-container");
    const W = 380, H = 195;
    const pad = { l: 42, r: 14, t: 22, b: 44 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const rows = [
      { label: "Participant", m: mPart },
      { label: "Item", m: mItem },
      { label: "Blended", m: mBlend },
    ];

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Pathway performance comparison">`;
    for (const yv of [0, 0.25, 0.5, 0.75, 1.0]) {
      const yy = pad.t + (1 - yv / 1.08) * ph;
      svg += `<line x1="${pad.l}" y1="${yy}" x2="${pad.l + pw}" y2="${yy}" stroke="#ece9e1" stroke-width="1"/>`;
      svg += `<text x="${pad.l - 6}" y="${yy + 3.5}" text-anchor="end" font-size="10" fill="#494e59">${yv.toFixed(2)}</text>`;
    }

    const colW = pw / 3;
    const barW = 28;
    rows.forEach((row, idx) => {
      const cx = pad.l + colW * (idx + 0.5);
      const rVal = Math.max(0, row.m.r);
      const aVal = Math.max(0, row.m.auc);
      const hr = (rVal / 1.08) * ph;
      const ha = (aVal / 1.08) * ph;
      const yBase = pad.t + ph;

      svg += `<rect x="${cx - barW - 2}" y="${yBase - hr}" width="${barW}" height="${hr}" fill="#4a4e58" rx="2"/>`;
      svg += `<rect x="${cx + 2}" y="${yBase - ha}" width="${barW}" height="${ha}" fill="#b8bcc6" rx="2"/>`;
      svg += `<text x="${cx - barW / 2 - 2}" y="${yBase - hr - 4}" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="#181a1f">${row.m.r.toFixed(2)}</text>`;
      svg += `<text x="${cx + barW / 2 + 2}" y="${yBase - ha - 4}" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="#181a1f">${row.m.auc.toFixed(2)}</text>`;
      svg += `<text x="${cx}" y="${yBase + 16}" text-anchor="middle" font-size="11" font-weight="600" fill="#181a1f">${row.label}</text>`;
      svg += `<text x="${cx}" y="${yBase + 30}" text-anchor="middle" font-size="9.5" font-family="var(--font-mono)" fill="#6e7482">MAE ${row.m.mae.toFixed(3)}</text>`;
    });

    svg += `<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ph}" stroke="#333740" stroke-width="1.1"/>`;
    svg += `<line x1="${pad.l}" y1="${pad.t + ph}" x2="${pad.l + pw}" y2="${pad.t + ph}" stroke="#333740" stroke-width="1.1"/>`;
    // Mini legend top-right
    svg += `<rect x="${W - 118}" y="4" width="9" height="9" fill="#4a4e58" rx="1.5"/>`;
    svg += `<text x="${W - 105}" y="12" font-size="10" font-style="italic" fill="#181a1f">r</text>`;
    svg += `<rect x="${W - 86}" y="4" width="9" height="9" fill="#b8bcc6" rx="1.5"/>`;
    svg += `<text x="${W - 73}" y="12" font-size="10" fill="#181a1f">AUC</text>`;
    svg += `</svg>`;
    container.innerHTML = svg;
  }

  function renderConfusionAndRoc(mBlend) {
    const cmWrap = document.getElementById("confusion-container");
    const rocWrap = document.getElementById("roc-container");

    const cells = [
      [mBlend.fn, mBlend.tn], // row 0: Pred without apraxia; col 0: True with, col 1: True without
      [mBlend.tp, mBlend.fp], // row 1: Pred with apraxia
    ];
    const maxC = Math.max(1, mBlend.fn, mBlend.tn, mBlend.tp, mBlend.fp);

    let cmSvg = `<svg viewBox="0 0 185 165" role="img" aria-label="Stroke apraxia confusion matrix">`;
    const ox = 46, oy = 14, sz = 54;
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const v = cells[r][c];
        const t = v / maxC;
        const red = Math.round(255 - t * (255 - 46));
        const grn = Math.round(255 - t * (255 - 125));
        const blu = Math.round(255 - t * (255 - 50));
        cmSvg += `<rect x="${ox + c * sz}" y="${oy + r * sz}" width="${sz - 2}" height="${sz - 2}" rx="4" fill="rgb(${red},${grn},${blu})" stroke="#e4e1d9"/>`;
        cmSvg += `<text x="${ox + c * sz + (sz - 2) / 2}" y="${oy + r * sz + 32}" text-anchor="middle" font-size="15" font-weight="700" font-family="var(--font-mono)" fill="${t > 0.58 ? "#fff" : "#181a1f"}">${v}</text>`;
      }
    }
    cmSvg += `<text x="${ox + 26}" y="${oy + 2 * sz + 13}" text-anchor="middle" font-size="9.5" fill="#494e59">With</text>`;
    cmSvg += `<text x="${ox + sz + 26}" y="${oy + 2 * sz + 13}" text-anchor="middle" font-size="9.5" fill="#494e59">Without</text>`;
    cmSvg += `<text x="${ox + sz}" y="${oy + 2 * sz + 28}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#181a1f">True Apraxia</text>`;
    cmSvg += `<text x="12" y="${oy + sz}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#181a1f" transform="rotate(-90 12 ${oy + sz})">Predicted</text>`;
    cmSvg += `<text x="${ox - 5}" y="${oy + 29}" text-anchor="end" font-size="9" fill="#494e59">Without</text>`;
    cmSvg += `<text x="${ox - 5}" y="${oy + sz + 29}" text-anchor="end" font-size="9" fill="#494e59">With</text>`;
    cmSvg += `</svg>`;
    cmWrap.innerHTML = cmSvg;

    // ROC Curve
    const rw = 185, rh = 165;
    const rp = { l: 34, r: 12, t: 12, b: 32 };
    const pw = rw - rp.l - rp.r;
    const ph = rh - rp.t - rp.b;
    const pts = mBlend.curve
      .map(([fpr, tpr]) => `${(rp.l + fpr * pw).toFixed(1)},${(rp.t + (1 - tpr) * ph).toFixed(1)}`)
      .join(" ");
    const polyPts = `${rp.l},${rp.t + ph} ${pts} ${rp.l + pw},${rp.t + ph}`;

    let rocSvg = `<svg viewBox="0 0 ${rw} ${rh}" role="img" aria-label="Stroke ROC curve">`;
    rocSvg += `<polygon points="${polyPts}" fill="#1976d2" fill-opacity="0.12"/>`;
    rocSvg += `<line x1="${rp.l}" y1="${rp.t + ph}" x2="${rp.l + pw}" y2="${rp.t}" stroke="#9ea4b0" stroke-dasharray="3,3"/>`;
    rocSvg += `<polyline points="${pts}" fill="none" stroke="#0a1f4d" stroke-width="2.2" stroke-linejoin="round"/>`;
    // Operating point at active cutoff
    const fprNow = mBlend.fp + mBlend.tn > 0 ? mBlend.fp / (mBlend.fp + mBlend.tn) : 0;
    const tprNow = mBlend.sens;
    rocSvg += `<circle cx="${(rp.l + fprNow * pw).toFixed(1)}" cy="${(rp.t + (1 - tprNow) * ph).toFixed(1)}" r="4.2" fill="#d32f2f" stroke="#fff" stroke-width="1.2"/>`;
    rocSvg += `<rect x="${rp.l}" y="${rp.t}" width="${pw}" height="${ph}" fill="none" stroke="#333740" stroke-width="1"/>`;
    rocSvg += `<text x="${rp.l + pw / 2}" y="${rh - 6}" text-anchor="middle" font-size="10" font-weight="600" fill="#181a1f">1 &minus; Specificity</text>`;
    rocSvg += `<text x="10" y="${rp.t + ph / 2}" text-anchor="middle" font-size="10" font-weight="600" fill="#181a1f" transform="rotate(-90 10 ${rp.t + ph / 2})">Sensitivity</text>`;
    rocSvg += `<text x="${rp.l + pw - 5}" y="${rp.t + ph - 8}" text-anchor="end" font-size="10" font-family="var(--font-mono)" font-weight="600" fill="#0a1f4d">AUC ${mBlend.auc.toFixed(3)}</text>`;
    rocSvg += `</svg>`;
    rocWrap.innerHTML = rocSvg;
  }

  function renderFamilyStackedBar(res) {
    const families = [
      { key: "Hand posture", lo: 0, hi: 19, color: "#882255" },
      { key: "Arm posture & trajectory", lo: 19, hi: 32, color: "#e69f00" },
      { key: "Amplitude & timing", lo: 32, hi: 45, color: "#44aa99" },
      { key: "Motion-specific", lo: 45, hi: 46, color: "#7a7a84" },
    ];
    const shares = families.map((fam) => {
      let s = 0;
      for (let i = fam.lo; i < fam.hi; i++) s += res.relWeightPct[i];
      return { ...fam, pct: s };
    });

    document.getElementById("family-share-readout").textContent = shares
      .map((s) => `${s.key.split(" ")[0]}: ${s.pct.toFixed(1)}%`)
      .join(" · ");

    let html = `<div style="display:flex; height:16px; border-radius:8px; overflow:hidden; background:#e5e2da;">`;
    for (const s of shares) {
      if (s.pct <= 0.05) continue;
      html += `<div style="width:${s.pct.toFixed(2)}%; background:${s.color}; transition:width 0.2s ease;" title="${s.key}: ${s.pct.toFixed(1)}%"></div>`;
    }
    html += `</div><div class="legend-inline" style="margin-top:0.45rem;">`;
    for (const s of shares) {
      html += `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}; border:none;"></span>${s.key}: <strong>${s.pct.toFixed(1)}%</strong></span>`;
    }
    html += `</div>`;
    document.getElementById("family-stacked-bar").innerHTML = html;
  }

  // ========================================================================
  // 5. FEATURE & TOOL CHECKBOX LISTS
  // ========================================================================

  function renderFeatureControls(res) {
    const fams = [
      { name: "Hand posture", short: "Hand (19)", color: "#882255", bg: "var(--fam-hand-bg)" },
      { name: "Arm posture & trajectory", short: "Arm (13)", color: "#e69f00", bg: "var(--fam-arm-bg)" },
      { name: "Amplitude & timing", short: "Amp/Time (13)", color: "#44aa99", bg: "var(--fam-amp-bg)" },
      { name: "Motion-specific", short: "Motion (1)", color: "#7a7a84", bg: "var(--fam-mot-bg)" },
    ];

    const famWrap = document.getElementById("family-toggles");
    famWrap.innerHTML = "";
    fams.forEach((fm) => {
      const members = STATE.data.features.filter((f) => f.family === fm.name);
      const activeCnt = members.filter((f) => STATE.activeFeatures.has(f.index)).length;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `fam-toggle ${activeCnt === 0 ? "off" : ""}`;
      btn.style.background = fm.bg;
      btn.innerHTML = `<span class="fam-dot" style="background:${fm.color};"></span>${fm.short}: ${activeCnt}/${members.length}`;
      btn.addEventListener("click", () => {
        const turnOff = activeCnt === members.length;
        if (turnOff && STATE.activeFeatures.size === members.length) return; // keep >= 1 feature active
        members.forEach((f) => {
          if (turnOff) STATE.activeFeatures.delete(f.index);
          else STATE.activeFeatures.add(f.index);
        });
        clearActivePresetHighlight();
        updateAll();
      });
      famWrap.appendChild(btn);
    });

    document.getElementById("count-features").textContent = `${STATE.activeFeatures.size}/46`;
    const listWrap = document.getElementById("feature-list");
    const q = STATE.featureSearch.trim().toLowerCase();
    const maxW = Math.max(4.0, ...Array.from(res.relWeightPct));

    const sorted = [...STATE.data.features].sort((a, b) => {
      const wa = res.relWeightPct[a.index] || a.rel_weight_53;
      const wb = res.relWeightPct[b.index] || b.rel_weight_53;
      return wb - wa;
    });

    listWrap.innerHTML = "";
    for (const f of sorted) {
      if (q && !f.name.toLowerCase().includes(q) && !f.family.toLowerCase().includes(q) && !f.description.toLowerCase().includes(q)) {
        continue;
      }
      const checked = STATE.activeFeatures.has(f.index);
      const wt = res.relWeightPct[f.index];
      const sign = res.signedMeanCoef[f.index];
      const row = document.createElement("label");
      row.className = "feat-item";
      row.innerHTML = `
        <input type="checkbox" ${checked ? "checked" : ""} data-fidx="${f.index}" />
        <div class="feat-name-wrap">
          <div class="feat-name">
            <span class="fam-dot" style="background:${f.color}; flex-shrink:0;"></span>
            <span>${f.name}</span>
            ${f.in_top8_train ? '<span class="badge-top8">Top 8</span>' : ""}
          </div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${Math.min(100, (wt / maxW) * 100).toFixed(1)}%; background:${f.color};"></div>
          </div>
        </div>
        <div class="feat-weight-num">${checked ? `${wt.toFixed(1)}% ${sign >= 0 ? "+" : "−"}` : "off"}</div>
      `;
      row.addEventListener("mousemove", (evt) => {
        showTooltip(
          evt,
          `<strong>${f.name}</strong><br/>` +
            `<span style="color:#cbd5e1;">${f.family}</span><br/>` +
            `${f.description}<br/>` +
            `Active weight share: <strong>${wt.toFixed(2)}%</strong> (full-sample baseline: ${f.rel_weight_53.toFixed(2)}%)`
        );
      });
      row.addEventListener("mouseleave", hideTooltip);
      const cb = row.querySelector("input");
      cb.addEventListener("change", () => {
        if (!cb.checked && STATE.activeFeatures.size <= 1) {
          cb.checked = true;
          return;
        }
        if (cb.checked) STATE.activeFeatures.add(f.index);
        else STATE.activeFeatures.delete(f.index);
        clearActivePresetHighlight();
        updateAll();
      });
      listWrap.appendChild(row);
    }
  }

  function renderItemControls() {
    document.getElementById("count-items").textContent = `${STATE.activeItems.size}/39`;
    const listWrap = document.getElementById("item-list");
    const q = STATE.itemSearch.trim().toLowerCase();
    const sorted = [...STATE.data.items].sort((a, b) => b.manual_auc - a.manual_auc);

    listWrap.innerHTML = "";
    for (const it of sorted) {
      if (q && !it.name.toLowerCase().includes(q) && !it.motion_type.toLowerCase().includes(q)) {
        continue;
      }
      const checked = STATE.activeItems.has(it.id);
      const row = document.createElement("label");
      row.className = "feat-item";
      row.style.gridTemplateColumns = "auto minmax(0, 1fr) 92px";
      row.innerHTML = `
        <input type="checkbox" ${checked ? "checked" : ""} data-iid="${it.id}" />
        <div class="feat-name-wrap">
          <div class="feat-name"><span>${it.name}</span></div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${((it.manual_auc - 0.45) / 0.4 * 100).toFixed(1)}%; background:var(--c-item);"></div>
          </div>
        </div>
        <div class="feat-weight-num">${it.manual_auc.toFixed(2)} &middot; ${it.weight_full53.toFixed(1)}%</div>
      `;
      row.addEventListener("mousemove", (evt) => {
        showTooltip(
          evt,
          `<strong>${it.name}</strong> (${it.motion_type.replace(/_/g, " ")})<br/>` +
            `Manual AUC: <strong>${it.manual_auc.toFixed(2)}</strong> &middot; Kinematic AUC: <strong>${it.kinematic_auc.toFixed(2)}</strong><br/>` +
            `Model reliability weight: <strong>${it.weight_full53.toFixed(2)}%</strong><br/>` +
            `<span style="color:#cbd5e1;">Hand: ${it.hand_posture}</span>`
        );
      });
      row.addEventListener("mouseleave", hideTooltip);
      const cb = row.querySelector("input");
      cb.addEventListener("change", () => {
        if (!cb.checked && STATE.activeItems.size <= 1) {
          cb.checked = true;
          return;
        }
        if (cb.checked) STATE.activeItems.add(it.id);
        else STATE.activeItems.delete(it.id);
        clearActivePresetHighlight();
        updateAll();
      });
      listWrap.appendChild(row);
    }
  }

  // ========================================================================
  // 6. SECTION 2: PARTICIPANT CONTRIBUTION INSPECTOR
  // ========================================================================

  function renderParticipantInspector(res) {
    const availPids = Object.keys(res.blendPred).sort((a, b) => res.trueByPid[a] - res.trueByPid[b]);
    if (!availPids.includes(STATE.selectedParticipant)) {
      const stAprax = availPids.find((p) => p.startsWith("ST") && res.trueByPid[p] < STATE.cutoff);
      STATE.selectedParticipant = stAprax || availPids[0];
    }
    const pid = STATE.selectedParticipant;
    const sel = document.getElementById("select-participant");
    sel.innerHTML = "";
    for (const p of availPids) {
      const tr = res.trueByPid[p];
      const grp = p.startsWith("CL")
        ? "Control"
        : tr < STATE.cutoff
        ? "Stroke: with apraxia"
        : "Stroke: without apraxia";
      const opt = document.createElement("option");
      opt.value = p;
      opt.selected = p === pid;
      opt.textContent = `${p} — ${grp} (True ${tr.toFixed(2)} | Pred ${res.blendPred[p].toFixed(2)})`;
      sel.appendChild(opt);
    }

    const pMeta = STATE.data.participants.find((x) => x.id === pid);
    const fo = res.foldOutputs[pid];
    const tr = res.trueByPid[pid];
    const bl = res.blendPred[pid];

    document.getElementById("participant-summary-pills").innerHTML = `
      <span class="part-stat">Manual Score: <strong>${tr.toFixed(3)}</strong> (HP ${pMeta.hp.toFixed(2)}, AP ${pMeta.ap.toFixed(2)}, AT ${pMeta.at.toFixed(2)})</span>
      <span class="part-stat">Participant Pathway: <strong>${res.partPred[pid].toFixed(3)}</strong></span>
      <span class="part-stat">Item Pathway: <strong>${res.itemPred[pid].toFixed(3)}</strong></span>
      <span class="part-stat">Blended Prediction: <strong>${bl.toFixed(3)}</strong></span>
    `;

    // Left plot: Top 16 signed feature contributions
    const featContribs = STATE.data.features
      .filter((f) => STATE.activeFeatures.has(f.index))
      .map((f) => ({
        name: f.name,
        family: f.family,
        color: f.color,
        val: fo.featContrib[f.index],
      }))
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 16)
      .sort((a, b) => a.val - b.val);

    const W = 540, rowH = 22;
    const H = 56 + featContribs.length * rowH;
    const pad = { l: 195, r: 24, t: 16, b: 34 };
    const pw = W - pad.l - pad.r;
    const vmax = Math.max(0.04, ...featContribs.map((d) => Math.abs(d.val))) * 1.15;
    const sx = (v) => pad.l + ((v + vmax) / (2 * vmax)) * pw;
    const xZero = sx(0);

    let fSvg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Participant signed feature contributions">`;
    fSvg += `<line x1="${xZero}" y1="${pad.t - 4}" x2="${xZero}" y2="${H - pad.b}" stroke="#333740" stroke-width="1.1"/>`;
    featContribs.forEach((d, idx) => {
      const yy = pad.t + idx * rowH;
      const x1 = Math.min(xZero, sx(d.val));
      const bw = Math.max(1.5, Math.abs(sx(d.val) - xZero));
      const barCol = d.val < 0 ? "#c93835" : "#1f7a4c";
      fSvg += `<line x1="${pad.l}" y1="${yy + 9}" x2="${W - pad.r}" y2="${yy + 9}" stroke="#f0ede6" stroke-width="1"/>`;
      fSvg += `<rect x="${x1.toFixed(1)}" y="${yy + 3}" width="${bw.toFixed(1)}" height="12" rx="2" fill="${barCol}" fill-opacity="0.85"/>`;
      fSvg += `<circle cx="${pad.l - 184}" cy="${yy + 9}" r="3.5" fill="${d.color}"/>`;
      fSvg += `<text x="${pad.l - 8}" y="${yy + 13}" text-anchor="end" font-size="10.5" fill="#181a1f">${d.name}</text>`;
    });
    for (const tv of [-vmax * 0.75, 0, vmax * 0.75]) {
      fSvg += `<text x="${sx(tv)}" y="${H - 14}" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="#494e59">${tv >= 0 ? "+" : ""}${tv.toFixed(3)}</text>`;
    }
    fSvg += `</svg>`;
    document.getElementById("inspector-features-svg").innerHTML = fSvg;

    // Right plot: Per-tool predictions vs manual score for this participant
    const itemById = new Map(STATE.data.items.map((it) => [it.id, it.name]));
    const byTool = new Map();
    for (const d of fo.itemDetail) {
      if (!byTool.has(d.item)) byTool.set(d.item, { preds: [], trues: [] });
      byTool.get(d.item).preds.push(d.pred);
      byTool.get(d.item).trues.push(d.trueScore);
    }
    const toolRows = Array.from(byTool.entries())
      .map(([iid, obj]) => {
        const pr = obj.preds.reduce((a, b) => a + b, 0) / obj.preds.length;
        const trS = obj.trues.reduce((a, b) => a + b, 0) / obj.trues.length;
        return { name: itemById.get(iid) || `Item ${iid}`, pred: pr, trueScore: trS, err: pr - trS };
      })
      .sort((a, b) => b.err - a.err)
      .slice(0, 16);

    const H2 = 56 + toolRows.length * rowH;
    const pad2 = { l: 130, r: 24, t: 16, b: 34 };
    const pw2 = W - pad2.l - pad2.r;
    const sx2 = (v) => pad2.l + (v / 3.0) * pw2;

    let iSvg = `<svg viewBox="0 0 ${W} ${H2}" role="img" aria-label="Per-tool predicted versus manual scores">`;
    for (const scv of [0, 1, 2, 3]) {
      const xx = sx2(scv);
      iSvg += `<line x1="${xx}" y1="${pad2.t - 4}" x2="${xx}" y2="${H2 - pad2.b}" stroke="#ece9e1" stroke-width="1"/>`;
      iSvg += `<text x="${xx}" y="${H2 - 14}" text-anchor="middle" font-size="10.5" font-family="var(--font-mono)" fill="#494e59">${scv}</text>`;
    }
    toolRows.forEach((d, idx) => {
      const yy = pad2.t + idx * rowH + 9;
      const xp = sx2(d.pred);
      const xt = sx2(d.trueScore);
      const lineCol = d.err > 0.25 ? "#c93835" : d.err < -0.25 ? "#1976d2" : "#8c8f99";
      iSvg += `<line x1="${pad2.l}" y1="${yy}" x2="${W - pad2.r}" y2="${yy}" stroke="#f3f1eb" stroke-width="1"/>`;
      iSvg += `<line x1="${xt.toFixed(1)}" y1="${yy}" x2="${xp.toFixed(1)}" y2="${yy}" stroke="${lineCol}" stroke-width="2"/>`;
      iSvg += `<circle cx="${xt.toFixed(1)}" cy="${yy}" r="4.5" fill="#ffffff" stroke="#181a1f" stroke-width="1.6"/>`;
      iSvg += `<circle cx="${xp.toFixed(1)}" cy="${yy}" r="4.6" fill="#7e6fb2" stroke="#181a1f" stroke-width="1"/>`;
      iSvg += `<text x="${pad2.l - 8}" y="${yy + 3.8}" text-anchor="end" font-size="10.5" fill="#181a1f">${d.name}</text>`;
    });
    iSvg += `</svg>`;
    document.getElementById("inspector-items-svg").innerHTML = iSvg;
  }

  // ========================================================================
  // 7. SECTION 3: INTERACTIVE 3D PANTOMIME TRAJECTORIES & SYNCHRONIZED PROFILES
  // ========================================================================

  const GROUP_COLORS = ["#0a1f4d", "#1976d2", "#d32f2f"];

  function project3D(pt, elevDeg, azimDeg, halfCube, cx, cy, scalePx) {
    const az = (azimDeg * Math.PI) / 180;
    const el = (elevDeg * Math.PI) / 180;
    const x = pt[0] / halfCube;
    const y = pt[1] / halfCube;
    const z = pt[2] / halfCube;

    // Match Matplotlib mplot3d azimuth/elevation convention
    const x1 = -x * Math.sin(az) + y * Math.cos(az);
    const y1 = -x * Math.cos(az) - y * Math.sin(az);
    const screenX = cx + x1 * scalePx;
    const screenY = cy - (z * Math.cos(el) + y1 * Math.sin(el)) * scalePx;
    return [screenX, screenY];
  }

  function build3DCards() {
    const wrap = document.getElementById("traces-3d-cards");
    const list = STATE.data.sample_traces[STATE.traceItem] || [];
    wrap.innerHTML = "";
    list.forEach((tr, idx) => {
      const col = GROUP_COLORS[idx] || "#333";
      const card = document.createElement("div");
      card.className = "trace-col-card";
      card.innerHTML = `
        <div class="trace-col-head">
          <strong style="color:${col}; font-size:0.9rem;">${tr.label} (${tr.participant})</strong>
          <span style="font-family:var(--font-mono); font-size:0.76rem; color:var(--text-secondary);">
            Score ${tr.score_total}/3 &middot; ${tr.duration.toFixed(1)} s
          </span>
        </div>
        <div style="font-size:0.73rem; color:var(--text-muted); margin-bottom:0.45rem;">
          HP: ${tr.score_hp} &middot; AP: ${tr.score_ap} &middot; AT: ${tr.score_at}
        </div>
        <canvas class="trace-canvas" width="360" height="250" data-cidx="${idx}"></canvas>
      `;
      wrap.appendChild(card);

      const canvas = card.querySelector("canvas");
      let dragging = false;
      let lastX = 0, lastY = 0;
      canvas.addEventListener("pointerdown", (e) => {
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        STATE.rotAzim += (e.clientX - lastX) * 0.65;
        STATE.rotElev = clip(STATE.rotElev + (e.clientY - lastY) * 0.5, -65, 75);
        lastX = e.clientX;
        lastY = e.clientY;
        draw3DCanvases();
      });
      canvas.addEventListener("pointerup", () => {
        dragging = false;
      });
    });
    draw3DCanvases();
    renderTracesTimeseries();
  }

  function draw3DCanvases() {
    const list = STATE.data.sample_traces[STATE.traceItem] || [];
    let halfCube = 8.0;
    for (const tr of list) {
      for (const stream of [tr.wrist_xyz, tr.index_xyz, tr.thumb_xyz]) {
        for (const p of stream) {
          halfCube = Math.max(halfCube, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
        }
      }
    }
    halfCube *= 1.08;

    const canvases = document.querySelectorAll("#traces-3d-cards canvas");
    canvases.forEach((canvas, idx) => {
      const tr = list[idx];
      if (!tr) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth || 360;
      const cssH = canvas.clientHeight || 250;
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const cx = cssW * 0.5;
      const cy = cssH * 0.54;
      const sc = Math.min(cssW, cssH) * 0.34;

      // Draw 3D bounding axes (-halfCube .. +halfCube)
      const h = halfCube;
      const axisEdges = [
        [[-h, -h, -h], [h, -h, -h]],
        [[h, -h, -h], [h, h, -h]],
        [[h, h, -h], [h, h, h]],
      ];
      ctx.strokeStyle = "#333740";
      ctx.lineWidth = 1.25;
      for (const [p1, p2] of axisEdges) {
        const [x1, y1] = project3D(p1, STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
        const [x2, y2] = project3D(p2, STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      const nFrames = tr.wrist_xyz.length;
      const maxFrame = Math.max(1, Math.round((STATE.traceTimePct / 100) * (nFrames - 1)));
      const baseHex = GROUP_COLORS[idx] || "#1976d2";

      const streamsToDraw = [];
      if (STATE.showWrist) streamsToDraw.push({ pts: tr.wrist_xyz, color: baseHex, lw: 2.8, dash: [] });
      if (STATE.showIndex) streamsToDraw.push({ pts: tr.index_xyz, color: "#882255", lw: 1.8, dash: [] });
      if (STATE.showThumb) streamsToDraw.push({ pts: tr.thumb_xyz, color: "#237c6d", lw: 1.8, dash: [3, 2] });

      for (const st of streamsToDraw) {
        ctx.setLineDash(st.dash);
        for (let i = 1; i <= maxFrame; i++) {
          const [x1, y1] = project3D(st.pts[i - 1], STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
          const [x2, y2] = project3D(st.pts[i], STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
          const alpha = 0.25 + 0.75 * (i / Math.max(1, nFrames - 1));
          ctx.strokeStyle = st.color;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = st.lw;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
        ctx.setLineDash([]);

        // Start ring & current timehead dot
        const [sx0, sy0] = project3D(st.pts[0], STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = st.color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(sx0, sy0, 4.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        const [cxNow, cyNow] = project3D(st.pts[maxFrame], STATE.rotElev, STATE.rotAzim, h, cx, cy, sc);
        ctx.fillStyle = st.color;
        ctx.beginPath();
        ctx.arc(cxNow, cyNow, 4.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  function renderTracesTimeseries() {
    const list = STATE.data.sample_traces[STATE.traceItem] || [];
    if (list.length === 0) return;

    const W = 1140, H = 265;
    const rows = [
      { key: "aperture", label: "Thumb–Index Aperture (cm)" },
      { key: "elbow_angle", label: "Elbow Angle (°)" },
      { key: "wrist_speed", label: "Wrist Speed (cm/s)" },
    ];
    const colGap = 24;
    const padL = 62, padR = 16, padT = 12, padB = 26;
    const colW = (W - padL - padR - 2 * colGap) / 3;
    const rowGap = 14;
    const rowH = (H - padT - padB - 2 * rowGap) / 3;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Synchronized kinematic time series">`;
    rows.forEach((rm, rIdx) => {
      let ymin = Infinity, ymax = -Infinity;
      for (const tr of list) {
        for (const v of tr[rm.key]) {
          if (v < ymin) ymin = v;
          if (v > ymax) ymax = v;
        }
      }
      if (rm.key === "wrist_speed") ymin = 0;
      const span = Math.max(1e-3, ymax - ymin);
      ymin -= 0.05 * span;
      ymax += 0.06 * span;
      const yTop = padT + rIdx * (rowH + rowGap);

      list.forEach((tr, cIdx) => {
        const xLeft = padL + cIdx * (colW + colGap);
        const col = GROUP_COLORS[cIdx] || "#333";
        svg += `<rect x="${xLeft}" y="${yTop}" width="${colW}" height="${rowH}" fill="#fbfaf7" stroke="#e4e1d9"/>`;

        const pts = tr.tnorm
          .map((tn, i) => {
            const x = xLeft + (tn / 100) * colW;
            const y = yTop + (1 - (tr[rm.key][i] - ymin) / (ymax - ymin)) * rowH;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(" ");
        svg += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;

        // Playhead cursor
        const xCur = xLeft + (STATE.traceTimePct / 100) * colW;
        svg += `<line class="trace-playhead" data-col="${cIdx}" data-xleft="${xLeft}" data-colw="${colW}" x1="${xCur.toFixed(1)}" y1="${yTop}" x2="${xCur.toFixed(1)}" y2="${yTop + rowH}" stroke="#181a1f" stroke-width="1.2" stroke-dasharray="3,2"/>`;

        if (cIdx === 0) {
          svg += `<text x="${xLeft - 6}" y="${yTop + 10}" text-anchor="end" font-size="9.5" font-family="var(--font-mono)" fill="#494e59">${ymax.toFixed(0)}</text>`;
          svg += `<text x="${xLeft - 6}" y="${yTop + rowH}" text-anchor="end" font-size="9.5" font-family="var(--font-mono)" fill="#494e59">${Math.max(0, ymin).toFixed(0)}</text>`;
          svg += `<text x="14" y="${yTop + rowH / 2}" text-anchor="middle" font-size="10" font-weight="600" fill="#181a1f" transform="rotate(-90 14 ${yTop + rowH / 2})">${rm.label}</text>`;
        }
        if (rIdx === 2) {
          svg += `<text x="${xLeft + colW / 2}" y="${H - 6}" text-anchor="middle" font-size="10.5" fill="#494e59">Normalized Movement Time (0–100%)</text>`;
        }
      });
    });
    svg += `</svg>`;
    document.getElementById("traces-timeseries-svg").innerHTML = svg;
  }

  function updateTracePlayheadsOnly() {
    document.querySelectorAll(".trace-playhead").forEach((ln) => {
      const xLeft = parseFloat(ln.getAttribute("data-xleft"));
      const colW = parseFloat(ln.getAttribute("data-colw"));
      const xCur = (xLeft + (STATE.traceTimePct / 100) * colW).toFixed(1);
      ln.setAttribute("x1", xCur);
      ln.setAttribute("x2", xCur);
    });
  }

  // ========================================================================
  // 8. SECTION 4: 39-TOOL BATTERY ATLAS (SCATTER + SORTABLE TABLE)
  // ========================================================================

  function renderAtlas() {
    const items = STATE.data.items;
    const xKey = STATE.atlasXMode;
    const xLabel = xKey === "manual_auc" ? "Manual Score Apraxia AUC" : "Cross-Validated Kinematic AUC";

    // Scatter plot
    const W = 400, H = 340;
    const pad = { l: 48, r: 18, t: 18, b: 44 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;
    const xmin = 0.45, xmax = 0.90;
    const ymax = Math.max(7.0, Math.ceil(Math.max(...items.map((d) => d.weight_full53)) + 0.5));
    const sx = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * pw;
    const sy = (v) => pad.t + (1 - v / ymax) * ph;

    const rCorr = pearsonR(
      items.map((d) => d[xKey]),
      items.map((d) => d.weight_full53)
    );

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Item reliability weight versus apraxia AUC">`;
    for (const xv of [0.5, 0.6, 0.7, 0.8, 0.9]) {
      const xx = sx(xv);
      svg += `<line x1="${xx}" y1="${pad.t}" x2="${xx}" y2="${pad.t + ph}" stroke="#ece9e1"/>`;
      svg += `<text x="${xx}" y="${pad.t + ph + 16}" text-anchor="middle" font-size="10.5" fill="#494e59">${xv.toFixed(1)}</text>`;
    }
    for (let yv = 0; yv <= ymax; yv += 2) {
      const yy = sy(yv);
      svg += `<line x1="${pad.l}" y1="${yy}" x2="${pad.l + pw}" y2="${yy}" stroke="#ece9e1"/>`;
      svg += `<text x="${pad.l - 7}" y="${yy + 3.5}" text-anchor="end" font-size="10.5" fill="#494e59">${yv}%</text>`;
    }
    svg += `<line x1="${sx(0.5)}" y1="${pad.t}" x2="${sx(0.5)}" y2="${pad.t + ph}" stroke="#8c8f99" stroke-dasharray="4,3"/>`;
    svg += `<rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" fill="none" stroke="#333740"/>`;
    svg += `<text x="${pad.l + pw / 2}" y="${H - 8}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#181a1f">${xLabel}</text>`;
    svg += `<text x="14" y="${pad.t + ph / 2}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#181a1f" transform="rotate(-90 14 ${pad.t + ph / 2})">Learned Reliability Weight (%)</text>`;
    svg += `<text x="${pad.l + 8}" y="${pad.t + 16}" font-size="10.5" font-family="var(--font-mono)" fill="#494e59">r = ${rCorr.toFixed(2)}</text>`;

    for (const it of items) {
      const cx = sx(clip(it[xKey], xmin, xmax));
      const cy = sy(it.weight_full53);
      const active = STATE.activeItems.has(it.id);
      svg += `<circle class="atlas-dot" data-iid="${it.id}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5.2" fill="${active ? "#7e6fb2" : "#cbd5e1"}" stroke="#181a1f" stroke-width="0.9" style="cursor:pointer;"/>`;
    }
    svg += `</svg>`;
    const scWrap = document.getElementById("atlas-scatter-svg");
    scWrap.innerHTML = svg;

    scWrap.querySelectorAll(".atlas-dot").forEach((el) => {
      const iid = Number(el.getAttribute("data-iid"));
      const it = items.find((x) => x.id === iid);
      el.addEventListener("mousemove", (evt) => {
        showTooltip(
          evt,
          `<strong>${it.name}</strong><br/>` +
            `Manual AUC: <strong>${it.manual_auc.toFixed(2)}</strong> &middot; Kinematic AUC: <strong>${it.kinematic_auc.toFixed(2)}</strong><br/>` +
            `Full-cohort reliability weight: <strong>${it.weight_full53.toFixed(2)}%</strong><br/>` +
            `Mean score (with apraxia): <strong>${it.mean_apraxic.toFixed(2)} / 3</strong>`
        );
      });
      el.addEventListener("mouseleave", hideTooltip);
    });

    // Table
    const q = STATE.atlasSearch.trim().toLowerCase();
    const filtered = items
      .filter(
        (it) =>
          !q ||
          it.name.toLowerCase().includes(q) ||
          it.motion_type.toLowerCase().includes(q) ||
          it.hand_posture.toLowerCase().includes(q) ||
          it.key_motion.toLowerCase().includes(q)
      )
      .sort((a, b) => {
        const va = a[STATE.atlasSortKey];
        const vb = b[STATE.atlasSortKey];
        if (typeof va === "string") return STATE.atlasSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return STATE.atlasSortAsc ? va - vb : vb - va;
      });

    const tbody = document.getElementById("atlas-tbody");
    tbody.innerHTML = filtered
      .map(
        (it) => `
      <tr>
        <td><strong>${it.name}</strong></td>
        <td style="color:var(--text-secondary);">${it.motion_type.replace(/_/g, " ")}</td>
        <td class="num-col">${it.manual_auc.toFixed(2)}</td>
        <td class="num-col">${it.kinematic_auc.toFixed(2)}</td>
        <td class="num-col">${it.weight_full53.toFixed(2)}%</td>
        <td class="num-col">${it.mean_apraxic !== null ? it.mean_apraxic.toFixed(2) : "—"}</td>
        <td style="font-size:0.76rem; color:var(--text-secondary);">
          <strong>Hand:</strong> ${it.hand_posture}; <strong>Arm:</strong> ${it.arm_posture}
        </td>
      </tr>`
      )
      .join("");
  }

  // ========================================================================
  // 9. MASTER UPDATE & EVENT WIRING
  // ========================================================================

  function clearActivePresetHighlight() {
    document.querySelectorAll(".preset-bar .preset-chip").forEach((b) => b.classList.remove("active"));
  }

  function applyPreset(presetName) {
    STATE.blendItemWeight = 0.5;
    STATE.alpha = 10.0;
    STATE.cutoff = 2.52;
    document.getElementById("slider-blend").value = "50";
    document.getElementById("slider-alpha").value = "3";
    document.getElementById("slider-cutoff").value = "2.52";

    STATE.activeFeatures = new Set();
    STATE.activeItems = new Set(STATE.data.items.map((it) => it.id));

    if (presetName === "full") {
      STATE.data.features.forEach((f) => STATE.activeFeatures.add(f.index));
    } else if (presetName === "top8") {
      STATE.data.features.filter((f) => f.in_top8_train).forEach((f) => STATE.activeFeatures.add(f.index));
    } else if (presetName === "screening") {
      STATE.data.features.filter((f) => f.in_top8_train).forEach((f) => STATE.activeFeatures.add(f.index));
      STATE.activeItems = new Set(
        STATE.data.items.filter((it) => COMPACT_12_ITEM_NAMES.has(it.name)).map((it) => it.id)
      );
    } else if (presetName === "hand_only") {
      STATE.data.features.filter((f) => f.family === "Hand posture").forEach((f) => STATE.activeFeatures.add(f.index));
    } else if (presetName === "arm_only") {
      STATE.data.features.filter((f) => f.family === "Arm posture & trajectory").forEach((f) => STATE.activeFeatures.add(f.index));
    } else if (presetName === "amp_only") {
      STATE.data.features.filter((f) => f.family === "Amplitude & timing").forEach((f) => STATE.activeFeatures.add(f.index));
    } else if (presetName === "no_hand") {
      STATE.data.features.filter((f) => f.family !== "Hand posture").forEach((f) => STATE.activeFeatures.add(f.index));
    }

    document.querySelectorAll(".preset-bar .preset-chip").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-preset") === presetName);
    });
    updateAll();
  }

  let _pendingRaf = 0;
  function updateAll() {
    const partPct = Math.round((1 - STATE.blendItemWeight) * 100);
    const itemPct = Math.round(STATE.blendItemWeight * 100);
    document.getElementById("val-blend").textContent = `${partPct}% Part. / ${itemPct}% Item`;
    document.getElementById("val-alpha").textContent = `\u03b1 = ${STATE.alpha}`;
    document.getElementById("val-cutoff").textContent = STATE.cutoff.toFixed(2);
    document.getElementById("active-split-caption").textContent =
      STATE.split === "full53"
        ? "53 LOPO folds"
        : STATE.split === "train41"
        ? "41 LOPO folds"
        : "Trained on 41 → 12 unseen";

    const featIdx = Array.from(STATE.activeFeatures).sort((a, b) => a - b);
    const itemArr = Array.from(STATE.activeItems).sort((a, b) => a - b);
    const key = `${STATE.split}|${STATE.alpha}|${featIdx.join(",")}|${itemArr.join(",")}`;
    const isCached = STATE.pathwayCache.has(key);

    const runRender = () => {
      const res = evaluateActivePipeline();
      STATE.lastResult = res;

      const mPart = computeMetrics(res.partPred, res.trueByPid, STATE.cutoff);
      const mItem = computeMetrics(res.itemPred, res.trueByPid, STATE.cutoff);
      const mBlend = computeMetrics(res.blendPred, res.trueByPid, STATE.cutoff);
      const base = STATE.baselineMetrics[STATE.split];

      document.getElementById("kpi-r").textContent = mBlend.r.toFixed(3);
      document.getElementById("sub-r").textContent = `Spearman \u03c1 = ${mBlend.rho.toFixed(3)}`;
      formatDeltaBadge(document.getElementById("delta-r"), mBlend.r, base.r, true, 3);

      document.getElementById("kpi-auc").textContent = mBlend.auc.toFixed(3);
      document.getElementById("sub-auc").textContent = `${mBlend.tp + mBlend.fn + mBlend.tn + mBlend.fp} stroke patients`;
      formatDeltaBadge(document.getElementById("delta-auc"), mBlend.auc, base.auc, true, 3);

      document.getElementById("kpi-mae").textContent = mBlend.mae.toFixed(3);
      document.getElementById("sub-mae").textContent = `RMSE = ${mBlend.rmse.toFixed(3)}`;
      formatDeltaBadge(document.getElementById("delta-mae"), mBlend.mae, base.mae, false, 3);

      document.getElementById("kpi-sens").textContent = mBlend.sens.toFixed(2);
      document.getElementById("sub-sens").textContent = `${mBlend.tp}/${mBlend.tp + mBlend.fn} with apraxia`;
      formatDeltaBadge(document.getElementById("delta-sens"), mBlend.sens, base.sens, true, 2);

      document.getElementById("kpi-spec").textContent = mBlend.spec.toFixed(2);
      document.getElementById("sub-spec").textContent = `${mBlend.tn}/${mBlend.tn + mBlend.fp} without apraxia`;
      formatDeltaBadge(document.getElementById("delta-spec"), mBlend.spec, base.spec, true, 2);

      document.getElementById("compute-status").textContent =
        `${STATE.activeFeatures.size}/46 features · ${STATE.activeItems.size}/39 tools · computed in ${res.elapsedMs.toFixed(1)} ms`;

      renderScatter(res);
      renderPathwayBars(mPart, mItem, mBlend);
      renderConfusionAndRoc(mBlend);
      renderFamilyStackedBar(res);
      renderFeatureControls(res);
      renderItemControls();
      renderParticipantInspector(res);
      renderAtlas();
    };

    if (isCached) {
      if (_pendingRaf) cancelAnimationFrame(_pendingRaf);
      runRender();
    } else {
      document.getElementById("compute-status").textContent = "Computing cross-validated Ridge folds…";
      if (_pendingRaf) cancelAnimationFrame(_pendingRaf);
      _pendingRaf = requestAnimationFrame(() => {
        _pendingRaf = requestAnimationFrame(runRender);
      });
    }
  }

  function wireEvents() {
    // Presets
    document.querySelectorAll(".preset-bar .preset-chip").forEach((btn) => {
      btn.addEventListener("click", () => applyPreset(btn.getAttribute("data-preset")));
    });
    document.getElementById("btn-reset-model").addEventListener("click", () => applyPreset("full"));

    // Split buttons
    document.querySelectorAll("[data-split]").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.split = btn.getAttribute("data-split");
        document.querySelectorAll("[data-split]").forEach((b) => b.classList.toggle("active", b === btn));
        updateAll();
      });
    });

    // Sliders
    document.getElementById("slider-blend").addEventListener("input", (e) => {
      STATE.blendItemWeight = Number(e.target.value) / 100.0;
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("slider-alpha").addEventListener("input", (e) => {
      STATE.alpha = ALPHA_STEPS[Number(e.target.value)] || 10.0;
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("slider-cutoff").addEventListener("input", (e) => {
      STATE.cutoff = Number(e.target.value);
      clearActivePresetHighlight();
      updateAll();
    });

    // Selector Tabs (Features vs Tool Battery)
    const tabFeat = document.getElementById("tab-btn-features");
    const tabItem = document.getElementById("tab-btn-items");
    tabFeat.addEventListener("click", () => {
      tabFeat.classList.add("active");
      tabItem.classList.remove("active");
      document.getElementById("pane-features").style.display = "";
      document.getElementById("pane-items").style.display = "none";
    });
    tabItem.addEventListener("click", () => {
      tabItem.classList.add("active");
      tabFeat.classList.remove("active");
      document.getElementById("pane-items").style.display = "";
      document.getElementById("pane-features").style.display = "none";
    });

    document.getElementById("search-features").addEventListener("input", (e) => {
      STATE.featureSearch = e.target.value;
      if (STATE.lastResult) renderFeatureControls(STATE.lastResult);
    });
    document.getElementById("search-items").addEventListener("input", (e) => {
      STATE.itemSearch = e.target.value;
      renderItemControls();
    });

    document.getElementById("btn-feat-all").addEventListener("click", () => {
      STATE.data.features.forEach((f) => STATE.activeFeatures.add(f.index));
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("btn-feat-top8").addEventListener("click", () => {
      STATE.activeFeatures = new Set(STATE.data.features.filter((f) => f.in_top8_train).map((f) => f.index));
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("btn-item-all").addEventListener("click", () => {
      STATE.activeItems = new Set(STATE.data.items.map((it) => it.id));
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("btn-item-top12").addEventListener("click", () => {
      STATE.activeItems = new Set(
        STATE.data.items.filter((it) => COMPACT_12_ITEM_NAMES.has(it.name)).map((it) => it.id)
      );
      clearActivePresetHighlight();
      updateAll();
    });
    document.getElementById("btn-item-prune6").addEventListener("click", () => {
      STATE.activeItems = new Set(
        STATE.data.items.filter((it) => !LOWEST_6_ITEM_NAMES.has(it.name)).map((it) => it.id)
      );
      clearActivePresetHighlight();
      updateAll();
    });

    // Participant dropdown
    document.getElementById("select-participant").addEventListener("change", (e) => {
      STATE.selectedParticipant = e.target.value;
      if (STATE.lastResult) {
        renderScatter(STATE.lastResult);
        renderParticipantInspector(STATE.lastResult);
      }
    });

    // 3D Traces controls
    document.querySelectorAll("[data-trace-item]").forEach((btn) => {
      btn.addEventListener("click", () => {
        STATE.traceItem = btn.getAttribute("data-trace-item");
        document.querySelectorAll("[data-trace-item]").forEach((b) => b.classList.toggle("active", b === btn));
        build3DCards();
      });
    });
    document.getElementById("chk-trace-wrist").addEventListener("change", (e) => {
      if (!e.target.checked && !STATE.showIndex && !STATE.showThumb) {
        e.target.checked = true;
        return;
      }
      STATE.showWrist = e.target.checked;
      draw3DCanvases();
    });
    document.getElementById("chk-trace-index").addEventListener("change", (e) => {
      STATE.showIndex = e.target.checked;
      draw3DCanvases();
    });
    document.getElementById("chk-trace-thumb").addEventListener("change", (e) => {
      STATE.showThumb = e.target.checked;
      draw3DCanvases();
    });
    document.getElementById("btn-reset-3d").addEventListener("click", () => {
      STATE.rotElev = 18;
      STATE.rotAzim = -62;
      draw3DCanvases();
    });
    const traceSlider = document.getElementById("slider-trace-time");
    traceSlider.addEventListener("input", (e) => {
      STATE.traceTimePct = Number(e.target.value);
      document.getElementById("val-trace-time").textContent = `${Math.round(STATE.traceTimePct)}%`;
      draw3DCanvases();
      updateTracePlayheadsOnly();
    });

    const playBtn = document.getElementById("btn-play-trace");
    playBtn.addEventListener("click", () => {
      STATE.playingTrace = !STATE.playingTrace;
      playBtn.innerHTML = STATE.playingTrace ? "&#10074;&#10074; Pause" : "&#9654; Play";
      if (STATE.playingTrace) {
        if (STATE.traceTimePct >= 99.5) STATE.traceTimePct = 0;
        const stepAnim = () => {
          if (!STATE.playingTrace) return;
          STATE.traceTimePct = Math.min(100, STATE.traceTimePct + 0.85);
          traceSlider.value = String(STATE.traceTimePct);
          document.getElementById("val-trace-time").textContent = `${Math.round(STATE.traceTimePct)}%`;
          draw3DCanvases();
          updateTracePlayheadsOnly();
          if (STATE.traceTimePct >= 100) {
            STATE.playingTrace = false;
            playBtn.innerHTML = "&#9654; Play";
            return;
          }
          requestAnimationFrame(stepAnim);
        };
        requestAnimationFrame(stepAnim);
      }
    });

    // Atlas controls
    const btnManualAuc = document.getElementById("btn-atlas-manual-auc");
    const btnKinAuc = document.getElementById("btn-atlas-kin-auc");
    btnManualAuc.addEventListener("click", () => {
      STATE.atlasXMode = "manual_auc";
      btnManualAuc.classList.add("btn-primary");
      btnKinAuc.classList.remove("btn-primary");
      renderAtlas();
    });
    btnKinAuc.addEventListener("click", () => {
      STATE.atlasXMode = "kinematic_auc";
      btnKinAuc.classList.add("btn-primary");
      btnManualAuc.classList.remove("btn-primary");
      renderAtlas();
    });
    document.getElementById("search-atlas").addEventListener("input", (e) => {
      STATE.atlasSearch = e.target.value;
      renderAtlas();
    });
    document.querySelectorAll("#atlas-table th[data-sort]").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.getAttribute("data-sort");
        if (STATE.atlasSortKey === key) STATE.atlasSortAsc = !STATE.atlasSortAsc;
        else {
          STATE.atlasSortKey = key;
          STATE.atlasSortAsc = key === "name" || key === "motion_type";
        }
        renderAtlas();
      });
    });

    // Manuscript Figure switcher (supports both docs/figures/ on GitHub Pages and ../figures/ locally)
    const figImg = document.getElementById("manuscript-fig-img");
    function setFigSrc(num) {
      figImg.onerror = () => {
        figImg.onerror = null;
        figImg.src = `../figures/fig${num}.png`;
      };
      figImg.src = `figures/fig${num}.png`;
      figImg.alt = `Manuscript Figure ${num}`;
    }
    setFigSrc(1);

    document.querySelectorAll("[data-fig]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const num = btn.getAttribute("data-fig");
        document.querySelectorAll("[data-fig]").forEach((b) => b.classList.toggle("active", b === btn));
        setFigSrc(num);
      });
    });

    document.getElementById("btn-copy-cmd").addEventListener("click", (e) => {
      const txt = document.getElementById("repro-commands").textContent;
      navigator.clipboard.writeText(txt).then(() => {
        e.target.textContent = "Copied!";
        setTimeout(() => {
          e.target.textContent = "Copy Commands";
        }, 1600);
      });
    });
  }

  async function init() {
    try {
      const resp = await fetch("data/explorer_data.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      STATE.data = await resp.json();

      STATE.data.features.forEach((f) => STATE.activeFeatures.add(f.index));
      STATE.data.items.forEach((it) => STATE.activeItems.add(it.id));

      // Populate baseline reference metrics (< 1 ms) from precomputed manuscript fits
      const baseSpecs = [
        ["full53", "full_lopo"],
        ["train41", "train_lopo"],
        ["test12", "heldout_test"],
      ];
      for (const [spKey, field] of baseSpecs) {
        const predMap = {};
        const trueMap = {};
        for (const p of STATE.data.participants) {
          if (p[field]) {
            predMap[p.id] = p[field].blend;
            trueMap[p.id] = p.true_score;
          }
        }
        STATE.baselineMetrics[spKey] = computeMetrics(predMap, trueMap, 2.52);
      }
      STATE.split = "full53";

      wireEvents();
      build3DCards();
      updateAll();
    } catch (err) {
      console.error(err);
      document.getElementById("compute-status").textContent = "Error loading dataset (serve via HTTP or GitHub Pages)";
    }
  }

  // Export testable math engine for automated Node verification
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { fitRidge, fitStandardScaler, fitBlendFold, predictFoldParticipants, combineWithCalibration, computeMetrics };
  } else {
    init();
  }
})();
