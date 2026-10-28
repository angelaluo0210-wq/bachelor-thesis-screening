#!/usr/bin/env node
/*
 * Reproducibly render the Chapter 4 precision--recall curve from the frozen
 * Final Test predictions. This script reads stored labels and probabilities
 * only; it never loads a model, performs inference, or changes any artifact.
 */

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PREDICTIONS = path.join(ROOT, "outputs", "final_test", "final_test_predictions.csv");
const CONFIG = path.join(ROOT, "outputs", "final_test", "final_test_config.json");
const METRICS = path.join(ROOT, "outputs", "final_test", "final_test_metrics.json");
const OUTPUT = path.join(ROOT, "thesis", "img", "final_test_pr_curve.pdf");

const EXPECTED = {
  predictionSha256: "1ccdbb13d577e5369da4f21e61138ef052f49e91127f32674f6800e9710af876",
  rows: 23592,
  positives: 472,
  negatives: 23120,
  threshold: 0.016082723,
  ap: 0.1553127754556926,
  tp: 470,
  fn: 2,
  tn: 5142,
  fp: 17978,
};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  const header = rows.shift();
  return rows.filter((r) => r.length === header.length).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

function averagePrecisionAndCurve(observations) {
  const sorted = [...observations].sort((a, b) => b.score - a.score);
  const positives = observations.reduce((sum, item) => sum + item.label, 0);
  let tp = 0, fp = 0, ap = 0, previousRecall = 0;
  const points = [[0, 1]];
  for (let index = 0; index < sorted.length;) {
    const score = sorted[index].score;
    let groupTp = 0, groupFp = 0;
    while (index < sorted.length && sorted[index].score === score) {
      if (sorted[index].label) groupTp += 1; else groupFp += 1;
      index += 1;
    }
    tp += groupTp; fp += groupFp;
    const recall = tp / positives;
    const precision = tp / (tp + fp);
    ap += (recall - previousRecall) * precision;
    previousRecall = recall;
    points.push([recall, precision]);
  }
  return { ap, points };
}

function thresholdCounts(observations, threshold) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const item of observations) {
    const predicted = item.score >= threshold;
    if (predicted && item.label) tp += 1;
    else if (predicted) fp += 1;
    else if (item.label) fn += 1;
    else tn += 1;
  }
  return { tp, tn, fp, fn };
}

function pdfEscape(value) {
  return String(value).replace(/[^\x20-\x7E]/g, "?").replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}

function makePdf(width, height, content) {
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>`,
    `<< /Length ${Buffer.byteLength(content, "ascii")} >>\nstream\n${content}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
  ];
  let output = "%PDF-1.4\n%????\n";
  const offsets = [0];
  objects.forEach((object, i) => {
    offsets.push(Buffer.byteLength(output, "ascii"));
    output += `${i + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xref = Buffer.byteLength(output, "ascii");
  output += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i < offsets.length; i += 1) output += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  output += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(output, "ascii");
}

function renderPdf(points, ap, prevalence, operatingPoint) {
  const width = 520, height = 360;
  const x0 = 76, x1 = 480, y0 = 62, y1 = 300;
  const x = (recall) => x0 + recall * (x1 - x0);
  const y = (precision) => y0 + precision * (y1 - y0);
  const c = [];
  const text = (font, size, px, py, value) => c.push(`BT /${font} ${size} Tf ${px} ${py} Td (${pdfEscape(value)}) Tj ET`);
  c.push("1 J 1 j");
  c.push("0.96 0.96 0.96 rg 0 0 520 360 re f");
  c.push("1 1 1 rg 56 42 440 278 re f");
  c.push("0.78 0.78 0.78 RG 0.45 w");
  for (let tick = 0; tick <= 1.001; tick += 0.2) {
    const px = x(tick), py = y(tick);
    c.push(`${px.toFixed(2)} ${y0} m ${px.toFixed(2)} ${y1} l S`);
    c.push(`${x0} ${py.toFixed(2)} m ${x1} ${py.toFixed(2)} l S`);
    text("F1", 9, px - 7, 45, tick.toFixed(1));
    text("F1", 9, 39, py - 3, tick.toFixed(1));
  }
  c.push("0.15 0.15 0.15 RG 1.0 w");
  c.push(`${x0} ${y0} m ${x1} ${y0} l S ${x0} ${y0} m ${x0} ${y1} l S`);
  c.push("0.35 0.35 0.35 RG [3 3] 0 d 0.65 w");
  c.push(`${x0} ${y(prevalence).toFixed(2)} m ${x1} ${y(prevalence).toFixed(2)} l S [] 0 d`);
  c.push("0.12 0.30 0.52 RG 1.15 w");
  c.push(`${x(points[0][0]).toFixed(2)} ${y(points[0][1]).toFixed(2)} m`);
  for (let i = 1; i < points.length; i += 1) c.push(`${x(points[i][0]).toFixed(2)} ${y(points[i][1]).toFixed(2)} l`);
  c.push("S");
  c.push("0.79 0.22 0.16 rg 0.79 0.22 0.16 RG");
  const ox = x(operatingPoint.recall), oy = y(operatingPoint.precision), radius = 3.8, k = 0.55228475 * 3.8;
  c.push(`${(ox + radius).toFixed(2)} ${oy.toFixed(2)} m ${(ox + radius).toFixed(2)} ${(oy + k).toFixed(2)} ${(ox + k).toFixed(2)} ${(oy + radius).toFixed(2)} ${ox.toFixed(2)} ${(oy + radius).toFixed(2)} c ${(ox - k).toFixed(2)} ${(oy + radius).toFixed(2)} ${(ox - radius).toFixed(2)} ${(oy + k).toFixed(2)} ${(ox - radius).toFixed(2)} ${oy.toFixed(2)} c ${(ox - radius).toFixed(2)} ${(oy - k).toFixed(2)} ${(ox - k).toFixed(2)} ${(oy - radius).toFixed(2)} ${ox.toFixed(2)} ${(oy - radius).toFixed(2)} c ${(ox + k).toFixed(2)} ${(oy - radius).toFixed(2)} ${(ox + radius).toFixed(2)} ${(oy - k).toFixed(2)} ${(ox + radius).toFixed(2)} ${oy.toFixed(2)} c f`);
  c.push("0.79 0.22 0.16 RG 0.75 w");
  c.push(`${x(operatingPoint.recall).toFixed(2)} ${y(operatingPoint.precision).toFixed(2)} m 362 117 l S`);
  text("F2", 10, 94, 328, `Average Precision (AP) = ${ap.toFixed(4)}`);
  text("F1", 9, 95, y(prevalence) + 7, `Prevalence / no-skill = ${(prevalence * 100).toFixed(2)}%`);
  text("F2", 9, 365, 121, "Frozen operating point");
  text("F1", 9, 365, 108, `Recall = ${(operatingPoint.recall * 100).toFixed(2)}%`);
  text("F1", 9, 365, 96, `Precision = ${(operatingPoint.precision * 100).toFixed(2)}%`);
  text("F1", 11, 238, 20, "Recall");
  c.push("q 0 1 -1 0 17 151 cm"); text("F1", 11, 0, 0, "Precision"); c.push("Q");
  return makePdf(width, height, c.join("\n"));
}

function main() {
  const config = JSON.parse(fs.readFileSync(CONFIG, "utf8"));
  const metrics = JSON.parse(fs.readFileSync(METRICS, "utf8"));
  const bytes = fs.readFileSync(PREDICTIONS);
  const hash = crypto.createHash("sha256").update(bytes).digest("hex");
  assert(hash === EXPECTED.predictionSha256, `Unexpected prediction SHA-256: ${hash}`);
  assert(config.training_strategy === "lora" && config.adapter_learning_rate === 5e-5 && config.pooling === "cls" && config.best_epoch === 4 && config.seed === 42, "Final Test config does not identify the selected CLS + LoRA run.");
  assert(config.threshold === EXPECTED.threshold && config.selection_data === "validation_only" && config.test_was_evaluation_only, "Final Test config does not preserve validation-only threshold selection.");
  const rows = parseCsv(bytes.toString("utf8"));
  const observations = rows.map((row) => ({ label: Number(row.true_label), score: Number(row.predicted_probability) }));
  assert(observations.every((item) => (item.label === 0 || item.label === 1) && Number.isFinite(item.score)), "Predictions lack binary labels or continuous probabilities.");
  const positives = observations.reduce((sum, item) => sum + item.label, 0);
  assert(observations.length === EXPECTED.rows && positives === EXPECTED.positives && observations.length - positives === EXPECTED.negatives, "Final Test rows or class counts differ from canonical evidence.");
  const { ap, points } = averagePrecisionAndCurve(observations);
  assert(Math.abs(ap - EXPECTED.ap) < 1e-12 && Math.abs(ap - metrics.threshold_independent.pr_auc) < 1e-12, `AP consistency check failed: ${ap}`);
  const counts = thresholdCounts(observations, EXPECTED.threshold);
  assert(JSON.stringify(counts) === JSON.stringify({ tp: EXPECTED.tp, tn: EXPECTED.tn, fp: EXPECTED.fp, fn: EXPECTED.fn }), `Operating-point consistency check failed: ${JSON.stringify(counts)}`);
  const operatingPoint = { recall: counts.tp / (counts.tp + counts.fn), precision: counts.tp / (counts.tp + counts.fp) };
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, renderPdf(points, ap, positives / observations.length, operatingPoint));
  console.log(JSON.stringify({ artifact: path.relative(ROOT, PREDICTIONS), sha256: hash, rows: observations.length, positives, negatives: observations.length - positives, score_column: "predicted_probability", label_column: "true_label", ap, threshold: EXPECTED.threshold, ...counts, ...operatingPoint, output: path.relative(ROOT, OUTPUT) }, null, 2));
}

main();
