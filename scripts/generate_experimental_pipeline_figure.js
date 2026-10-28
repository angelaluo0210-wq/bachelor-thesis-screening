#!/usr/bin/env node
/* Render the Chapter 3 experimental-design diagram from fixed canonical facts. */

"use strict";

const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const OUTPUT = process.argv[2]
  ? path.resolve(ROOT, process.argv[2])
  : path.join(ROOT, "thesis", "img", "experimental_pipeline.pdf");

function escapePdf(value) {
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
  objects.forEach((object, i) => { offsets.push(Buffer.byteLength(output, "ascii")); output += `${i + 1} 0 obj\n${object}\nendobj\n`; });
  const xref = Buffer.byteLength(output, "ascii");
  output += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i < offsets.length; i += 1) output += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  output += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(output, "ascii");
}

function main() {
  const c = [];
  const text = (font, size, x, y, value) => c.push(`0 0 0 rg BT /${font} ${size} Tf ${x} ${y} Td (${escapePdf(value)}) Tj ET`);
  const box = (x, y, w, h, fill, lines, border = "0 0 0") => {
    c.push(`${fill} rg ${x} ${y} ${w} ${h} re f ${border} RG 0.8 w ${x} ${y} ${w} ${h} re S`);
    const start = y + h / 2 + (lines.length - 1) * 6;
    lines.forEach((line, i) => text(i === 0 ? "F2" : "F1", i === 0 ? 9.5 : 8.2, x + 8, start - i * 12, line));
  };
  const arrow = (x1, y1, x2, y2) => {
    c.push(`0 0 0 RG 0 0 0 rg 0.9 w ${x1} ${y1} m ${x2} ${y2} l S`);
    const a = Math.atan2(y2 - y1, x2 - x1), d = 6;
    c.push(`${x2} ${y2} m ${(x2 - d * Math.cos(a - 0.45)).toFixed(2)} ${(y2 - d * Math.sin(a - 0.45)).toFixed(2)} l ${(x2 - d * Math.cos(a + 0.45)).toFixed(2)} ${(y2 - d * Math.sin(a + 0.45)).toFixed(2)} l h f`);
  };
  c.push("1 1 1 rg 0 0 540 470 re f");
  text("F2", 14, 150, 445, "Review-disjoint experimental pipeline");
  box(187, 390, 166, 38, "0.94 0.94 0.94", ["Corpus", "26 systematic reviews | 169,280 records"]);
  box(187, 334, 166, 34, "0.90 0.90 0.90", ["Review-disjoint split", "Complete reviews assigned once"]);
  arrow(270, 390, 270, 368);
  box(28, 260, 145, 50, "0.94 0.94 0.94", ["Train", "18 reviews", "123,291 records"]);
  box(198, 260, 145, 50, "0.94 0.94 0.94", ["Validation", "4 reviews", "22,397 records"]);
  box(368, 260, 145, 50, "0.94 0.94 0.94", ["Test", "4 reviews | 23,592 records", "Held out from all selection"]);
  arrow(270, 334, 100, 310); arrow(270, 334, 270, 310); arrow(270, 334, 440, 310);
  c.push("0.97 0.97 0.97 rg 18 24 340 215 re f 0 0 0 RG 1.0 w 18 24 340 215 re S");
  text("F2", 10, 36, 226, "MODEL DEVELOPMENT: Train + Validation only");
  box(38, 170, 140, 35, "1 1 1", ["PubMedBERT-based model", "criteria + title + abstract"]);
  box(198, 155, 140, 65, "1 1 1", ["Pooling comparison", "CLS | mean | attention | concat", "Select CLS by Validation AP"]);
  box(38, 84, 140, 65, "1 1 1", ["Training strategies", "head only | partial | LoRA", "Select LoRA, adapter LR 5e-5"]);
  box(198, 84, 140, 65, "1 1 1", ["Checkpoint selection", "Best epoch 4 by Validation AP", "Threshold calibration: Recall >= 0.95"]);
  box(198, 34, 140, 35, "0.88 0.88 0.88", ["FREEZE", "model + checkpoint + threshold"]);
  arrow(100, 260, 100, 205); arrow(270, 260, 270, 220); arrow(178, 187, 198, 187); arrow(268, 155, 108, 149); arrow(108, 155, 108, 149); arrow(178, 116, 198, 116); arrow(268, 84, 268, 69);
  c.push("0.97 0.97 0.97 rg 360 35 153 202 re f 0 0 0 RG 1.0 w 360 35 153 202 re S");
  text("F2", 11, 374, 212, "FINAL EVALUATION");
  box(374, 164, 127, 43, "1 1 1", ["One final held-out", "Test evaluation"]);
  box(374, 63, 127, 45, "1 1 1", ["POST-HOC sensitivity", "exact-identifier overlap", "stored Test predictions only"]);
  arrow(440, 260, 440, 207); arrow(338, 51, 374, 185); arrow(500, 164, 500, 108);
  text("F1", 8.5, 378, 145, "No Test use for pooling,");
  text("F1", 8.5, 378, 135, "strategy, checkpoint, or");
  text("F1", 8.5, 378, 125, "threshold selection");
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, makePdf(540, 470, c.join("\n")));
  console.log(`Wrote ${path.relative(ROOT, OUTPUT)}`);
}

main();
