/** MLP learning rate: log10 scale 0.001 … 0.1 (slider center → 0.01). */
export const MLP_LR_MIN = 0.001;
export const MLP_LR_MAX = 0.1;

export function mlpLearningRateToSlider(lr: number): number {
  const clamped = Math.min(MLP_LR_MAX, Math.max(MLP_LR_MIN, lr));
  const logMin = Math.log10(MLP_LR_MIN);
  const logMax = Math.log10(MLP_LR_MAX);
  return (Math.log10(clamped) - logMin) / (logMax - logMin);
}

export function sliderToMlpLearningRate(t: number): number {
  const s = Math.min(1, Math.max(0, t));
  const logMin = Math.log10(MLP_LR_MIN);
  const logMax = Math.log10(MLP_LR_MAX);
  return 10 ** (logMin + s * (logMax - logMin));
}

export function formatMlpLearningRate(lr: number): string {
  if (lr >= 0.01) {
    return lr.toFixed(3);
  }
  return lr.toFixed(4);
}

/** MLP batch size: powers of two from 2^0 … 2^7 (slider center → 16). */
const MLP_BATCH_MAX_EXP = 7;

export function snapMlpBatchSize(size: number): number {
  const exp = Math.round(Math.log2(Math.max(1, size)));
  const clampedExp = Math.min(MLP_BATCH_MAX_EXP, Math.max(0, exp));
  return 2 ** clampedExp;
}

export function mlpBatchSizeToSlider(batch: number): number {
  const snapped = snapMlpBatchSize(batch);
  const exp = Math.log2(snapped);
  return exp / MLP_BATCH_MAX_EXP;
}

export function sliderToMlpBatchSize(t: number): number {
  const s = Math.min(1, Math.max(0, t));
  const exp = Math.round(MLP_BATCH_MAX_EXP * s);
  return 2 ** exp;
}
