import RNFS from 'react-native-fs';
import {AIModelSize} from '../types/media';
import {ClassifierWeights, MODEL_ARCHITECTURES} from './types';
import {createWeights} from './classifier';

const WEIGHTS_DIR = `${RNFS.DocumentDirectoryPath}/ai_models`;

function getWeightsPath(size: AIModelSize): string {
  return `${WEIGHTS_DIR}/ai_model_${size}.bin`;
}

async function ensureDir(): Promise<void> {
  const exists = await RNFS.exists(WEIGHTS_DIR);
  if (!exists) {
    await RNFS.mkdir(WEIGHTS_DIR);
  }
}

export function serializeWeights(weights: ClassifierWeights): ArrayBuffer {
  let totalBytes = 4; // number of layers (uint32)
  for (const layer of weights.layers) {
    totalBytes += 8; // outFeatures + inFeatures (uint32 each)
    totalBytes += layer.weight.byteLength;
    totalBytes += layer.bias.byteLength;
  }

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint32(offset, weights.layers.length, true);
  offset += 4;

  for (const layer of weights.layers) {
    view.setUint32(offset, layer.outFeatures, true);
    offset += 4;
    view.setUint32(offset, layer.inFeatures, true);
    offset += 4;

    const weightBytes = new Uint8Array(layer.weight.buffer, layer.weight.byteOffset, layer.weight.byteLength);
    new Uint8Array(buffer, offset, weightBytes.length).set(weightBytes);
    offset += weightBytes.length;

    const biasBytes = new Uint8Array(layer.bias.buffer, layer.bias.byteOffset, layer.bias.byteLength);
    new Uint8Array(buffer, offset, biasBytes.length).set(biasBytes);
    offset += biasBytes.length;
  }

  return buffer;
}

export function deserializeWeights(buffer: ArrayBuffer): ClassifierWeights {
  const view = new DataView(buffer);
  let offset = 0;

  const numLayers = view.getUint32(offset, true);
  offset += 4;

  const layers = [];
  for (let l = 0; l < numLayers; l++) {
    const outFeatures = view.getUint32(offset, true);
    offset += 4;
    const inFeatures = view.getUint32(offset, true);
    offset += 4;

    const weightLen = outFeatures * inFeatures;
    const weight = new Float32Array(buffer.slice(offset, offset + weightLen * 4));
    offset += weightLen * 4;

    const bias = new Float32Array(buffer.slice(offset, offset + outFeatures * 4));
    offset += outFeatures * 4;

    layers.push({weight, bias, outFeatures, inFeatures});
  }

  return {layers};
}

export async function saveWeights(
  size: AIModelSize,
  weights: ClassifierWeights,
): Promise<void> {
  await ensureDir();
  const buffer = serializeWeights(weights);
  const base64 = arrayBufferToBase64(buffer);
  await RNFS.writeFile(getWeightsPath(size), base64, 'base64');
}

export async function loadWeights(
  size: AIModelSize,
): Promise<ClassifierWeights | null> {
  const path = getWeightsPath(size);
  const exists = await RNFS.exists(path);
  if (!exists) {
    return null;
  }
  const base64 = await RNFS.readFile(path, 'base64');
  const buffer = base64ToArrayBuffer(base64);
  return deserializeWeights(buffer);
}

export async function deleteWeights(size: AIModelSize): Promise<void> {
  const path = getWeightsPath(size);
  const exists = await RNFS.exists(path);
  if (exists) {
    await RNFS.unlink(path);
  }
}

export async function weightsExist(size: AIModelSize): Promise<boolean> {
  return RNFS.exists(getWeightsPath(size));
}

export function initializeWeightsIfNeeded(
  size: AIModelSize,
  current: ClassifierWeights | null,
): ClassifierWeights {
  if (current) return current;
  return createWeights(size);
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return globalThis.btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}
