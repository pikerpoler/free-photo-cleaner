import {AIModelSize} from '../types/media';
import {
  ClassifierWeights,
  ClassifierGradients,
  LayerWeights,
  LayerGradients,
  MODEL_ARCHITECTURES,
} from './types';

function xavierInit(inFeatures: number, outFeatures: number): Float32Array {
  const scale = Math.sqrt(2.0 / (inFeatures + outFeatures));
  const arr = new Float32Array(outFeatures * inFeatures);
  for (let i = 0; i < arr.length; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    arr[i] = scale * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return arr;
}

export function createWeights(size: AIModelSize): ClassifierWeights {
  const arch = MODEL_ARCHITECTURES[size];
  const layers: LayerWeights[] = [];
  for (let i = 0; i < arch.length - 1; i++) {
    const inFeatures = arch[i];
    const outFeatures = arch[i + 1];
    layers.push({
      weight: xavierInit(inFeatures, outFeatures),
      bias: new Float32Array(outFeatures),
      outFeatures,
      inFeatures,
    });
  }
  return {layers};
}

export function createGradients(weights: ClassifierWeights): ClassifierGradients {
  const layers: LayerGradients[] = weights.layers.map(layer => ({
    weightGrad: new Float32Array(layer.weight.length),
    biasGrad: new Float32Array(layer.bias.length),
  }));
  return {layers, count: 0};
}

export function zeroGradients(grads: ClassifierGradients): void {
  grads.count = 0;
  for (const layer of grads.layers) {
    layer.weightGrad.fill(0);
    layer.biasGrad.fill(0);
  }
}

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

export interface ForwardResult {
  activations: Float32Array[]; // input + each layer output (pre-activation stored separately)
  preActivations: Float32Array[];
  output: number;
}

export function forward(
  weights: ClassifierWeights,
  input: Float32Array,
): ForwardResult {
  const activations: Float32Array[] = [input];
  const preActivations: Float32Array[] = [];
  let current = input;

  for (let l = 0; l < weights.layers.length; l++) {
    const layer = weights.layers[l];
    const out = new Float32Array(layer.outFeatures);

    for (let o = 0; o < layer.outFeatures; o++) {
      let sum = layer.bias[o];
      const wOffset = o * layer.inFeatures;
      for (let i = 0; i < layer.inFeatures; i++) {
        sum += layer.weight[wOffset + i] * current[i];
      }
      out[o] = sum;
    }

    preActivations.push(out);

    const isLastLayer = l === weights.layers.length - 1;
    if (isLastLayer) {
      // Sigmoid for output
      const activated = new Float32Array(layer.outFeatures);
      for (let o = 0; o < layer.outFeatures; o++) {
        activated[o] = sigmoid(out[o]);
      }
      activations.push(activated);
    } else {
      // ReLU for hidden layers
      const activated = new Float32Array(layer.outFeatures);
      for (let o = 0; o < layer.outFeatures; o++) {
        activated[o] = relu(out[o]);
      }
      activations.push(activated);
    }

    current = activations[activations.length - 1];
  }

  return {
    activations,
    preActivations,
    output: current[0],
  };
}

/**
 * Backward pass: compute gradients for BCE loss and accumulate into grads.
 * BCE loss = -[y*log(p) + (1-y)*log(1-p)]
 * dLoss/dp = (p - y) / (p * (1-p))
 * Combined with sigmoid derivative: dLoss/dz_last = p - y
 */
export function backward(
  weights: ClassifierWeights,
  grads: ClassifierGradients,
  fwdResult: ForwardResult,
  label: 0 | 1,
): void {
  const {activations, preActivations} = fwdResult;
  const numLayers = weights.layers.length;

  // Output layer gradient: sigmoid + BCE → dL/dz = p - y
  let delta = new Float32Array(weights.layers[numLayers - 1].outFeatures);
  const p = activations[numLayers][0];
  delta[0] = p - label;

  // Backpropagate through layers in reverse
  for (let l = numLayers - 1; l >= 0; l--) {
    const layer = weights.layers[l];
    const input = activations[l];
    const layerGrad = grads.layers[l];

    // Accumulate weight and bias gradients
    for (let o = 0; o < layer.outFeatures; o++) {
      layerGrad.biasGrad[o] += delta[o];
      const wOffset = o * layer.inFeatures;
      for (let i = 0; i < layer.inFeatures; i++) {
        layerGrad.weightGrad[wOffset + i] += delta[o] * input[i];
      }
    }

    // Propagate delta to previous layer (skip for first layer)
    if (l > 0) {
      const prevDelta = new Float32Array(layer.inFeatures);
      for (let i = 0; i < layer.inFeatures; i++) {
        let sum = 0;
        for (let o = 0; o < layer.outFeatures; o++) {
          sum += layer.weight[o * layer.inFeatures + i] * delta[o];
        }
        // ReLU derivative
        const preAct = preActivations[l - 1][i];
        prevDelta[i] = preAct > 0 ? sum : 0;
      }
      delta = prevDelta;
    }
  }

  grads.count++;
}

export function sgdStep(
  weights: ClassifierWeights,
  grads: ClassifierGradients,
  learningRate: number,
): void {
  const scale = 1.0 / grads.count;

  for (let l = 0; l < weights.layers.length; l++) {
    const layer = weights.layers[l];
    const layerGrad = grads.layers[l];

    for (let i = 0; i < layer.weight.length; i++) {
      layer.weight[i] -= learningRate * scale * layerGrad.weightGrad[i];
    }
    for (let i = 0; i < layer.bias.length; i++) {
      layer.bias[i] -= learningRate * scale * layerGrad.biasGrad[i];
    }
  }
}

export function predict(weights: ClassifierWeights, input: Float32Array): number {
  return forward(weights, input).output;
}
