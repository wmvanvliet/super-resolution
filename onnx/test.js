function fromPixels(img) {
  var canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  var context = canvas.getContext('2d');
  context.drawImage(img, 0, 0);
  const pixelData = context.getImageData(0, 0, img.width, img.height).data;
  const pixelArray = new Float32Array(img.width * img.height);
  for (let i = 0; i < img.width * img.height; i++) {
      let lightness = (pixelData[4 * i + 0] + pixelData[4 * i + 1] + pixelData[4 * i + 2]) / 3;
      pixelArray[i] = lightness / 255;
  }
  return new Tensor(pixelArray, 'float32', [1, 1, img.width, img.height]);
}

function clampUint8(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function toPixels(data) {
  const width = data.dims[2];
  const height = data.dims[3];
  var canvas = document.createElement('canvas');
  var context = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  const bytes = new Uint8ClampedArray(4 * width * height);
  for (let i = 0; i < width * height; i++) {
    let lightness = clampUint8(data.data[i] * 255);
    bytes[4 * i + 0] = lightness;
    bytes[4 * i + 1] = lightness;
    bytes[4 * i + 2] = lightness;
    bytes[4 * i + 3] = 255;
  }
  const img = new ImageData(bytes, width, height);
  context.putImageData(img, 0, 0);
  return canvas
}

async function test() {
  const inputImage = document.getElementById('inputImage');
  await new Promise(resolve => inputImage.addEventListener('load', event => resolve()));
  let inputData = fromPixels(inputImage);

  const session = new onnx.InferenceSession({backendHint: 'cpu'});
  //await session.loadModel('./super_resolution.onnx');
  await session.loadModel('./ESPCN.onnx');
  console.log('Model loaded');

  const outputMap = await session.run([inputData]);
  console.log('Model finished');
  const outputData = outputMap.get('output');

  const canvas = toPixels(outputData);
  document.body.appendChild(canvas);
}

test();
