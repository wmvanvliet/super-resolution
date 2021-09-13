/**
 * Some test code to load an ONNX model using ONNX.js and attempt
 * superresolution processing on a video.
 *
 * Author: Marijn van Vliet <marijn.vanvliet@aalto.fi>
 */


/**
 * Global variables that are initialized in the load() function that is called
 * when the page and all resources have finished loading. Then, these are
 * available to all other functions.
 */
let session = null;
let inputVideo = null;
let outputVideo = null;
let framesProcessed = 0;


/**
 * Load the model and initialize the global variables.
 * This is called by document.body.load event, so after the page and its
 * resources have finished loading.
 */
async function load() {
  inputVideo = document.getElementById('inputVideo');
  outputVideo = document.getElementById('outputVideo');
  session = new onnx.InferenceSession();
  await session.loadModel('./ESPCN.onnx');
  console.log('Model loaded');
  inputVideo.play();
  window.requestAnimationFrame(processFrame);
}


/**
 * Process a single frame of video data.
 */
async function processFrame() {
  // Stop after one frame for debugging purposes.
  // Disable this line to process the entire video.
  if (framesProcessed >= 1) return;

  let inputData = fromPixels(inputVideo);
  console.log('Image loaded:', inputData.dims);

  const outputMap = await session.run([inputData]);
  console.log('Model finished');

  const outputData = outputMap.get('output');
  const canvas = toPixels(outputData, outputVideo);
  console.log('Image rendered.');

  framesProcessed++;
  window.requestAnimationFrame(processFrame);
}


/**
 * Given an <img> or <video>, grab the pixels as an ONNX Tensor in the proper
 * format for feeding into the model.
 */
function fromPixels(img) {
  // Draw the image to a <canvas> so we can access the pixel data.
  var canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  var context = canvas.getContext('2d');
  context.drawImage(img, 0, 0);
  const pixelData = context.getImageData(0, 0, img.width, img.height).data;

  // Convert the pixel data (currently an ImageData object) to an ONNX tensor.
  // The tensor should only have one channel: lightness, instead of RGB color.
  const pixelArray = new Float32Array(img.height * img.width);
  for (let i = 0; i < img.height * img.width; i++) {
      let r = pixelData[4 * i + 0]
      let g = pixelData[4 * i + 1]
      let b = pixelData[4 * i + 2]

      // Compute the lightness. This is the Y component of the YUV color space. 
      // https://en.wikipedia.org/wiki/YUV
      const y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
      pixelArray[i] = y / 255;
  }
  return new Tensor(pixelArray, 'float32', [1, 1, img.height, img.width]);
}


/**
 * Given an ONNX tensor as produced by the model, render the values as an image
 * into the given <canvas>.
 */
function toPixels(data, canvas) {
  const height = data.dims[2];
  const width = data.dims[3];
  var context = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  const bytes = new Uint8ClampedArray(4 * height * width);
  for (let i = 0; i < height * width; i++) {
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


/**
 * Clamp a value between 0 and 255 so it fits into an uint8 (=byte).
 */
function clampUint8(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}
