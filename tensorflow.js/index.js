/**
 * Using TensorflowJS to do realtime superresolution.
 * Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
 */

// TensorflowJS backend to use. Can be 'webgl' or 'cpu'.
tf.setBackend('webgl');

/**
 * Global variables that are initialized in the load() function that is called
 * when the page and all resources have finished loading. Then, these are
 * available to all other functions.
 */
let inputVideo = null;
let outputVideo = null;
let scaledVideo = null;
let zoomRect = null;
let model = null;
let framesProcessed = 0;

// Some constants
const zoomRectWidth = 64;
const zoomRectHeight = 64;
const zoomFactor = 4;


/**
 * Load the model and initialize the global variables.
 * This is called by document.body.load event, so after the page and its
 * resources have finished loading.
 */
async function load() {
  inputVideo = document.getElementById('inputVideo');
  outputVideo = document.getElementById('outputVideo');
  scaledVideo = document.getElementById('scaledVideo');
  zoomRect = document.getElementById('zoomRect');

  model = await tf.loadLayersModel('./ESPCN/model.json');
  console.log('Model loaded.')

  inputVideo.play();

  // Make the zoomRect follow the mouse cursor
  let inputVideoPos = inputVideo.getBoundingClientRect();
  function onMouseMove(e) {
    zoomRect.style.left = Math.max(inputVideoPos.left, Math.min(inputVideoPos.right - zoomRectWidth - 1, e.pageX)) + 'px';
    zoomRect.style.top = Math.max(inputVideoPos.top, Math.min(inputVideoPos.bottom - zoomRectHeight - 1, e.pageY)) + 'px';
  }
  document.addEventListener('mousemove', onMouseMove);

  // Initialization is done. Start processing frames!
  window.requestAnimationFrame(doFrame);
}


/**
 * Process a single frame.
 */
async function doFrame() {
  // Stop after one frame for debugging purposes.
  // Disable this line to process the entire video.
  // if (framesProcessed >= 1) return;

  let inputData = grabPixelsInZoomRect();
  console.log('LR shape:', inputData.shape);

  let outputData = model.predict(inputData)
  outputData = outputData.squeeze();
  console.log('SR shape:', outputData.shape);

  outputData = outputData.clipByValue(0, 1);
  toOutputVideo(await outputData.data());

  // Render the next frame
  framesProcessed++;
  window.requestAnimationFrame(doFrame);
}


/**
 * Grab the video pixels inside the zoomRect and transfer them both to the
 * model and the scaledVideo output.
 */
function grabPixelsInZoomRect() {
  // Render the video frame onto a canvas so we can access it
  var canvas = document.createElement('canvas');
  canvas.width = inputVideo.width;
  canvas.height = inputVideo.height;
  var context = canvas.getContext('2d');
  context.drawImage(inputVideo, 0, 0);

  // Find the pixel data inside the zoomRect
  let inputVideoPos = inputVideo.getBoundingClientRect();
  let zoomRectPos = zoomRect.getBoundingClientRect();
  let pixelData = context.getImageData(zoomRectPos.left - inputVideoPos.left, zoomRectPos.top - inputVideoPos.top, zoomRectWidth, zoomRectHeight);

  // Write the pixel data to scaledVideo (scaled of course)
  context = scaledVideo.getContext('2d');
  context.drawImage(imageDataToCanvas(pixelData), 0, 0, zoomFactor * zoomRectWidth, zoomFactor * zoomRectHeight);

  // Convert the pixel data (currently an ImageData object) to an JavascriptJS
  // Tensor. The tensor should only have one channel: luminance, instead of RGB
  // color.
  const luminanceArray = new Float32Array(zoomRectHeight * zoomRectWidth);
  for (let i = 0; i < zoomRectHeight * zoomRectWidth; i++) {
    let r = pixelData.data[4 * i + 0]
    let g = pixelData.data[4 * i + 1]
    let b = pixelData.data[4 * i + 2]

    // Compute the luminance. This is the Y component of the YUV color space. 
    // https://en.wikipedia.org/wiki/YUV
    const y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
    luminanceArray[i] = y / 255;
  }
  return new tf.tensor(luminanceArray).reshape([1, 1, zoomRectHeight, zoomRectWidth]);
}


/**
 * Write the superresolution output of the model to the outputVideo component.
 *
 * To speed things along, the model operates on a single channel: the luminance
 * (the Y of YUV color space). In order to obtain a color image, we scale the U
 * and V components using the regular scaling of the browser (the scaledVideo
 * element). Then, we mix in the Y that was scaled by the model.
 */
function toOutputVideo(y) {
  let context1 = scaledVideo.getContext('2d');
  let rgba = context1.getImageData(0, 0, scaledVideo.width, scaledVideo.height);
  for (let i = 0; i < y.length; i++) {
    let r = rgba.data[4 * i + 0]
    let g = rgba.data[4 * i + 1]
    let b = rgba.data[4 * i + 2]
    let a = rgba.data[4 * i + 3]
    let mixed = mix(y[i], r, g, b, a);
    rgba.data[4 * i + 0] = mixed.r;
    rgba.data[4 * i + 1] = mixed.g;
    rgba.data[4 * i + 2] = mixed.b;
    rgba.data[4 * i + 3] = mixed.a;
  }
  let context2 = outputVideo.getContext('2d');
  context2.putImageData(rgba, 0, 0);
  console.log('Render complete');
}


/**
 * Write an ImageData object to a new canvas. This is needed to scale it later.
 */
function imageDataToCanvas(imageData) {
  let canvas = document.createElement('canvas');
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(imageData, 0, 0);
  return canvas;
}


/**
 * Mix the Y component with the RGBA component.
 */
function mix(y, r, g, b, a) {
  let yuv = rgb2yuv(r, g, b);
  yuv.y = y * 255;
  const rgb = yuv2rgb(yuv.y, yuv.u, yuv.v);
  return {r: rgb.r, g: rgb.g, b: rgb.b, a: a};
}


/**
 * Convert RGB to YUV color space.
 */
function rgb2yuv(r, g, b) {
  const y = 0.257 * r + 0.504 * g + 0.098 * b + 16;
  const u = -0.148 * r - 0.291 * g + 0.439 * b + 128;
  const v = 0.439 * r - 0.368 * g - 0.071 * b + 128;
  return({y:y, u:u, v:v});
}


/**
 * Convert YUV to RGB color space.
 */
function yuv2rgb(y, u, v){
  const r = y + 1.4075 * (v - 128);
  const g = y - 0.3455 * (u - 128) - (0.7169 * (v-128));
  const b = y + 1.7790 * (u - 128);
  return({r:r, g:g, b:b});
}
