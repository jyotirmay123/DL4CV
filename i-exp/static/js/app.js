// document.onload = () => {
    //Parameters
    // const s = document.getElementById('objDetect');
    const sourceVideo = "local-video"; //s.getAttribute("data-source");  //the source video to use
    const emoFeedVideo = "remote-video";
    const uploadWidth = 385; // s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
    const uploadHeight = 288;
    const mirror = false; // s.getAttribute("data-mirror") || false; //mirror the boundary boxes
    const scoreThreshold =  0.5; // s.getAttribute("data-scoreThreshold") || 0.5;
    const apiServer = window.location.origin + '/image'; // s.getAttribute("data-apiServer") || window.location.origin + '/image'; //the full TensorFlow Object Detection API server url
    const emo_list = ["NEUTRAL", "ANGER", "CONTEMPT", "DISGUST", "FEAR", "HAPPY", "SADNESS", "SURPRISE"];

    //Video element selector
    v = document.getElementById(sourceVideo);
    vRemote = document.getElementById(emoFeedVideo);
    // i = document.getElementById("bg");
    //for starting events
    let isPlaying = false,
        gotMetadata = false;

    //Canvas setup

    //create a canvas to grab an image for upload
    let imageCanvas = document.createElement('canvas');
    let imageCtx = imageCanvas.getContext("2d");

    //create a canvas for drawing object boundaries
    let drawCanvas = document.createElement('canvas');
    document.body.appendChild(drawCanvas);
    let drawCtx = drawCanvas.getContext("2d");


    // Starting events

    //check if metadata is ready - we need the video size
    v.onloadedmetadata = () => {
        gotMetadata = true;
        if (isPlaying)
            startObjectDetection();
    };

    //see if the video has started playing
    v.onplaying = () => {
        isPlaying = true;
        if (gotMetadata) {
            startObjectDetection();
        }
    };

    // v.addEventListener('play', function () {
    //     isPlaying = true;
    //     if (gotMetadata) {
    //         startObjectDetection();
    //     }
    //     var $this = this; //cache
    //     (function loop() {
    //         if (!$this.paused && !$this.ended) {
    //             drawCtx.drawImage($this, 0, 0);
    //             setTimeout(loop, 1000 / 30); // drawing at 30fps
    //         }
    //     })();
    // }, 0);
// };

let loadCam = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    // const localVideo = document.getElementById("local-video");
    if (v) {
        v.srcObject = stream;
    }
     if (vRemote) {
        vRemote.srcObject = stream;
    }
}

//draw boxes and labels on each detected object
async function drawBoxes(object) {
    let x = 0;
    let y = 0;
    let width = (uploadWidth * drawCanvas.width) - x;
    let height = (uploadHeight * drawCanvas.height) - y;
    //clear the previous drawings
    drawCtx.clearRect(x, y, width, height);
    // drawCtx.fillText("JYOTIRMAY" + " - " + Math.round(98.76 * 100) + "%", 100 + 5, 100 + 20);
    //filter out objects that contain a class_name and then draw boxes and labels on each

    if(object.max_index !== null) {
        let fontSize=24
        drawTextBG(drawCtx, emo_list[object.max_index], `bold ${fontSize}px verdana, sans-serif`, x + 5, y + 20, fontSize);
    }

    if(object.face !== null) {
        let x_ = x + object.face[0] * drawCanvas.width;
        let y_ = y + object.face[1] * drawCanvas.height;
        let width_ = object.face[2] * drawCanvas.width;
        let height_ = object.face[3] * drawCanvas.height;
        //flip the x axis if local video is mirrored
        if (mirror) {
            x_ = drawCanvas.width - (x_ + width_);
        }
        drawCtx.strokeRect(x_, y_, width_, height_);
    }
}

/// expand with color, background etc.
function drawTextBG(ctx, txt, font, x, y, fontSize) {

    /// lets save current state as we make a lot of changes
    ctx.save();

    /// set font
    ctx.font = font;

    /// draw text from top - makes life easier at the moment
    ctx.textBaseline = 'top';

    /// color for background
    ctx.fillStyle = 'rgba(0,0,0,0.8)';

    /// get width of text
    var width = ctx.measureText(txt).width;

    /// draw background rect assuming height of font
    ctx.fillRect(x-2, y-2, width+4, fontSize+4);

    /// text color
    ctx.fillStyle = '#ffb700';

    /// draw text on top
    ctx.fillText(txt, x, y);

    /// restore original state
    ctx.restore();
}

//Add file blob to a form and post
function postFile(file) {

    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("threshold", scoreThreshold);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiServer, true);
    xhr.onload = function () {
        if (this.status === 200) {
            let resObjects = JSON.parse(this.response);
            //draw the boxes
            drawBoxes(resObjects);

            // i.src = resObjects.feed;
        }
        else {
            console.error(xhr);
        }

        //Save and send the next image
        imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
        imageCanvas.toBlob(postFile, 'image/jpeg');
    };
    xhr.send(formdata);
}

//Start object detection
function startObjectDetection() {

    console.log("starting object detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 4;
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "20px Verdana";
    drawCtx.fillStyle = "cyan";

    //Save and send the first image
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    imageCanvas.toBlob(postFile, 'image/jpeg');

}