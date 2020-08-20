// document.onload = () => {
    //Parameters
    // const s = document.getElementById('objDetect');
    const sourceVideo = "local-video"; //s.getAttribute("data-source");  //the source video to use
    const uploadWidth = 640; // s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
    const mirror = false; // s.getAttribute("data-mirror") || false; //mirror the boundary boxes
    const scoreThreshold =  0.5; // s.getAttribute("data-scoreThreshold") || 0.5;
    const apiServer = window.location.origin + '/image'; // s.getAttribute("data-apiServer") || window.location.origin + '/image'; //the full TensorFlow Object Detection API server url

    //Video element selector
    v = document.getElementById(sourceVideo);
    i = document.getElementById("bg");
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
        console.log("video metadata ready");
        gotMetadata = true;
        if (isPlaying)
            startObjectDetection();
    };

    //see if the video has started playing
    v.onplaying = () => {
        console.log("video playing");
        isPlaying = true;
        if (gotMetadata) {
            startObjectDetection();
        }
    };
// };

let loadCam = async () => {
    console.log("loaddd");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const localVideo = document.getElementById("local-video");
    console.log(localVideo, stream);
    if (localVideo) {
        console.log("coming here");
        localVideo.srcObject = stream;
    }
}

//draw boxes and labels on each detected object
function drawBoxes(objects) {

    //clear the previous drawings
    drawCtx.clearRect(0, 0, drawCanvas.width-100, drawCanvas.height-100);
    drawCtx.fillText("JYOTIRMAY" + " - " + Math.round(98.76 * 100) + "%", 100 + 5, 100 + 20);
    //filter out objects that contain a class_name and then draw boxes and labels on each
    // objects.filter(object => object.class_name).forEach(object => {
    //
    //     let x = object.x * drawCanvas.width;
    //     let y = object.y * drawCanvas.height;
    //     let width = (object.width * drawCanvas.width) - x;
    //     let height = (object.height * drawCanvas.height) - y;
    //
    //     //flip the x axis if local video is mirrored
    //     if (mirror) {
    //         x = drawCanvas.width - (x + width)
    //     }
    //
    //     drawCtx.fillText(object.class_name + " - " + Math.round(object.score * 100) + "%", x + 5, y + 20);
    //     drawCtx.strokeRect(x, y, width, height);
    //
    // });
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
            //
            // //draw the boxes
            // drawBoxes(objects);
            // console.log(objects);
            i.src = resObjects.feed;
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