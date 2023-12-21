let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  const currentSlide = Reveal.getCurrentSlide();
  switch(event.data){
    case "right":
      console.log("received 'right' event");
      Reveal.right();
      break;
    case "left":
      console.log("received 'left' event");
      Reveal.left();
      break;
    case "up":
      console.log("received 'up' event");
      Reveal.up();
      break;
    case "down":
      console.log("received 'down' event");
      Reveal.down();
      break;
    case "rotate_right":
      console.log("received 'rotate_right' event");
      console.log(currentSlide);
      rotate(currentSlide, 90);
      break;
    case "rotate_left":
      console.log("received 'rotate_left' event");
      rotate(currentSlide, -90);
      break;
    case "zoom_in":
      console.log("received 'zoom_in' event");
      // increases zoom by 10%
      zoom(0.4); // `zoom()` is defined in helper_methods.js
      break;
    case "zoom_out":
      console.log("received 'zoom_out' event");

      // decreases zoom by 10%
      zoom(-0.4); // `zoom()` is defined in helper_methods.js
      break;
    case "flip":
      console.log("received 'flip' event");
      Reveal.toggleOverview();
      //flipRotatables(currentSlide);  // defined in helper_methods.js
      break;
    case "spin":
      rotate(currentSlide, 360);
      let videoElementSpin = currentSlide.querySelector('video')
      if (videoElementSpin.playbackRate == 1){
        videoElementSpin.playbackRate = 3;
      }
      else {
        videoElementSpin.playbackRate = 1;
      }
      break;
    case "point":
      console.log("received 'point' event");
      let videoElement = currentSlide.querySelector('video')
      videoElement.paused ? videoElement.play() : videoElement.pause()
      break;

    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
