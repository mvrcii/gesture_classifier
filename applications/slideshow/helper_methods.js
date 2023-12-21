const uid = function (i) {
    return function () {
        return "generated_id-" + (++i);
    };
}(0);

const rotateRightRotatables = function(rotationAngles) {
  return function(slide) {
    const rotatables = Array.from(slide.getElementsByClassName("rotatable"))
    if(rotatables.length > 0){
      rotatables.forEach(function(elem){
        if (!elem.id) elem.id = uid();

        if(!rotationAngles[elem.id]) {
          rotationAngles[elem.id] = 0
        }
        new_rotation = rotationAngles[elem.id] + 90
        console.log(new_rotation);
        elem.style.transform = "rotate(" + (new_rotation) + "deg)"
        rotationAngles[elem.id] = new_rotation
      });
    }
  }
}({})

const rotateLeftRotatables = function(rotationAngles) {
  return function(slide) {
    const rotatables = Array.from(slide.getElementsByClassName("rotatable"))
    if(rotatables.length > 0){
      rotatables.forEach(function(elem){
        if (!elem.id) elem.id = uid();

        if(!rotationAngles[elem.id]) {
          rotationAngles[elem.id] = 0
        }
        new_rotation = rotationAngles[elem.id] - 90
        console.log(new_rotation);
        elem.style.transform = "rotate(" + (new_rotation) + "deg)"
        rotationAngles[elem.id] = new_rotation
      });
    }
  }
}({})

const rotate = function(slide, rotationStepSize) {
  const rotatables = Array.from(slide.getElementsByClassName("rotatable"));
  if(rotatables.length > 0){
    rotatables.forEach(function(elem){
      console.log(elem);
      var currentTransform = elem.style.transform;
      var currentRotation = 0; // Default rotation angle is 0

      if (currentTransform && currentTransform !== '') {
        var match = currentTransform.match(/rotate\(\s*([-]?\d+(\.\d+)?)\s*deg\)/);

        if (match && match[1]) {
          currentRotation = parseFloat(match[1]);
        }
      }
      console.log('Current rotation: ' + currentRotation);
      var newRotation = currentRotation + rotationStepSize;
      console.log('New Rotation: ' + newRotation);
      rotateContent(elem, newRotation);
    });
  }
}

const zoom = function(zoomStepSize) {
  var contentElement = document.querySelector('.reveal');
  // Get current scale factor
  var currentTransform = contentElement.style.transform;
  var currentScale = 1; // Default scale factor is 1

  // Check if currentTransform is not empty or null
  if (currentTransform && currentTransform !== '') {
    // Use regular expression to match and extract the numeric value
    var match = currentTransform.match(/scale\(\s*(\d+(\.\d+)?)\s*\)/);

    // Check if match is not null and has a valid scale factor
    if (match && match[1]) {
      currentScale = parseFloat(match[1]);
    }
  }

  // Calculate new scale factor
  var newScale = currentScale + zoomStepSize;

  // Call the scaleContent function with new scale factor
  scaleContent(newScale);
}

function scaleContent(scaleFactor) {
  var contentElement = document.querySelector('.reveal');
  contentElement.style.transform = 'scale(' + scaleFactor + ')';
}

const rotateContent = function(element, rotationAngle) {
  element.style.transform = 'rotate(' + rotationAngle + 'deg)';
}
