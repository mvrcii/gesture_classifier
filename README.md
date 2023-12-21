# FNN Gesture Classifier
This repository showcases a self-written gesture classifier developed with pure numpy and without the usage of a machine learning library, such as TensorFlow or PyTorch. It integrates two applications: a reveal.js slideshow and a tetris game. The user is able to control both applications with the use of predefined human gestures.

[Teaser Video](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission/-/blob/main/Teaser/MachineLearning_WS2223_Team7_Teaser.mp4) as well as the [presentation slides](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission/-/blob/main/Teaser/ML_Presentation.pdf)

## 1. Getting Started
1. Clone the project on your local machine (we recommend to use `git clone --depth 1 https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission.git` to not clone history including large video files)
2. Create a new conda environment with python version 3.7
3. Activate the environment, navigate to the main directory of the project and install the requirements via
   `pip install -r requirements.txt`

## 2. Applications
### 2.1 Slideshow:
1. Run the `/Prediction_Mode/slideshow.py` script
2. Click on the IP address `http://127.0.0.1:8000` in the python console to open the slideshow in the browser
3. Click inside the browser window to give the slideshow the focus

#### Usable Gestures:
- Swipe Left -> One slide to the left
- Swipe Right -> One slide to the right
- Swipe Up -> One slide upwards
- Swipe Down -> One slide downwards
- Rotate Right -> Rotates all rotatable elements to the right
- Rotate Right -> Rotates all rotatable elements to the left
- Spread -> Increases the slide size
- Pinch -> Decreases the slide size
- Flip Table -> Opens / Closes the overview mode
- Spin -> If the current slide contains a rotatable picture, it spins by 360° | If the current slide contains a video, it changes the video speed from 1x to 4x and vice versa
- Point -> Starts / Stops a video

### 2.2 Tetris:
1. Run the `/Game/tetris.py` script and have fun!

#### Usable Gestures:
- Swipe Left -> Tetris tile to the left
- Swipe Right -> Tetris tile to the right
- Swipe Down -> Increases the downward speed of a Tetris tile
- Rotate Left -> Rotates the Tetris tile to the left
- Rotate Right -> Rotates the Tetris tile to the right
- Flip Table -> Changes the Game status to Play / Pause
- Spin -> Rotates the Tetris grid by 180°

## 3. Troubleshooting

* **OpenCV-Python Package:**
  If the used methods from CV2 cannot be found and are thus highlighted, it is necessary to [add the CV2 path to the interpreter paths](https://github.com/opencv/opencv/issues/20997#issuecomment-1328068006).
* **Confusion Matrix Animation:** In order to successfully create an animation from the confusion matrix, `ffmpeg` must be installed. On macOS use the following: `brew install ffmpeg`
## 4. Authors and acknowledgment
Nowak Micha, Roth Marcel, Friese Jan-Philipp

## 5. License
MIT License
