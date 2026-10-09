# Legacy code (not used by Vani-Setu)

These files come from the open-source project this repository started from:
**Sign Language Interpreter using Deep Learning** by Harsh Gupta (harshbg), built at HackUNT-19 in 2019,
MIT licence: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning

That project segmented the hand with a colour histogram and classified cropped hand images with a Keras CNN.
Vani-Setu replaced that approach with MediaPipe hand landmarks, a scikit-learn classifier and a Flask web app,
so nothing in this folder is imported by the current app.

The files are kept for reference and credit only. They cannot run in this checkout: the gesture images,
`cnn_model_keras2.h5` and the original dependencies are not included.

| File | Original purpose |
|------|------------------|
| `set_hand_histogram.py` | Capture a skin-colour histogram for segmentation |
| `create_gestures.py`, `Rotate_images.py`, `load_images.py`, `display_gestures.py` | Record, augment and load gesture images |
| `cnn_model_train.py` | Train the Keras CNN on gesture images |
| `final.py` | Webcam app using the CNN |
| `gesture_db.db`, `hist` | Gesture labels and saved histogram |
| `sign_language_app.py` | Early desktop prototype (OpenCV + MediaPipe + pyttsx3), replaced by the web app |
| `Install_Packages*.txt` | Original dependency lists |
