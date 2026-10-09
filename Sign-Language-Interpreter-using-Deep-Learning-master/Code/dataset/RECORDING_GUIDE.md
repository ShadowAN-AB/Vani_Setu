# How to record samples

Recording runs only on **localhost** (the public site turns it off). Every sample is saved as the
raw 21 hand points plus the signer's name, the hand MediaPipe saw, and whether it is a test sample,
in `dataset/recordings/samples.csv`. Letters and phrases share that file but are kept apart by the
`kind` column, so phrase samples never reach the letter model.

## Steps

1. Start the app (`./run.sh`) and open http://127.0.0.1:5001.
2. Log in and click **Start camera**.
3. Choose **SPELL** to record letters, or **PHRASE** (ASL / ISL tab) to record phrases.
4. Type the **signer's name** (required) and pick the sign in the dropdown.
5. Tick **Test set** only when recording the new-signer test set (see below).
6. Click **Record dataset**. Hold the sign for 3–4 seconds while slowly changing distance and angle.
   One sample is saved every 4 frames.
7. Click **Stop recording** when the counter reaches about 25, then move to the next sign.

## What to record

| Set | Who | What | Test set box |
|-----|-----|------|--------------|
| New-signer test set | 4–5 classmates who are **not** in the training data | All 26 letters, ~10 samples each | Ticked |
| Extra letter training data | Team members | Letters the model confuses most: D, P, R, U, X, Z | Not ticked |
| Phrase training data | All three team members and at least one other person | Each phrase or ISL sign, ~25 samples per person | Not ticked |

Use a different name for every person, and the same name every time that person records.
The trainer keeps all **Test set** samples out of training and reports their accuracy separately.

## After recording

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
.venv/bin/python train_classifier.py   # retrain the letter model
.venv/bin/python evaluate.py           # curves, robustness and new-signer results
```

Or click **Retrain model** in the app. Commit `dataset/recordings/samples.csv` so teammates get the samples.

## Tips for good data

- Good light, the whole hand inside the frame
- Vary distance and tilt the wrist slightly while recording
- Record with the hand you will use in the demo; also record a few with the other hand
- Pause recording while changing to the next sign
