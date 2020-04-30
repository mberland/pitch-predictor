# pitch-predictor

This is the repo for the blog post here: [Pitch Prediction with Keras](https://medium.com/@matthewberland/pitch-prediction-with-keras-part-i-d2a3c28e6568)

Requires: Python 3.6+; [`statsapi`](https://github.com/toddrob99/MLB-StatsAPI/); keras 1.15.0

1. Run `python ./download_games.py` to get the data from MLB.
2. Run `python ./pitch_predictor.py` to run the model.
3. If it works, you can set `TESTING_FLAG = False` in `./pitch_predictor.py` and run it at full strength.
