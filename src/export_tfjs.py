# src/export_tfjs.py
#
# PURPOSE:
#   Convert the trained Keras model to TensorFlow.js format for
#   client-side browser deployment (no backend server).
#
#   - Reads the trained model from models/checkpoints/.
#   - Exports layers/model artifacts to models/tfjs_model/ (and a copy
#     to web/model/ for the demo to load).
#   - CRITICAL: browser-side preprocessing in web/index.html must exactly
#     match training preprocessing (input size 224x224, normalization
#     scheme used by the chosen backbone).
#
# STATUS: Skeleton — no implementation logic yet.
