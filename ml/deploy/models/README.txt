# Model weights placeholder
# The ONNX file goes here after MATLAB training.
# It is gitignored (too large) and tracked by DVC instead.
#
# Expected file: netrapulse_resnet50.onnx
# To generate:  Run matlab/train_dr_classifier.m, then at the end:
#                 exportONNXNetwork(net, '../ml/deploy/models/netrapulse_resnet50.onnx')
# To verify:    python ml/scripts/verify_onnx.py
