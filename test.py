from scipy.io import loadmat

model = loadmat("models/gesture_model.mat", simplify_cells=True)

gesture = model["gesture_model"]
print(gesture)