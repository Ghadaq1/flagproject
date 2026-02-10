def predict_flag(model, svm, le, img_array):
    features = model.predict(img_array)
    prediction = svm.predict(features)
    label = le.inverse_transform(prediction)
    return label[0]
