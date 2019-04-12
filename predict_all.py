from predict import predict_all
steps = [16, 8, 4, 2, 1]
for i in steps:
    print('\n\n',i)
    predict_all(i)
