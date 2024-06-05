import opensmile
print("imported")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
y = smile.process_file('/home/abhijith/Documents/Abijith dataset/ADReSSo21/diagnosis/train/audio/ad/adrso024.wav')
print(y)