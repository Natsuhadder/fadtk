import librosa
import numpy as np

y = np.arange(82000)
a = librosa.util.fix_length(y, size=131072)

print(y.shape)
print(a.shape)
