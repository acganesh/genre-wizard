genre-wizard
============
 
A machine learning system to classify music genres from audio.  This code was written years ago, so it is very messy and out of date.  Still, I'd like to expand on these ideas eventually.

This implementation uses [mel-frequency cepstrum coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) fed into an SVM for classification.  Broadly, the algorithm works as follows (from `SciPy`'s feature toolbox):

- Take the Fourier transform of a signal.
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0d2aab0c0d32f0438d2ccf5bf779458053ba2bd9)

- Transform the spectra onto the [mel scale](https://en.wikipedia.org/wiki/Mel_scale), due to Stevens, Volkmann, and Newman in 1937.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2e8a48e66fa73f33901e824ceb1ad6009007ffda)

- Take logarithms of the powers at each mel frequencies.

- Take the discrete cosine transform of the list of mel log powers.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/be8dacb1e78120e504f6fa9d98757c5fc1cd8f89)

- The MFCCs are the amplitudes of the resulting spectrum.


# Goals

- Try to understand the "essence" of what a musical genre means.
- Try to understand the math and physics behind music.

# Todo

- Explore Fourier analysis to model sound.
- Explore the connections between Fourier analysis and ConvNets.
- Potentially: explore Haskell implementations of GLMs or deep learning models.
- Potentially: make this code nice.
