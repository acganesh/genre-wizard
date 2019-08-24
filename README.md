fourier
============
 
A machine learning system to classify music genres from audio.  This code was written in 2014, so it is messy and out of date.

**Reflection (Aug. 2019).** Looking back on this, it's interesting that 1-D ConvNets / RNNs utterly crush this clever technique for the same classification task.  As Rich Sutton noted in ["The Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), compute triumphs over cleverness.

The pessimistic view is that clever researchers have low marginal utility.  But the optimistic view suggests that deep learning is here to stay.  Since compute will likely continue to accelerate, AI applications will gradually become more accessible to software engineers and the world more broadly.

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
