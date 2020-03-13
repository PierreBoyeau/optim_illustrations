import pywt

from proximal_prior import ProximalPower


class Denoiser:
    def __init__(self, wavelet, level, q, chi):
        self.img2wt = lambda data: pywt.wavedec2(data, wavelet=wavelet, level=level,
                                                 axes=(0, 1))
        self.wt2img = lambda wt: pywt.waverec2(wt, wavelet=wavelet, axes=(0, 1))
        self.proximal_power = ProximalPower(q=q, chi=chi)
        self.chi = chi

    def denoise(self, img):
        noise_wt = self.img2wt(img)

        # Proximal computation
        for level in range(1, len(noise_wt)):
            prox_0 = self.proximal_power.compute(noise_wt[level][0])
            prox_1 = self.proximal_power.compute(noise_wt[level][1])
            prox_2 = self.proximal_power.compute(noise_wt[level][2])

            noise_wt[level] = (prox_0, prox_1, prox_2)
        # Inverse wavelet transform
        return self.wt2img(wt=noise_wt)


if __name__ == '__main__':
    from PIL import Image
    import numpy as np


    def load_image(infilename):
        img = Image.open(infilename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data


    img = load_image('florence.jpg')

    # Adding gaussian noise to the image
    noisy_img = img + 30 * np.random.normal(size=img.shape)
    noisy_img = np.clip(noisy_img, a_min=0.0, a_max=255).astype(np.int32)

    WAVELET = "db8"
    LEVEL = 4

    denoiser = Denoiser(wavelet=WAVELET, level=LEVEL, q=1, chi=10)
    denoised_image = denoiser.denoise(noisy_img)
