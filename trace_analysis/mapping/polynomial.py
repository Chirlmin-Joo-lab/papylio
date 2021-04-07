from skimage.transform._geometric import PolynomialTransform

class PolynomialTransform(PolynomialTransform):
    def estimate(self, src, dst, order=4):
        return super().estimate(src, dst, order)
