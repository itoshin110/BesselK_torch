import torch
torch.set_default_dtype(torch.float64)

class BesselK:
    """
    Modified Bessel functions of the second kind:
      - k0(x), k1(x)
      - kn(n, x) for integer n >= 0 via upward recurrence

    Notes
    -----
    * Small-x (x <= x_switch):
        K0: exact series  K0 = -(log(x/2)+γ) I0 + Σ_{k>=1} H_k (x/2)^{2k}/(k!)^2
        K1: exact series  K1 = 1/x + (log(x/2)+γ) I1
                           - (1/2) Σ_{k>=0} (H_k + H_{k+1}) (x/2)^{2k+1}/(k!(k+1)!)
    * Large-x (x > x_switch):
        Rational approximations in the form
        Kν(x) ≈ exp(-x)/sqrt(x) * (P(1/x)/Q(1/x))
    * Autograd friendly; works on CPU/GPU; dtype=float64
    """
    EULER_GAMMA = 0.5772156649015328606

    # --- Large-x coefficients (double-grade) ---
    # K0: x^{1/2} e^x K0 ≈ P21(1/x)/Q2(1/x)
    K0_P21 = [
        1.0694678222191263215918328e-01,  9.0753360415683846760792445e-01,
        1.7215172959695072045669045e+00, -1.7172089076875257095489749e-01,
        7.3154750356991229825958019e-02, -5.4975286232097852780866385e-02,
        5.7217703802970844746230694e-02, -7.2884177844363453190380429e-02,
        1.0443967655783544973080767e-01, -1.5741597553317349976818516e-01,
        2.3582486699296814538802637e-01, -3.3484166783257765115562496e-01,
        4.3328524890855568555069622e-01, -4.9470375304462431447923425e-01,
        4.8474122247422388055091847e-01, -3.9725799556374477699937953e-01,
        2.6507653322930767914034592e-01, -1.3951265948137254924254912e-01,
        5.5500667358490463548729700e-02, -1.5636955694760495736676521e-02,
        2.7741514506299244078981715e-03, -2.3261089001545715929104236e-04,
    ]
    K0_Q2  = [8.5331186362410449871043129e-02, 7.3477344946182065340442326e-01, 1.4594189037511445958046540e+00]

    # K1: x^{1/2} e^x K1 ≈ P22(1/x)/Q2(1/x)
    K1_P22 = [
        1.0234817795732426171122752e-01,  9.4576473594736724815742878e-01,
        2.1876721356881381470401990e+00,  6.0143447861316538915034873e-01,
       -1.3961391456741388991743381e-01,  8.8229427272346799004782764e-02,
       -8.5494054051512748665954180e-02,  1.0617946033429943924055318e-01,
       -1.5284482951051872048173726e-01,  2.3707700686462639842005570e-01,
       -3.7345723872158017497895685e-01,  5.6874783855986054797640277e-01,
       -8.0418742944483208700659463e-01,  1.0215105768084562101457969e+00,
       -1.1342221242815914077805587e+00,  1.0746932686976675016706662e+00,
       -8.4904532475797772009120500e-01,  5.4542251056566299656460363e-01,
       -2.7630896752209862007904214e-01,  1.0585982409547307546052147e-01,
       -2.8751691985417886721803220e-02,  4.9233441525877381700355793e-03,
       -3.9900679319457222207987456e-04,
    ]
    K1_Q2  = [8.1662031018453173425764707e-02, 7.2398781933228355889996920e-01, 1.4835841581744134589980018e+00]

    def __init__(self, series_terms: int = 16, x_switch: float = 1.0):
        """
        Parameters
        ----------
        series_terms : int
            Number of terms for the small-x series (>= 12 recommended).
        x_switch : float
            Threshold to switch between small-x series and large-x rational approx.
        """
        self.N = int(series_terms)
        self.x_switch = float(x_switch)

    # -------- utilities --------
    @staticmethod
    def _horner(p, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        for c in reversed(p):
            y = y * x + x.new_tensor(c)
        return y

    @staticmethod
    def _safe_x(x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        finfo = torch.finfo(x.dtype)
        return torch.clamp(x, min=finfo.tiny)

    # -------- I0, I1 series (for small-x) --------
    def _i0_series(self, x: torch.Tensor) -> torch.Tensor:
        t = (x * 0.5) ** 2
        term = torch.ones_like(x)
        s = term.clone()
        for k in range(1, self.N + 1):
            term = term * t / (k * k)
            s = s + term
        return s

    def _i1_series(self, x: torch.Tensor) -> torch.Tensor:
        # I1(x) = Σ_{k>=0} (x/2)^{2k+1}/(k!(k+1)!)
        acc = 0.5 * x   # k=0
        s = acc.clone()
        u = (x * 0.5) ** 2
        for k in range(1, self.N + 1):
            acc = acc * u / (k * (k + 1))
            s = s + acc
        return s

    # -------- small-x series for K0, K1 --------
    def _k0_small(self, x: torch.Tensor) -> torch.Tensor:
        logx2 = torch.log(x * 0.5)
        I0 = self._i0_series(x)
        t = (x * 0.5) ** 2
        acc = torch.ones_like(x)  # (x/2)^{2k}/(k!)^2
        series = torch.zeros_like(x)
        for k in range(1, self.N + 1):
            acc = acc * t / (k * k)
            Hk = torch.digamma(torch.tensor(float(k + 1), dtype=x.dtype, device=x.device)) + self.EULER_GAMMA
            series = series + Hk * acc
        return -(logx2 + self.EULER_GAMMA) * I0 + series

    def _k1_small(self, x: torch.Tensor) -> torch.Tensor:
        # K1 = 1/x + (log(x/2)+γ) I1  - (1/2) Σ (H_k + H_{k+1}) (x/2)^{2k+1}/(k!(k+1)!)
        invx = 1.0 / x
        I1 = self._i1_series(x)
        logx2 = torch.log(x * 0.5)

        u = (x * 0.5) ** 2
        acc = 0.5 * x   # k=0
        # k=0 : -(1/2)*(H_0 + H_1) = -1/2
        corr = (-0.5) * (0.0 + 1.0) * acc
        for k in range(1, self.N + 1):
            acc = acc * u / (k * (k + 1))
            Hk   = torch.digamma(torch.tensor(float(k + 1), dtype=x.dtype, device=x.device)) + self.EULER_GAMMA
            Hkp1 = torch.digamma(torch.tensor(float(k + 2), dtype=x.dtype, device=x.device)) + self.EULER_GAMMA
            corr = corr + (-0.5) * (Hk + Hkp1) * acc

        return invx + (logx2 + self.EULER_GAMMA) * I1 + corr

    # -------- public: K0, K1, Kn --------
    def k0(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        x_safe = self._safe_x(x)
        small = x_safe <= self.x_switch
        y = torch.empty_like(x_safe)

        xs = x_safe[small]
        if xs.numel() > 0:
            y[small] = self._k0_small(xs)

        xl = x_safe[~small]
        if xl.numel() > 0:
            u = 1.0 / xl
            num = self._horner(self.K0_P21, u)
            den = self._horner(self.K0_Q2,  u)
            y[~small] = torch.exp(-xl) * (num / den) / torch.sqrt(xl)
        return y

    def k1(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        x_safe = self._safe_x(x)
        small = x_safe <= self.x_switch
        y = torch.empty_like(x_safe)

        xs = x_safe[small]
        if xs.numel() > 0:
            y[small] = self._k1_small(xs)

        xl = x_safe[~small]
        if xl.numel() > 0:
            u = 1.0 / xl
            num = self._horner(self.K1_P22, u)
            den = self._horner(self.K1_Q2,  u)
            y[~small] = torch.exp(-xl) * (num / den) / torch.sqrt(xl)
        return y

    def kn(self, n: int, x: torch.Tensor) -> torch.Tensor:
        if n < 0:
            raise ValueError("n must be >= 0.")
        x = torch.as_tensor(x)
        if n == 0:
            return self.k0(x)
        if n == 1:
            return self.k1(x)

        x_safe = self._safe_x(x)
        km1 = self.k0(x)
        kcur = self.k1(x)
        for m in range(1, n):
            knext = km1 + (2.0 * m / x_safe) * kcur
            km1, kcur = kcur, knext
        return kcur

import torch, numpy as np
from scipy.special import kv

xmin = 0.01
xmax = 1e2

numpoints = 10000

x = torch.linspace(xmin, xmax, numpoints, dtype=torch.float64, device='cpu')

for n in range(10):
    y = BesselK().kn(n, x)
    y_ref = torch.from_numpy(kv(n, x.numpy()))
    error = (y - y_ref)/y_ref
    print("K{:d}, max relative error: {:.8e}".format(n,error.abs().max().item()))

