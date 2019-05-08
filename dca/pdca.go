package dca

import "math"

type proximalDCAlgorithmWithExtrapolation struct {
	xk               float64
	xkOld            float64
	proximalOperator func(yk, xi float64) float64
	subGradient      func(xk float64) float64
	beta             float64
	theta            float64
	thetaOld         float64
}

func MakeProximalDCAlgorithmWithExtrapolation(
	x0 float64,
	proximalOperator func(yk, xi float64) float64,
	subGradient func(xk float64) float64,
) proximalDCAlgorithmWithExtrapolation {
	return proximalDCAlgorithmWithExtrapolation{
		xk:               x0,
		xkOld:            x0,
		proximalOperator: proximalOperator,
		subGradient:      subGradient,
		beta:             1.0,
		theta:            1.0,
		thetaOld:         1.0,
	}
}

func (p proximalDCAlgorithmWithExtrapolation) pDCA() float64 {
	yk := p.xk + p.beta*(p.xk-p.xkOld)
	xi := p.subGradient(p.xk)
	xk := p.proximalOperator(yk, xi)
	if math.Abs(xk-p.xk)/math.Max(1.0, math.Abs(xk)) < 1e-5 {
		return xk
	}
	p.xkOld = p.xk
	p.xk = xk
	return p.pDCA()
}
