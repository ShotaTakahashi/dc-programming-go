package dca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

type proximalDCAlgorithmWithExtrapolation struct {
	xk               *mat64.Vector
	xkOld            *mat64.Vector
	proximalOperator func(yk, xi *mat64.Vector) *mat64.Vector
	subGradient      func(xk *mat64.Vector) *mat64.Vector
	beta             float64
	theta            float64
	thetaOld         float64
}

func MakeProximalDCAlgorithmWithExtrapolation(
	x0 *mat64.Vector,
	proximalOperator func(yk, xi *mat64.Vector) *mat64.Vector,
	subGradient func(xk *mat64.Vector) *mat64.Vector,
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

func (p proximalDCAlgorithmWithExtrapolation) PDCA() *mat64.Vector {
	N := p.xk.Len()

	yk := mat64.NewVector(N, nil)
	diff := mat64.NewVector(N, nil)
	diff.SubVec(p.xk, p.xkOld)
	diff.ScaleVec(p.beta, diff)
	yk.AddVec(yk, diff)

	xi := p.subGradient(p.xk)

	xk := p.proximalOperator(yk, xi)
	diff.SubVec(xk, p.xk)
	if mat64.Norm(diff, 2)/math.Max(1.0, mat64.Norm(xk, 2)) < 1e-5 {
		return xk
	}
	p.xkOld = p.xk
	p.xk = xk
	return p.PDCA()
}
