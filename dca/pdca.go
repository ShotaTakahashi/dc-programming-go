package dca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

var IterTheta = 0

type proximalDCAlgorithmWithExtrapolation struct {
	xk               *mat64.Vector
	xkOld            *mat64.Vector
	yk               *mat64.Vector
	N                int
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
		yk:               x0,
		N:                x0.Len(),
		proximalOperator: proximalOperator,
		subGradient:      subGradient,
		beta:             0.0,
		theta:            1.0,
		thetaOld:         1.0,
	}
}

func (p proximalDCAlgorithmWithExtrapolation) PDCA() *mat64.Vector {
	p.thetaRestart()

	p.beta = (p.thetaOld - 1.0) / p.theta

	theta := (1.0 + math.Pow(1.0+4.0*p.theta*p.theta, 1.0/2.0)) * 0.5
	p.thetaOld = p.theta
	p.theta = theta

	diff := mat64.NewVector(p.N, nil)
	diff.SubVec(p.xk, p.xkOld)
	diff.ScaleVec(p.beta, diff)

	yk := mat64.NewVector(p.N, nil)
	yk.AddVec(yk, diff)

	xi := p.subGradient(p.xk)

	xk := p.proximalOperator(yk, xi)
	diff.SubVec(xk, p.xk)
	if mat64.Norm(diff, 2)/math.Max(1.0, mat64.Norm(xk, 2)) < 1e-5 || Iter > STOP {
		return xk
	}
	p.xkOld = p.xk
	p.xk = xk
	p.yk = yk
	Iter++
	IterTheta++
	return p.PDCA()
}

func (p proximalDCAlgorithmWithExtrapolation) thetaRestart() {

	ykXk := mat64.NewVector(p.N, nil)
	ykXk.SubVec(p.yk, p.xk)

	xkXk := mat64.NewVector(p.N, nil)
	xkXk.SubVec(p.xk, p.xkOld)

	if mat64.Dot(ykXk, xkXk) > 0 || IterTheta > 200 {
		p.theta = 1.0
		p.thetaOld = 1.0
		if IterTheta > 200 {
			IterTheta -= 200
		}
	}
}
