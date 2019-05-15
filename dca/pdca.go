package dca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

var (
	StopPDCA = 1000
)

type proximalDCAlgorithmWithExtrapolation struct {
	xk               *mat64.Vector
	xkOld            *mat64.Vector
	yk               *mat64.Vector
	proximalOperator func(yk, xi *mat64.Vector) *mat64.Vector
	subGradient      func(xk *mat64.Vector) *mat64.Vector
	beta             float64
	theta            float64
	thetaOld         float64
	iter             int
	iterTheta        int
}

func MakeProximalDCAlgorithmWithExtrapolation(
	x0 *mat64.Vector,
	proximalOperator func(yk, xi *mat64.Vector) *mat64.Vector,
	subGradient func(xk *mat64.Vector) *mat64.Vector,
) *proximalDCAlgorithmWithExtrapolation {
	return &proximalDCAlgorithmWithExtrapolation{
		xk:               x0,
		xkOld:            x0,
		yk:               x0,
		proximalOperator: proximalOperator,
		subGradient:      subGradient,
		beta:             0.0,
		theta:            1.0,
		thetaOld:         1.0,
		iter:             0,
		iterTheta:        0,
	}
}

func (p proximalDCAlgorithmWithExtrapolation) PDCA() (*mat64.Vector, int) {
	p.betaUpdate()

	diff := mat64.NewVector(p.xk.Len(), nil)
	diff.SubVec(p.xk, p.xkOld)
	diff.ScaleVec(p.beta, diff)

	yk := mat64.NewVector(p.xk.Len(), nil)
	yk.AddVec(p.xk, diff)

	xi := p.subGradient(p.xk)

	xk := p.proximalOperator(yk, xi)

	diff.SubVec(xk, p.xk)
	if mat64.Norm(diff, 2)/math.Max(1.0, mat64.Norm(xk, 2)) < 1e-5 || p.iter > StopPDCA {
		return xk, p.iter
	}
	p.xkOld = p.xk
	p.xk = xk
	p.yk = yk
	p.iter++
	p.iterTheta++
	return p.PDCA()
}

func (p *proximalDCAlgorithmWithExtrapolation) betaUpdate() {
	p.thetaRestart()
	p.beta = (p.thetaOld - 1.0) / p.theta
	p.thetaUpdate()
}

func (p *proximalDCAlgorithmWithExtrapolation) thetaUpdate() {
	theta := (1.0 + math.Pow(1.0+4.0*p.theta*p.theta, 0.5)) * 0.5
	p.thetaOld = p.theta
	p.theta = theta
}

func (p *proximalDCAlgorithmWithExtrapolation) thetaRestart() {
	yk := mat64.NewVector(p.xk.Len(), nil)
	yk.SubVec(p.yk, p.xk)

	xk := mat64.NewVector(p.xk.Len(), nil)
	xk.SubVec(p.xk, p.xkOld)

	if mat64.Dot(yk, xk) > 0 || p.iter > 200 {
		p.theta = 1.0
		p.thetaOld = 1.0
		if p.iterTheta > 200 {
			p.iterTheta -= 200
		}
	}
}
