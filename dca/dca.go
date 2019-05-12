package dca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

var (
	Iter      = 0
	STOP      = 1000
	alpha     = 0.4
	beta      = 0.7
	eps       = 1e-20
	lambdaBar = 1.5
)

func DCAlgorithm(xk *mat64.Vector, update func(x *mat64.Vector) *mat64.Vector) (*mat64.Vector, int) {
	yk := update(xk)

	dk := mat64.NewVector(xk.RawVector().Inc, nil)
	dk.SubVec(xk, yk)
	if mat64.Dot(dk, dk) < eps || Iter > STOP {
		return xk, Iter
	}

	Iter++
	xk = yk
	return DCAlgorithm(xk, update)
}

func BDCAlgorithm(
	xk *mat64.Vector,
	update func(x *mat64.Vector) *mat64.Vector,
	obj func(x *mat64.Vector) float64,
) (*mat64.Vector, int) {
	yk := update(xk)

	dk := mat64.NewVector(xk.RawVector().Inc, nil)
	dk.SubVec(yk, xk)
	dkNorm := mat64.Dot(dk, dk)
	if dkNorm < eps || Iter > STOP {
		return xk, Iter
	}

	lambda := lambdaBar
	objVal := obj(yk)

	dkNew := mat64.NewVector(xk.RawVector().Inc, nil)
	dkNew.ScaleVec(lambda, dk)

	xkNew := mat64.NewVector(xk.RawVector().Inc, nil)
	xkNew.AddVec(yk, dkNew)

	for obj(xkNew) > objVal-alpha*lambda*dkNorm {
		lambda *= beta
		dkNew.ScaleVec(lambda, dk)
		xkNew.AddVec(yk, dkNew)
	}

	diff := mat64.NewVector(xk.RawVector().Inc, nil)
	diff.SubVec(xk, xkNew)
	if mat64.Dot(diff, diff) < eps {
		return xk, Iter
	}

	Iter++
	return BDCAlgorithm(xkNew, update, obj)
}

func BDCAlgorithmQuadratic(
	xk *mat64.Vector,
	update func(x *mat64.Vector) *mat64.Vector,
	obj func(x *mat64.Vector) float64,
	grad func(x *mat64.Vector) *mat64.Vector,
) (*mat64.Vector, int) {
	yk := update(xk)

	dk := mat64.NewVector(xk.RawVector().Inc, nil)
	dk.SubVec(yk, xk)
	dkNorm := mat64.Dot(dk, dk)
	if dkNorm < eps || Iter > STOP {
		return xk, Iter
	}

	lambda := lambdaBar
	objVal := obj(yk)

	dkNew := mat64.NewVector(xk.RawVector().Inc, nil)
	dkNew.ScaleVec(lambda, dk)

	xkNew := mat64.NewVector(xk.RawVector().Inc, nil)
	xkNew.AddVec(yk, dkNew)

	objLambda := obj(xkNew)
	gradDk := mat64.Dot(grad(yk), dk)
	optLambda := -gradDk * lambda * lambda / (2 * (objLambda - objVal - gradDk*lambda))

	dkNew.ScaleVec(optLambda, dk)
	xkNew.AddVec(yk, dkNew)
	if optLambda > 0 && obj(xkNew) < objLambda {
		lambda = math.Min(lambdaBar, optLambda)
	}

	for obj(xkNew) > objVal-alpha*lambda*dkNorm {
		lambda *= beta
		dkNew.ScaleVec(lambda, dk)
		xkNew.AddVec(yk, dkNew)
	}

	diff := mat64.NewVector(xk.RawVector().Inc, nil)
	diff.SubVec(xk, xkNew)
	if mat64.Dot(diff, diff) < eps {
		return xk, Iter
	}

	Iter++
	return BDCAlgorithmQuadratic(xkNew, update, obj, grad)
}
