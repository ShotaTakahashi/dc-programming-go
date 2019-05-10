package dca

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

var (
	Iter      = 0
	STOP      = 100
	alpha     = 0.4
	beta      = 0.7
	eps       = 1e-20
	lambdaBar = 1.5
)

func DCAlgorithm(xk *mat64.Vector, update func(x *mat64.Vector) *mat64.Vector) (*mat64.Vector, int) {
	yk := update(xk)

	diff := mat64.NewVector(xk.RawVector().Inc, nil)
	diff.SubVec(xk, yk)
	if mat64.Dot(diff, diff) < eps || Iter > STOP {
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
	dk.SubVec(xk, yk)
	dkNorm := mat64.Dot(dk, dk)
	if dkNorm < eps || Iter > STOP {
		return xk, Iter
	}

	lambda := lambdaBar
	objVal := obj(yk)
	newYk := mat64.NewVector(xk.RawVector().Inc, nil)
	newDk := mat64.NewVector(xk.RawVector().Inc, nil)
	newDk.ScaleVec(lambda, dk)
	newYk.AddVec(yk, newDk)
	for obj(newYk) > objVal-alpha*lambda*dkNorm {
		lambda *= beta
		newDk.ScaleVec(lambda, dk)
		newYk.AddVec(yk, newDk)
	}
	dk.ScaleVec(lambda, dk)
	yk.AddVec(yk, dk)
	diff := mat64.NewVector(xk.RawVector().Inc, nil)
	diff.SubVec(xk, yk)
	if mat64.Norm(diff, 2) < eps {
		return xk, Iter
	}

	Iter++
	return BDCAlgorithm(yk, update, obj)
}

func BDCAlgorithmQuadratic(
	xk *mat64.Vector,
	update func(x *mat64.Vector) *mat64.Vector,
	obj func(x *mat64.Vector) float64,
	grad func(x *mat64.Vector) *mat64.Vector,
) (*mat64.Vector, int) {
	yk := update(xk)

	dk := mat64.NewVector(xk.RawVector().Inc, nil)
	dk.SubVec(xk, yk)
	dkNorm := mat64.Dot(dk, dk)
	if dkNorm < eps || Iter > STOP {
		return xk, Iter
	}

	lambda := lambdaBar
	newYk := mat64.NewVector(xk.RawVector().Inc, nil)
	newDk := mat64.NewVector(xk.RawVector().Inc, nil)
	newDk.ScaleVec(lambda, dk)
	newYk.AddVec(yk, newDk)
	objVal := obj(yk)
	objLambda := obj(newYk)
	gradDk := mat64.Dot(grad(yk), dk)
	optLambda := -gradDk * lambda * lambda / (2 * (objLambda - objVal - gradDk*lambda))

	newDk.ScaleVec(optLambda, dk)
	newYk.AddVec(yk, newDk)
	if optLambda > 0 && obj(newYk) < objLambda {
		lambda = math.Min(lambdaBar, optLambda)
	}

	for obj(newYk) > objVal-alpha*lambda*dkNorm {
		lambda *= beta
		newDk.ScaleVec(lambda, dk)
		newYk.AddVec(yk, newDk)
	}
	dk.ScaleVec(lambda, dk)
	yk.AddVec(yk, dk)
	diff := mat64.NewVector(xk.RawVector().Inc, nil)
	diff.SubVec(xk, yk)
	if mat64.Dot(diff, diff) < eps {
		return xk, Iter
	}

	Iter++
	return BDCAlgorithmQuadratic(yk, update, obj, grad)
}
