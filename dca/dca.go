package dca

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

var (
	STOP      = 1000
	alpha     = 0.4
	beta      = 0.7
	eps       = 1e-20
	lambdaBar = 1.5
)

func DCAlgorithm(xk *mat64.Vector, update func(x *mat64.Vector) *mat64.Vector) (*mat64.Vector, int) {
	iter := 0
	for iter < STOP {
		yk := update(xk)

		dk := mat64.NewVector(xk.RawVector().Inc, nil)
		dk.SubVec(xk, yk)
		if mat64.Dot(dk, dk) < eps {
			return xk, iter
		}

		iter++
		xk = yk
	}
	return xk, iter
}

func BDCAlgorithm(
	xk *mat64.Vector,
	update func(x *mat64.Vector) *mat64.Vector,
	obj func(x *mat64.Vector) float64,
) (*mat64.Vector, int) {
	iter := 0
	for iter < STOP {
		yk := update(xk)

		dk := mat64.NewVector(xk.RawVector().Inc, nil)
		dk.SubVec(yk, xk)
		dkNorm := mat64.Dot(dk, dk)
		if dkNorm < eps {
			return xk, iter
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
			return xk, iter
		}
		iter++
		xk = xkNew
	}
	fmt.Print(iter)
	return xk, iter
}

func BDCAlgorithmQuadratic(
	xk *mat64.Vector,
	update func(x *mat64.Vector) *mat64.Vector,
	obj func(x *mat64.Vector) float64,
	grad func(x *mat64.Vector) *mat64.Vector,
) (*mat64.Vector, int) {
	iter := 0
	for iter < STOP {
		yk := update(xk)

		dk := mat64.NewVector(xk.RawVector().Inc, nil)
		dk.SubVec(yk, xk)
		dkNorm := mat64.Dot(dk, dk)
		if dkNorm < eps {
			return xk, iter
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
			return xk, iter
		}

		iter++
		xk = xkNew
	}
	return xk, iter
}
