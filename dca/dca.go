package dca

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

var (
	Iter      = 0
	STOP      = 100
	alpha     = 0.4
	beta      = 0.7
	eps       = 1e-10
	lambdaBar = 1.5
)

func DCAlgorithm(xk *mat64.Dense, update func(x *mat64.Dense) *mat64.Dense) (*mat64.Dense, int) {
	yk := update(xk)

	diff := mat64.NewDense(xk.RawMatrix().Rows, xk.RawMatrix().Cols, nil)
	diff.Sub(xk, yk)
	if mat64.Norm(diff, 2) < eps || Iter > STOP {
		return xk, Iter
	}

	Iter++
	xk = yk
	return DCAlgorithm(xk, update)
}

func BDCAlgorithm(
	xk *mat64.Dense,
	update func(x *mat64.Dense) *mat64.Dense,
	obj func(x *mat64.Dense) float64,
) (*mat64.Dense, int) {
	yk := update(xk)

	dk := mat64.NewDense(xk.RawMatrix().Rows, xk.RawMatrix().Cols, nil)
	dk.Sub(xk, yk)
	dkNorm := mat64.Norm(dk, 2)
	if dkNorm < eps || Iter > STOP {
		return xk, Iter
	}

	lambda := lambdaBar
	objVal := obj(yk)
	newYk := mat64.NewDense(xk.RawMatrix().Rows, xk.RawMatrix().Cols, nil)
	newDk := mat64.NewDense(xk.RawMatrix().Rows, xk.RawMatrix().Cols, nil)
	newDk.Scale(lambda, dk)
	newYk.Add(yk, newDk)
	for obj(newYk) > objVal-alpha*lambda*dkNorm {
		lambda *= beta
		newDk.Scale(lambda, dk)
		newYk.Add(yk, newDk)
	}
	dk.Scale(lambda, dk)
	yk.Add(yk, dk)
	diff := mat64.NewDense(xk.RawMatrix().Rows, xk.RawMatrix().Cols, nil)
	diff.Sub(xk, yk)
	if mat64.Norm(diff, 2) < eps {
		return xk, Iter
	}

	Iter++
	xk = yk
	return BDCAlgorithm(xk, update, obj)
}

func BDCAlgorithmQuadratic(
	xk float64,
	update func(x float64) float64,
	obj func(x float64) float64,
	grad func(x float64) float64,
) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < eps {
		return xk, Iter
	}

	lambda := lambdaBar
	dk := yk - xk
	objVal := obj(yk)
	objLambda := obj(yk + lambda*dk)
	gradDk := grad(yk) * dk
	optLambda := -gradDk * lambda * lambda / (2 * (objLambda - objVal - gradDk*lambda))

	if optLambda > 0 && obj(yk+optLambda*dk) < objLambda {
		lambda = math.Min(lambdaBar, optLambda)
	}

	for obj(yk+lambda*dk) > objVal-alpha*lambda*dk*dk {
		lambda *= beta
	}

	yk = yk + lambda*dk
	if math.Abs(xk-yk) < eps {
		return xk, Iter
	}

	Iter++
	xk = yk
	return BDCAlgorithmQuadratic(xk, update, obj, grad)
}
