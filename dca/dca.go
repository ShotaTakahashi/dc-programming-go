package dca

import (
	"math"
)

var (
	Iter      = 0
	alpha     = 0.4
	beta      = 0.7
	eps       = 1e-10
	lambdaBar = 1.5
)

func DCAlgorithm(xk float64, update func(x float64) float64) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < eps {
		return xk, Iter
	}

	Iter++
	xk = yk
	return DCAlgorithm(xk, update)
}

func BDCAlgorithm(
	xk float64,
	update func(x float64) float64,
	obj func(x float64) float64,
) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < eps {
		return xk, Iter
	}

	lambda := lambdaBar
	dk := yk - xk
	objVal := obj(yk)

	for obj(yk+lambda*dk) > objVal-alpha*lambda*dk*dk {
		lambda *= beta
	}

	yk = yk + lambda*dk
	if math.Abs(xk-yk) < eps {
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
