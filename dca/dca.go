package dca

import (
	"math"
)

var iter = 0
var alpha = 0.4
var beta = 0.7

func DCA(xk float64, update func(x float64) float64) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < 1e-10 {
		return xk, iter
	}
	iter++
	xk = yk
	return DCA(xk, update)
}

func BDCA(
	xk float64,
	update func(x float64) float64,
	obj func(x float64) float64,
) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < 1e-10 {
		return xk, iter
	}

	lambda := 1.5
	dk := yk - xk
	objVal := obj(yk)
	for obj(yk+lambda*dk) > objVal-alpha*lambda*dk*dk {
		lambda *= beta
	}

	yk = yk + lambda*dk
	if math.Abs(xk-yk) < 1e-10 {
		return xk, iter
	}
	xk = yk
	iter++
	return BDCA(xk, update, obj)
}

func BDCAQuadratic(
	xk float64,
	update func(x float64) float64,
	obj func(x float64) float64,
	grad func(x float64) float64,
) (float64, int) {
	yk := update(xk)
	if math.Abs(xk-yk) < 1e-10 {
		return xk, iter
	}

	lambda := 1.5
	dk := yk - xk
	objVal := obj(yk)
	objLambda := obj(yk + lambda*dk)
	gradDk := grad(yk) * dk
	optLambda := -gradDk * lambda * lambda / (2 * (objLambda - objVal - gradDk*lambda))

	if optLambda > 0 && obj(yk+optLambda*dk) < objLambda {
		lambda = math.Min(1.5, optLambda)
	}

	for obj(yk+lambda*dk) > objVal-alpha*lambda*dk*dk {
		lambda *= beta
	}

	yk = yk + lambda*dk
	if math.Abs(xk-yk) < 1e-10 {
		return xk, iter
	}
	xk = yk
	iter++
	return BDCAQuadratic(xk, update, obj, grad)
}
