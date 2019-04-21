package dca

import "math"

func Dca(xk float64, update func(x float64) float64) float64 {
	yk := 0.0
	yk = update(xk)
	if math.Abs(xk-yk) < 1e-10 {
		return xk
	}
	xk = yk
	return Dca(xk, update)
}
