package main

import (
	"dc-programming-go/dca"
	"fmt"
	"math"
	"time"

	"github.com/gonum/matrix/mat64"
)

func main() {
	x0Elm := []float64{27.0 / 125.0}
	x0 := mat64.NewDense(1, 1, x0Elm)

	dca.Iter = 0
	t0 := time.Now()
	opt, iter := dca.DCAlgorithm(x0, update)
	fmt.Println(time.Since(t0), opt, iter)

	t0 = time.Now()
	dca.Iter = 0
	opt, iter = dca.BDCAlgorithm(x0, update, obj)
	fmt.Println(time.Since(t0), opt, iter)

	dca.Iter = 0
	t0 = time.Now()
	//opt, iter = dca.BDCAlgorithmQuadratic(x0, update, obj, grad)
	fmt.Println(time.Since(t0), opt, iter)
}

func update(x *mat64.Dense) *mat64.Dense {
	resultElm := make([]float64, 1)
	result := mat64.NewDense(1, 1, resultElm)
	result.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, 1.0/3.0)
	}, x)
	return result
}

func obj(x *mat64.Dense) float64 {
	xElm := x.RawMatrix().Data
	x0 := xElm[0]
	return math.Pow(x0, 4)/4.0 - math.Pow(x0, 2)/2.0
}

func grad(x float64) float64 {
	return math.Pow(x, 3) - x
}
