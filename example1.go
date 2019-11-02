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
	x0 := mat64.NewVector(1, x0Elm)

	t0 := time.Now()
	opt, iter := dca.DCAlgorithm(x0, update)
	fmt.Println(time.Since(t0), opt, iter)

	t0 = time.Now()
	opt, iter = dca.BDCAlgorithm(x0, update, obj)
	fmt.Println(time.Since(t0), opt, iter)

	t0 = time.Now()
	opt, iter = dca.BDCAlgorithmQuadratic(x0, update, obj, grad)
	fmt.Println(time.Since(t0), opt, iter)
}

func update(x *mat64.Vector) *mat64.Vector {
	result := mat64.NewVector(1, nil)
	for i := 0; i < x.Len(); i++ {
		result.SetVec(i, math.Pow(x.At(i, 0), 1.0/3.0))
	}
	return result
}

func obj(x *mat64.Vector) float64 {
	xElm := x.RawVector().Data[0]
	return math.Pow(xElm, 4)/4.0 - math.Pow(xElm, 2)/2.0
}

func grad(x *mat64.Vector) *mat64.Vector {
	result := mat64.NewVector(1, nil)
	result.MulElemVec(x, x)
	result.MulElemVec(result, x)
	result.SubVec(result, x)
	return result
}
