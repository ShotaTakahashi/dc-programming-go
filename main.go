package main

import (
	"dc-programming-go/dca"
	"fmt"
	"math"
	"time"
)

func main() {
	var x0 = 27.0 / 125.0

	t0 := time.Now()
	opt, iter := dca.DCA(x0, update)
	fmt.Println(time.Since(t0), opt, iter)

	t0 = time.Now()
	opt, iter = dca.BDCA(x0, update, obj)
	fmt.Println(time.Since(t0), opt, iter)

	t0 = time.Now()
	opt, iter = dca.BDCAQuadratic(x0, update, obj, grad)
	fmt.Println(time.Since(t0), opt, iter)
}

func update(x float64) float64 {
	return math.Pow(x, 1.0/3.0)
}

func obj(x float64) float64 {
	return math.Pow(x, 4)/4.0 - math.Pow(x, 2)/2.0
}

func grad(x float64) float64 {
	return math.Pow(x, 3) - x
}