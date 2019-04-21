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
	solution := dca.Dca(
		x0,
		func(x float64) float64 {
			return math.Pow(x, 1.0/3.0)
		},
	)
	fmt.Println(time.Since(t0), solution)
}
