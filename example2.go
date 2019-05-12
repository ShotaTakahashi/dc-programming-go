package main

import (
	"dc-programming-go/dca"
	"fmt"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"log"
	"math/rand"
	"time"
)

var (
	eig mat64.EigenSym

	i        = 1
	N        = 256 * i
	M        = 72 * i
	SPARSITY = 8 * i
	LAMBDA   = 1e-3
	EPS      = 1.0
	A        = makeMatrixA()
	b        = makeVectorB()
	L        = maxEigenvalue()
)

func main() {
	x0 := mat64.NewVector(N, nil)
	problem := dca.MakeProximalDCAlgorithmWithExtrapolation(x0, proximalOperator, subGradient)

	t0 := time.Now()
	opt := problem.PDCA()
	fmt.Println(time.Since(t0), opt)
}

func proximalOperator(yk, xi *mat64.Vector) *mat64.Vector {
	xk := mat64.NewVector(N, nil)
	prox := mat64.NewVector(N, nil)
	prox.ScaleVec(L, yk)
	prox.AddVec(prox, xi)

	gradLeastSquare := mat64.NewVector(N, nil)
	temp := mat64.NewVector(M, nil)
	temp.MulVec(A, yk)
	temp.SubVec(temp, b)
	gradLeastSquare.MulVec(A.T(), temp)
	prox.SubVec(prox, gradLeastSquare)
	LambdaEps := LAMBDA / EPS
	for i := 0; i < xk.Len(); i++ {
		if prox.At(i, 0) > LambdaEps {
			xk.SetVec(i, prox.At(i, 0)-LambdaEps)
		} else if prox.At(i, 0) < LambdaEps {
			xk.SetVec(i, prox.At(i, 0)+LambdaEps)
		}
	}
	xk.ScaleVec(1/L, xk)
	return xk
}

func subGradient(x *mat64.Vector) *mat64.Vector {
	xNorm := mat64.Norm(x, 2)
	if xNorm == 0 {
		return mat64.NewVector(N, make([]float64, N))
	}
	subGradient := mat64.NewVector(N, nil)
	subGradient.ScaleVec(LAMBDA/xNorm, x)
	return subGradient
}

func makeMatrixA() *mat64.Dense {
	data := make([]float64, N*M)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	A := mat64.NewDense(M, N, data)
	return A
}

func makeVectorB() *mat64.Vector {
	b := mat64.NewVector(M, nil)
	data := make([]float64, N)
	for _, i := range sampling(N) {
		data[i] = rand.NormFloat64()
	}
	y := mat64.NewVector(N, data)
	b.MulVec(A, y)
	b.AddVec(b, makeRandomVector(M))
	return b
}

func makeRandomVector(size int) *mat64.Vector {
	data := make([]float64, size)
	rand.Seed(42)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	result := mat64.NewVector(size, data)
	result.ScaleVec(0.01, result)
	return result
}

func sampling(size int) []int {
	var sample []int
	for i := 0; i < size; i++ {
		sample = append(sample, i)
	}
	rand.Shuffle(size, func(i, j int) { sample[i], sample[j] = sample[j], sample[i] })
	sample = sample[:SPARSITY]
	return sample
}

func maxEigenvalue() float64 {
	data := mat64.NewDense(N, N, nil)
	data.Mul(A.T(), A)
	AtA := mat64.NewSymDense(N, data.RawMatrix().Data)
	ok := eig.Factorize(AtA, true)
	if !ok {
		log.Fatal("Eigendecomposition failed")
	}
	return floats.Max(eig.Values(nil))
}
