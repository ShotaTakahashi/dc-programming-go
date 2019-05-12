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
	dca.Iter = 0
	x0 := mat64.NewVector(N, nil)
	problem := dca.MakeProximalDCAlgorithmWithExtrapolation(x0, proximalOperator, subGradient)
	fmt.Println(L)
	t0 := time.Now()
	opt := problem.PDCA()
	fmt.Println(time.Since(t0), objectFunc(opt), dca.Iter)
}

func proximalOperator(yk, xi *mat64.Vector) *mat64.Vector {
	xk := mat64.NewVector(N, nil)
	proximal := mat64.NewVector(N, nil)
	proximal.ScaleVec(L, yk)
	proximal.AddVec(proximal, xi)

	gradLeastSquare := mat64.NewVector(N, nil)
	Ay := mat64.NewVector(M, nil)
	Ay.MulVec(A, yk)
	Ay.SubVec(Ay, b)
	gradLeastSquare.MulVec(A.T(), Ay)
	proximal.SubVec(proximal, gradLeastSquare)
	LambdaEps := LAMBDA / EPS
	for i := 0; i < N; i++ {
		if proximal.At(i, 0) > LambdaEps {
			xk.SetVec(i, proximal.At(i, 0)-LambdaEps)
		} else if proximal.At(i, 0) < LambdaEps {
			xk.SetVec(i, proximal.At(i, 0)+LambdaEps)
		}
	}
	xk.ScaleVec(1/L, xk)
	return xk
}

func subGradient(x *mat64.Vector) *mat64.Vector {
	xNorm := mat64.Norm(x, 2)
	if xNorm == 0 {
		return x
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
	for j := 0; j < N; j++ {
		norm := mat64.Norm(A.ColView(j), 2)
		for i := 0; i < M; i++ {
			A.Set(i, j, A.At(i, j)/norm)
		}
	}
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

func objectFunc(x *mat64.Vector) float64 {
	leastSquare := mat64.NewVector(M, nil)
	leastSquare.MulVec(A, x)
	leastSquare.SubVec(leastSquare, b)
	return mat64.Dot(leastSquare, leastSquare)*0.5 + LAMBDA*(mat64.Norm(x, 1)-mat64.Norm(x, 2))
}
