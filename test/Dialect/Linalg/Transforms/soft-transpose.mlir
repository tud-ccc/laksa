// RUN: laksa-opt %s --linalg-soft-transpose | FileCheck %s

// CHECK-LABEL: func.func @transpose
// CHECK-NEXT:    %[[CST:.+]] = arith.constant dense<{{\[\[1, 5\], \[2, 6\], \[3, 7\], \[4, 8\]\]}}> : tensor<4x2xi32>
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[CST]] : tensor<4x2xi32>

func.func @transpose() -> tensor<4x2xi32>
{
    %0 = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
    %1 = tensor.empty() : tensor<4x2xi32>
    %2 = linalg.transpose ins(%0: tensor<2x4xi32>) outs(%1: tensor<4x2xi32>) permutation = [1,0]
    func.return %2 : tensor<4x2xi32>
}
