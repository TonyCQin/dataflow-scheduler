// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s
func.func @bad_named(%a: tensor<4xf16>, %b: tensor<4xf16>, %o: tensor<4xf16>) -> tensor<4xf16> {
  // expected-error @+1 {{V1 only supports add/mul/sub/reduce compute ops; found unsupported compute op}}
  %r = linalg.div ins(%a, %b : tensor<4xf16>, tensor<4xf16>) outs(%o : tensor<4xf16>) -> tensor<4xf16>
  return %r : tensor<4xf16>
}
