#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x32x32x8xi8>) -> tensor<1x30x30x8xi8> {
    %c127_i32 = arith.constant 127 : i32
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c31_i32 = arith.constant 31 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<40> : tensor<8xi8>
    %cst_0 = arith.constant dense<[1288418624, 1300707578, 1298230215, 1293868576, 1310478957, 1307418510, 1313803263, 1306561467]> : tensor<8xi32>
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %cst_1 = arith.constant dense<"0x3A971B7863DFAEAE6E9FA742A6D9AB301B052A3BF1413FB6EC4E28EFC60FF4B9650E1755841553970E4F814D52CBCCBD43C24EC473D98E2779CE48C0F54C31A48175AF3F539C32C3357144E98D85200A322FD4256F5559B979DB1308187F8B8DB08211798BFF403DD9ABD2E06EF6D5D9B1F28D2CFCCB236F81AA0EC1E03D00C6EE7AC44E00D966650E5B6BDEB4981B5D81F5758D3FB5F5768850CE45368D641D05CE09EC74E79BC06664E43B02C0ECE875D905EFEA9312668195A928174028C07857706CF608ED278BC5341EA7B7D8845AF989176D47750F90054CC0382EF3BE107E33812941AA57AD40B25D72370586299255D91ADF44961F6857A0213292992070342E1DF79DC4052E2CBAEA5D596A02918A19CB9C9AA0F46EF267E901FB5BF079B1FCE4015B7FB5142D5187009705F955FDB7DE90FEDF5482F75AADD055D4647BC2B0682A6420960F569EDF0D65843C9BBDEE111E1AE3D56D51B9F7AD7CDA0FB0EC68ED23A1EA75810211712A40C5E86194701DED4D079F8B0BFDCC02B8497BC4B768427FA666669BE915A4F6E844F49F6D743B4ADDDE6BA7617728528D8E9D329A69BD0FC541D48D8121FE25E512158B5299D4FB1C3EDB6F138182E2A67DDD7277831AA3A4DC8FBA913D3C129BA4A191C593855FCCDED6A0D1094078EAAAC197641D5413DD289387517B8E8EB851B8DC0ABCCA70A7A7FF9C81E67D33A6665D8A174C88E24AF5BF6A1908349E8185B125522DA4BB4F218C842D7BC904F6E31CEE7FACD9E44824A04BDB6F0D3A6734B41C02B04485AA563686A7667F099863"> : tensor<8x3x3x8xi8>
    %cst_2 = arith.constant dense<0> : tensor<8xi32>
    %0 = tensor.empty() : tensor<1x30x30x8xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<8xi32>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x30x30x8xi32>
    %2 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst_1, %c-128_i32, %c0_i32 : tensor<1x32x32x8xi8>, tensor<8x3x3x8xi8>, i32, i32) outs(%1 : tensor<1x30x30x8xi32>) -> tensor<1x30x30x8xi32>
    %3 = tensor.empty() : tensor<1x30x30x8xi8>
    %4 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_0, %cst : tensor<1x30x30x8xi32>, tensor<8xi32>, tensor<8xi8>) outs(%3 : tensor<1x30x30x8xi8>) {
    ^bb0(%in: i32, %in_3: i32, %in_4: i8, %out: i8):
      %5 = arith.extui %in_4 : i8 to i32
      %6 = arith.extsi %in : i32 to i64
      %7 = arith.extsi %in_3 : i32 to i64
      %8 = arith.muli %6, %7 : i64
      %9 = arith.extui %in_4 : i8 to i64
      %10 = arith.shli %c1_i64, %9 : i64
      %11 = arith.shrui %10, %c1_i64 : i64
      %12 = arith.addi %8, %11 : i64
      %13 = arith.cmpi sge, %in, %c0_i32 : i32
      %14 = arith.select %13, %c1073741824_i64, %c-1073741824_i64 : i64
      %15 = arith.addi %14, %12 : i64
      %16 = arith.cmpi sgt, %5, %c31_i32 : i32
      %17 = arith.select %16, %15, %12 : i64
      %18 = arith.shrsi %17, %9 : i64
      %19 = arith.trunci %18 : i64 to i32
      %20 = arith.addi %19, %c-128_i32 : i32
      %21 = arith.maxsi %20, %c-128_i32 : i32
      %22 = arith.minsi %21, %c127_i32 : i32
      %23 = arith.trunci %22 : i32 to i8
      linalg.yield %23 : i8
    } -> tensor<1x30x30x8xi8>
    return %4 : tensor<1x30x30x8xi8>
  }
}

