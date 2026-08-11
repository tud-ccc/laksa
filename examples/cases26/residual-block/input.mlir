#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x32x32x8xi8> {ml_program.identifier = "serving_default_keras_tensor:0", tf_saved_model.index_path = ["keras_tensor"]}) -> (tensor<1x28x28x8xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %c281474976710656_i64 = arith.constant 281474976710656 : i64
    %c49_i64 = arith.constant 49 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c11_i64 = arith.constant 11 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2147483648_i64 = arith.constant 2147483648 : i64
    %c1633833687_i64 = arith.constant 1633833687 : i64
    %c512_i64 = arith.constant 512 : i64
    %c10_i64 = arith.constant 10 : i64
    %cst = arith.constant dense<[39, 39, 39, 40, 40, 39, 40, 40]> : tensor<8xi8>
    %cst_0 = arith.constant dense<[1079388899, 1080650220, 1080683627, 2123794270, 2107169143, 1079132406, 2033284180, 2112265834]> : tensor<8xi32>
    %cst_1 = arith.constant dense<40> : tensor<8xi8>
    %cst_2 = arith.constant dense<[1241604906, 1329922038, 1341576575, 1322904542, 1326948123, 1339807398, 1325877483, 1340943676]> : tensor<8xi32>
    %c127_i32 = arith.constant 127 : i32
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c31_i32 = arith.constant 31 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst_3 = arith.constant dense<41> : tensor<8xi8>
    %cst_4 = arith.constant dense<[1523322938, 1544671534, 1547120170, 1546480699, 1536582660, 1557469766, 1525058969, 1526555847]> : tensor<8xi32>
    %c-128_i8 = arith.constant -128 : i8
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %cst_5 = arith.constant dense<0> : tensor<8xi32>
    %cst_6 = arith.constant dense<"0x9F9F3BCAF4377099E429251884D61EA18C8199B9064B9D317A7F96A5E7FB44171EC7E93624713912363A8ECF112398EAAD61F48CA5DB63D2A0E785167C2C0A5383392CA3867BCEA76C82758DDD17F59CA1D5E7319CB90B61ED4B92D0A1DAE80A35741025CEF31312FE9D7D702E7BD9A03CA69B6BFA36421BD1FD81428984634AA5C43103E84A19C1D555C8A0183B47F84FF9F7BF2B1F60B90F011912F1F28DFB93ACF531D6AEE95E7637A489DBF170CCBA0F22D19510741D7CEEFCE81461385C25F12A9268AAD0BF046C3461B81496EF818B07AC3E0FECB7D3E225D2B5E1A9154144EEA042EE670D6D13F573276A1DDF1D1DF26D5377F3CED69FDD137DB0590917D13D873150DE411FEC7FE4760BA1F5A395C8BA13F557079A0B4D0C9CC6559173F221D392F28E094781B0B5DB9DA2D88484670A69B795472CE02E924AE7F90695BDCC6CF757EE38CEB0667910FC752C1D95E11F38EACF83CEAB3F4C49C2B91B392F23711D997D54EB6D33469714D150FCD2767870ED00FBA3F22EE6EA26A36F77CB9A55BF764CCBB307374A352F469E5BC5BCDC165149DF4E27BFE29285B8DECBF4A028288106F500D869CB6D8D5E0C6E38EA1D5E04C426C0994D541265D0DB61E0C76DFC07C3FDBDFE65B99C4769E86948932003A853B74D6ED42735DC431FDC5448A2A971057C74D7EC167F304B0DECE460785DE070C36BBD1601EC20B88C11CB935DE2D337D6D9943C4A1FBF262C81484D7D8270219AB064E5A45A2ECC78EF0E683A3E97585D87E2397149A15DC7D85997AF5E4694298ABE860755939CDF"> : tensor<8x3x3x8xi8>
    %cst_7 = arith.constant dense<"0x17267CCEFA0F535FC845DF1CBF6C7541DD0BF157195A9D7A8CC0D718E13105E0368C3094529C1B4D2E11EE97F0C97331C455EC6A84FFFCCA3D548132EA8FC277B0BAD7412FD5CBC1DD27AF5912792B539A31083C6712DF54F88FECC75BFDBC9619B910CC03CB1CFD715EA02C084ABDD525ACF70DC7ACF05573A27FBBB7D806DA509010EF44C96E40AAAF49EE64ABF90248BC025AFBBE52F313ACEF0B2538A2006A12B3AE0C3B27DF4BE0BCB9E51A8856850EAC34D5178251714787624ADD2C28F4F09A0EE97D112E7B30F65663174762B1E45854E9A78146FD205767E1FC96EC900322618154D69C0D993A6F04B73238566D0CE8DA51BF3094606527F9B6B4FC17D96A2D23EFC487301441CC2DEA17E60C24BDA394D9C2D9674CE5097BF20C7D72F3EFF1FF6D9471E5742F03A99C2F7E78D4796C9FB5C2D335BF81056969753000691A741116F8C7A5844A0A5B459A862DAFB84B510A522D012B6EF8C490C5A156FE6EE3FB7527CA6DDABD96832D777DE2508F2081BF9DB2F13772C033047DFD9DBD8EC893EF751C4B24058C469A90638D5FF2FD26142AAEE4D808601813662440FB87DC0F395A5502A715EE6B855208B5D28D5E03E67307BCEF25A7F67111BAD926765394CB1117E7B345A720248150056E8DF6C584C8DD1D8E36D83019468171004519CB2AB34BCF0BD4EC8D833B5046C1613B1983DAE3867BBFAD0832AA9216D397CB144A369CAEC813AF2812225C7D7F141EE0848234A86AB113648EEDBA5D7E0B075CB635367AED9088FEBDEB6CA181B6736297DA27E16EDF5FBB30C79B"> : tensor<8x3x3x8xi8>
    %cst_8 = arith.constant dense<"0xA8E1F0D9B6D5B2D7E20820518941E71E1A55F5B6EA4CA253F2DE71A5CC721F67C2045534814DC652535CB603DF496493ED4DD6CD4397765ADCB91A300A4CCCEC831AE29C109B827306DD00C55BA6A2002528B0236ABB86FAB274A7C463C646812F28CB05D008B1E1304B0921C65F77A2366ED11436D1F8D4161F5584E1323FF90AD00D79B542F0C8575D06FD2FB7D4304542F709DEC3EECAD35E3D5756BEF24E63114488797F30DF591AB6A137A4438D98EC87F1C2A992462E64DE86A1AD97F6C3706CED4AF744C7AB6DA15998D38C13B7096AE27EF2A023F72AB22F51AE20878C67613F5DC487A7721FF0D248F57EBD7B0291697DDB125B7E8175D4B8D828311A002E28029447E160DFD77D0CF703E67F1024F957A9BE4DB367E925BD2A5E73A507520272DC843A6E093B44EFD00342CB3703469F628B79F2623E48D2295B67BEDDC61DFBB443C77340E4A57BA6161CCFC2583A332534AC7738929C3B77CB43F0123F4B5F22597F7399B697E33495CC4905C720A95E2F64CB3B5BE26111713862B6DE19650F1D11D7BFDC97157F6451C95B1C36FAF8D0C59035085F6BAF843A0E24707657F1E3042F5A24038F56DAEE2B1C619F47C1EDBF1ECF1E3BADFA21E1D9F9EE21E74D71A37ED697F2CFFE9A6CE645709E7326033EC09585E6815B795D00C324A1EB1631EF0695073F97D0B17D00E22B6B71394C3773E3161561DAA63F7F4BE45588025C4482F2F55418ED938488141E4B18936C12DFA19695D401C5DF0EB63BB1C1A014BA442FA4828A68E4A994FB288B1D245148D09215EB097924E4"> : tensor<8x3x3x8xi8>
    %0 = tensor.empty() : tensor<1x30x30x8xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<8xi32>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x30x30x8xi32>
    %2 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst_8, %c-128_i32, %c0_i32 : tensor<1x32x32x8xi8>, tensor<8x3x3x8xi8>, i32, i32) outs(%1 : tensor<1x30x30x8xi32>) -> tensor<1x30x30x8xi32>
    %3 = tensor.empty() : tensor<1x30x30x8xi8>
    %4 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %cst_4, %cst_3 : tensor<1x30x30x8xi32>, tensor<8xi32>, tensor<8xi8>) outs(%3 : tensor<1x30x30x8xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %17 = arith.extui %in_10 : i8 to i32
      %18 = arith.extsi %in : i32 to i64
      %19 = arith.extsi %in_9 : i32 to i64
      %20 = arith.muli %18, %19 : i64
      %21 = arith.extui %in_10 : i8 to i64
      %22 = arith.shli %c1_i64, %21 : i64
      %23 = arith.shrui %22, %c1_i64 : i64
      %24 = arith.addi %20, %23 : i64
      %25 = arith.cmpi sge, %in, %c0_i32 : i32
      %26 = arith.select %25, %c1073741824_i64, %c-1073741824_i64 : i64
      %27 = arith.addi %26, %24 : i64
      %28 = arith.cmpi sgt, %17, %c31_i32 : i32
      %29 = arith.select %28, %27, %24 : i64
      %30 = arith.shrsi %29, %21 : i64
      %31 = arith.trunci %30 : i64 to i32
      %32 = arith.addi %31, %c-128_i32 : i32
      %33 = arith.maxsi %32, %c-128_i32 : i32
      %34 = arith.minsi %33, %c127_i32 : i32
      %35 = arith.trunci %34 : i32 to i8
      linalg.yield %35 : i8
    } -> tensor<1x30x30x8xi8>
    %padded = tensor.pad %4 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c-128_i8 : i8
    } : tensor<1x30x30x8xi8> to tensor<1x32x32x8xi8>
    %5 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_7, %c-128_i32, %c0_i32 : tensor<1x32x32x8xi8>, tensor<8x3x3x8xi8>, i32, i32) outs(%1 : tensor<1x30x30x8xi32>) -> tensor<1x30x30x8xi32>
    %6 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst_2, %cst_1 : tensor<1x30x30x8xi32>, tensor<8xi32>, tensor<8xi8>) outs(%3 : tensor<1x30x30x8xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %17 = arith.extui %in_10 : i8 to i32
      %18 = arith.extsi %in : i32 to i64
      %19 = arith.extsi %in_9 : i32 to i64
      %20 = arith.muli %18, %19 : i64
      %21 = arith.extui %in_10 : i8 to i64
      %22 = arith.shli %c1_i64, %21 : i64
      %23 = arith.shrui %22, %c1_i64 : i64
      %24 = arith.addi %20, %23 : i64
      %25 = arith.cmpi sge, %in, %c0_i32 : i32
      %26 = arith.select %25, %c1073741824_i64, %c-1073741824_i64 : i64
      %27 = arith.addi %26, %24 : i64
      %28 = arith.cmpi sgt, %17, %c31_i32 : i32
      %29 = arith.select %28, %27, %24 : i64
      %30 = arith.shrsi %29, %21 : i64
      %31 = arith.trunci %30 : i64 to i32
      %32 = arith.addi %31, %c-128_i32 : i32
      %33 = arith.maxsi %32, %c-128_i32 : i32
      %34 = arith.minsi %33, %c127_i32 : i32
      %35 = arith.trunci %34 : i32 to i8
      linalg.yield %35 : i8
    } -> tensor<1x30x30x8xi8>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x30x30x8xi8>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i8, %out: i32):
      %17 = arith.extsi %in : i8 to i32
      %18 = arith.subi %17, %c-128_i32 : i32
      %19 = arith.extsi %18 : i32 to i64
      %20 = arith.muli %19, %c1073741824_i64 : i64
      %21 = arith.addi %20, %c512_i64 : i64
      %22 = arith.shrsi %21, %c10_i64 : i64
      %23 = arith.trunci %22 : i64 to i32
      linalg.yield %23 : i32
    } -> tensor<1x30x30x8xi32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<1x30x30x8xi32>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      %17 = arith.extsi %in : i32 to i64
      %18 = arith.muli %17, %c1633833687_i64 : i64
      %19 = arith.addi %18, %c2147483648_i64 : i64
      %20 = arith.cmpi sge, %in, %c0_i32 : i32
      %21 = arith.select %20, %c1073741824_i64, %c-1073741824_i64 : i64
      %22 = arith.addi %21, %19 : i64
      %23 = arith.shrui %22, %c32_i64 : i64
      %24 = arith.trunci %23 : i64 to i32
      linalg.yield %24 : i32
    } -> tensor<1x30x30x8xi32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x30x30x8xi8>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i8, %out: i32):
      %17 = arith.extsi %in : i8 to i32
      %18 = arith.subi %17, %c-128_i32 : i32
      %19 = arith.extsi %18 : i32 to i64
      %20 = arith.muli %19, %c1073741824_i64 : i64
      %21 = arith.addi %20, %c1024_i64 : i64
      %22 = arith.shrsi %21, %c11_i64 : i64
      %23 = arith.trunci %22 : i64 to i32
      linalg.yield %23 : i32
    } -> tensor<1x30x30x8xi32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<1x30x30x8xi32>, tensor<1x30x30x8xi32>) outs(%0 : tensor<1x30x30x8xi32>) {
    ^bb0(%in: i32, %in_9: i32, %out: i32):
      %17 = arith.addi %in, %in_9 : i32
      linalg.yield %17 : i32
    } -> tensor<1x30x30x8xi32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x30x30x8xi32>) outs(%3 : tensor<1x30x30x8xi8>) {
    ^bb0(%in: i32, %out: i8):
      %17 = arith.extsi %in : i32 to i64
      %18 = arith.muli %17, %c1073741824_i64 : i64
      %19 = arith.addi %18, %c281474976710656_i64 : i64
      %20 = arith.cmpi sge, %in, %c0_i32 : i32
      %21 = arith.select %20, %c1073741824_i64, %c-1073741824_i64 : i64
      %22 = arith.addi %21, %19 : i64
      %23 = arith.shrsi %22, %c49_i64 : i64
      %24 = arith.trunci %23 : i64 to i32
      %25 = arith.addi %24, %c-128_i32 : i32
      %26 = arith.maxsi %25, %c-128_i32 : i32
      %27 = arith.minsi %26, %c127_i32 : i32
      %28 = arith.trunci %27 : i32 to i8
      linalg.yield %28 : i8
    } -> tensor<1x30x30x8xi8>
    %12 = tensor.empty() : tensor<1x28x28x8xi32>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<8xi32>) outs(%12 : tensor<1x28x28x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x28x28x8xi32>
    %14 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%11, %cst_6, %c-128_i32, %c0_i32 : tensor<1x30x30x8xi8>, tensor<8x3x3x8xi8>, i32, i32) outs(%13 : tensor<1x28x28x8xi32>) -> tensor<1x28x28x8xi32>
    %15 = tensor.empty() : tensor<1x28x28x8xi8>
    %16 = linalg.generic {indexing_maps = [#map1, #map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %cst_0, %cst : tensor<1x28x28x8xi32>, tensor<8xi32>, tensor<8xi8>) outs(%15 : tensor<1x28x28x8xi8>) {
    ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
      %17 = arith.extui %in_10 : i8 to i32
      %18 = arith.extsi %in : i32 to i64
      %19 = arith.extsi %in_9 : i32 to i64
      %20 = arith.muli %18, %19 : i64
      %21 = arith.extui %in_10 : i8 to i64
      %22 = arith.shli %c1_i64, %21 : i64
      %23 = arith.shrui %22, %c1_i64 : i64
      %24 = arith.addi %20, %23 : i64
      %25 = arith.cmpi sge, %in, %c0_i32 : i32
      %26 = arith.select %25, %c1073741824_i64, %c-1073741824_i64 : i64
      %27 = arith.addi %26, %24 : i64
      %28 = arith.cmpi sgt, %17, %c31_i32 : i32
      %29 = arith.select %28, %27, %24 : i64
      %30 = arith.shrsi %29, %21 : i64
      %31 = arith.trunci %30 : i64 to i32
      %32 = arith.addi %31, %c-128_i32 : i32
      %33 = arith.maxsi %32, %c-128_i32 : i32
      %34 = arith.minsi %33, %c127_i32 : i32
      %35 = arith.trunci %34 : i32 to i8
      linalg.yield %35 : i8
    } -> tensor<1x28x28x8xi8>
    return %16 : tensor<1x28x28x8xi8>
  }
}
