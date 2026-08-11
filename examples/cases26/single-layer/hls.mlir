// This is the output of command
// ladle input.mlir -p convert-to-emithls -o hls.mlir

module attributes {tf_saved_model.semantics} {
  emithls.include "algorithm"
  emithls.include "ap_int.h"
  emithls.include "cstddef"
  emithls.include "hls_stream.h"
  emithls.func @main_top_read_i8_0(%arg0: !emithls.ptr<i64>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_index_0 = emithls.variable as const index = 1
    %const_index_1 = emithls.variable as const index = 8
    emithls.for %idx0 = 0 to 1024 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      %0 = emithls.array.ptr_read %arg0[%idx0] : !emithls.ptr<i64> -> i64
      emithls.for %idx1 = 0 to 8 step 1 {
        %expr2 = emithls.expr : i8 {
          %expr0 = emithls.expr : index {
            %2 = emithls.arith.add %idx1, %const_index_0 : index
            %3 = emithls.arith.mul %2, %const_index_1 : index
            %4 = emithls.arith.sub %3, %const_index_0 : index
            emithls.yield %4 : index
          }
          %expr1 = emithls.expr : index {
            %2 = emithls.arith.mul %idx1, %const_index_1 : index
            emithls.yield %2 : index
          }
          %1 = emithls.arith.data_range %0(%expr0, %expr1) : i64 -> i8
          emithls.yield %1 : i8
        }
        emithls.stream.write %expr2 to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_top_write_i8_0(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.ptr<i64>) {
    emithls.pragma.inline off
    %const_index_0 = emithls.variable as const index = 1
    %const_index_1 = emithls.variable as const index = 8
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      %var_int64_0 = emithls.variable as i64 = 0
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i8>> -> i8
        %expr2 = emithls.expr : i8 {
          %expr0 = emithls.expr : index {
            %2 = emithls.arith.add %idx1, %const_index_0 : index
            %3 = emithls.arith.mul %2, %const_index_1 : index
            %4 = emithls.arith.sub %3, %const_index_0 : index
            emithls.yield %4 : index
          }
          %expr1 = emithls.expr : index {
            %2 = emithls.arith.mul %idx1, %const_index_1 : index
            emithls.yield %2 : index
          }
          %1 = emithls.arith.data_range %var_int64_0(%expr0, %expr1) : i64 -> i8
          emithls.yield %1 : i8
        }
        emithls.update %expr2 with %0 : i8 <- i8
      }
      emithls.array.ptr_write %var_int64_0, %arg1[%idx0] : i64 -> !emithls.ptr<i64>
    }
  }
  emithls.func @main_node_0(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
    emithls.pragma.inline off
    %const_index_0 = emithls.variable as const index = 2
    %const_index_1 = emithls.variable as const index = 1
    %const_int8_0 = emithls.variable as const i8 = 0
    %const_index_2 = emithls.variable as const index = 0
    %var_array_0 = emithls.variable as !emithls.array<8x3x3xi8>
    emithls.pragma.bind_storage variable=%var_array_0(!emithls.array<8x3x3xi8>) type=ram_2p impl=lutram
    emithls.pragma.array_partition variable=%var_array_0(!emithls.array<8x3x3xi8>) type=complete dim=1
    emithls.pragma.array_partition variable=%var_array_0(!emithls.array<8x3x3xi8>) type=complete dim=2
    emithls.pragma.array_partition variable=%var_array_0(!emithls.array<8x3x3xi8>) type=complete dim=3
    %var_array_1 = emithls.variable as !emithls.array<8x2x32xi8>
    emithls.pragma.bind_storage variable=%var_array_1(!emithls.array<8x2x32xi8>) type=ram_2p impl=bram
    emithls.pragma.array_partition variable=%var_array_1(!emithls.array<8x2x32xi8>) type=complete dim=1
    emithls.pragma.array_partition variable=%var_array_1(!emithls.array<8x2x32xi8>) type=complete dim=2
    %const_array_0 = emithls.variable as const !emithls.array<8x3x3x8xi8> = dense<"0x3A971B7863DFAEAE6E9FA742A6D9AB301B052A3BF1413FB6EC4E28EFC60FF4B9650E1755841553970E4F814D52CBCCBD43C24EC473D98E2779CE48C0F54C31A48175AF3F539C32C3357144E98D85200A322FD4256F5559B979DB1308187F8B8DB08211798BFF403DD9ABD2E06EF6D5D9B1F28D2CFCCB236F81AA0EC1E03D00C6EE7AC44E00D966650E5B6BDEB4981B5D81F5758D3FB5F5768850CE45368D641D05CE09EC74E79BC06664E43B02C0ECE875D905EFEA9312668195A928174028C07857706CF608ED278BC5341EA7B7D8845AF989176D47750F90054CC0382EF3BE107E33812941AA57AD40B25D72370586299255D91ADF44961F6857A0213292992070342E1DF79DC4052E2CBAEA5D596A02918A19CB9C9AA0F46EF267E901FB5BF079B1FCE4015B7FB5142D5187009705F955FDB7DE90FEDF5482F75AADD055D4647BC2B0682A6420960F569EDF0D65843C9BBDEE111E1AE3D56D51B9F7AD7CDA0FB0EC68ED23A1EA75810211712A40C5E86194701DED4D079F8B0BFDCC02B8497BC4B768427FA666669BE915A4F6E844F49F6D743B4ADDDE6BA7617728528D8E9D329A69BD0FC541D48D8121FE25E512158B5299D4FB1C3EDB6F138182E2A67DDD7277831AA3A4DC8FBA913D3C129BA4A191C593855FCCDED6A0D1094078EAAAC197641D5413DD289387517B8E8EB851B8DC0ABCCA70A7A7FF9C81E67D33A6665D8A174C88E24AF5BF6A1908349E8185B125522DA4BB4F218C842D7BC904F6E31CEE7FACD9E44824A04BDB6F0D3A6734B41C02B04485AA563686A7667F099863">
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=1
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=2
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=3
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=4
    %const_int32_0 = emithls.variable as const i32 = -128
    emithls.for %idx0 = 0 to 32 step 1 {
      emithls.for %idx1 = 0 to 32 step 1 {
        emithls.for %idx2 = 0 to 1 step 1 {
          emithls.pragma.pipeline II=1 style=flp
          emithls.for %idx3 = 0 to 8 step 1 {
            %0 = emithls.stream.read %arg0[%idx3] : !emithls.array<8x!emithls.stream<i8>> -> i8
            %expr0 = emithls.expr : i1 {
              %1 = emithls.arith.cmp eq, %idx0, %const_index_2 : index
              emithls.yield %1 : i1
            }
            emithls.for %idx4 = 0 to 3 step 1 {
              emithls.for %idx5 = 0 to 2 step 1 {
                emithls.if %expr0 {
                  emithls.update %var_array_0[%idx3, %idx4, %idx5] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
                } else {
                  %expr1 = emithls.expr : index {
                    %1 = emithls.arith.add %idx5, %const_index_1 : index
                    emithls.yield %1 : index
                  }
                  emithls.update %var_array_0[%idx3, %idx4, %idx5] with %var_array_0[%idx3, %idx4, %expr1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x3x3xi8>
                }
              }
            }
            %expr2 = emithls.expr : i1 {
              %1 = emithls.arith.cmp ge, %idx0, %const_index_0 : index
              emithls.yield %1 : i1
            }
            emithls.if %expr2 {
              %expr3 = emithls.expr : index {
                %1 = emithls.arith.sub %idx0, %const_index_0 : index
                %2 = emithls.arith.rem %1, %const_index_0 : index
                emithls.yield %2 : index
              }
              emithls.update %var_array_0[%idx3, %const_index_2, %const_index_0] with %var_array_1[%idx3, %expr3, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
            } else {
              emithls.update %var_array_0[%idx3, %const_index_2, %const_index_0] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
            }
            %expr4 = emithls.expr : i1 {
              %1 = emithls.arith.cmp ge, %idx0, %const_index_1 : index
              emithls.yield %1 : i1
            }
            emithls.if %expr4 {
              %expr5 = emithls.expr : index {
                %1 = emithls.arith.sub %idx0, %const_index_1 : index
                %2 = emithls.arith.rem %1, %const_index_0 : index
                emithls.yield %2 : index
              }
              emithls.update %var_array_0[%idx3, %const_index_1, %const_index_0] with %var_array_1[%idx3, %expr5, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
            } else {
              emithls.update %var_array_0[%idx3, %const_index_1, %const_index_0] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
            }
            %expr6 = emithls.expr : index {
              %1 = emithls.arith.rem %idx0, %const_index_0 : index
              emithls.yield %1 : index
            }
            emithls.update %var_array_0[%idx3, %const_index_0, %const_index_0] with %0 : !emithls.array<8x3x3xi8> <- i8
            emithls.update %var_array_1[%idx3, %expr6, %idx1] with %0 : !emithls.array<8x2x32xi8> <- i8
          }
        }
        %expr7 = emithls.expr : i1 {
          %0 = emithls.arith.cmp ge, %idx0, %const_index_0 : index
          %1 = emithls.arith.cmp ge, %idx1, %const_index_0 : index
          %2 = emithls.arith.logical_and %0, %1 : i1
          emithls.yield %2 : i1
        }
        emithls.if %expr7 {
          emithls.for %idx2 = 0 to 1 step 1 {
            emithls.pragma.pipeline II=1 style=flp
            emithls.for %idx3 = 0 to 8 step 1 {
              %var_int32_0 = emithls.variable as i32 = 0
              emithls.for %idx4 = 0 to 3 step 1 {
                emithls.for %idx5 = 0 to 3 step 1 {
                  emithls.for %idx6 = 0 to 8 step 1 {
                    %0 = emithls.array.read %var_array_0[%idx6, %idx4, %idx5] : !emithls.array<8x3x3xi8> -> i8
                    %1 = emithls.array.read %const_array_0[%idx3, %idx4, %idx5, %idx6] : !emithls.array<8x3x3x8xi8> -> i8
                    %2 = emithls.arith.cast %0 : i8 to i32
                    %3 = emithls.arith.sub %2, %const_int32_0 : i32
                    %4 = emithls.arith.cast %1 : i8 to i32
                    %5 = emithls.arith.mul %3, %4 : i32
                    emithls.arith.fused add, %var_int32_0, %5 : i32
                  }
                }
              }
              emithls.stream.write %var_int32_0 to %arg1[%idx3] : i32 -> !emithls.array<8x!emithls.stream<i32>>
            }
          }
        }
      }
    }
  }
  emithls.func @main_node_1(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_array_0 = emithls.variable as const !emithls.array<8xi32> = dense<[1288418624, 1300707578, 1298230215, 1293868576, 1310478957, 1307418510, 1313803263, 1306561467]>
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8xi32>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8xi32>) type=complete dim=1
    %const_int32_0 = emithls.variable as const i32 = 127
    %const_int32_1 = emithls.variable as const i32 = -128
    %const_int64_0 = emithls.variable as const i64 = 40
    %const_int64_1 = emithls.variable as const i64 = -1073741824
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_2 = emithls.variable as const i32 = 0
    %const_int64_3 = emithls.variable as const i64 = 549755813888
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %1 = emithls.array.read %const_array_0[%idx1] : !emithls.array<8xi32> -> i32
        %2 = emithls.arith.cast %0 : i32 to i64
        %3 = emithls.arith.cast %1 : i32 to i64
        %4 = emithls.arith.mul %2, %3 : i64
        %5 = emithls.arith.add %4, %const_int64_3 : i64
        %6 = emithls.arith.cmp ge, %0, %const_int32_2 : i32
        %7 = emithls.arith.select %6, %const_int64_2, %const_int64_1 : i64
        %8 = emithls.arith.add %7, %5 : i64
        %9 = emithls.arith.shr %8, %const_int64_0 : i64
        %10 = emithls.arith.cast %9 : i64 to i32
        %11 = emithls.arith.add %10, %const_int32_1 : i32
        %12 = emithls.arith.max %11, %const_int32_1 : i32
        %13 = emithls.arith.min %12, %const_int32_0 : i32
        %14 = emithls.arith.cast %13 : i32 to i8
        emithls.stream.write %14 to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_top(%arg0: !emithls.ptr<i64>, %arg1: !emithls.ptr<i64>) {
    emithls.top_interface
    %var_array_0 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) depth=10
    %var_array_1 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_1(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_1(!emithls.array<8x!emithls.stream<i8>>) depth=2
    %var_array_2 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) depth=16
    emithls.pragma.dataflow {
      emithls.call @main_top_read_i8_0(%arg0, %var_array_0) : (!emithls.ptr<i64>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_0(%var_array_0, %var_array_2) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_1(%var_array_2, %var_array_1) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_top_write_i8_0(%var_array_1, %arg1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.ptr<i64>) -> ()
    }
  }
}

