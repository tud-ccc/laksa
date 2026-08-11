// Generated from input.mlir with command
// laksa-opt --convert-to-emithls

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
    emithls.for %idx0 = 0 to 784 step 1 {
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
    %const_array_0 = emithls.variable as const !emithls.array<8x3x3x8xi8> = dense<"0xA8E1F0D9B6D5B2D7E20820518941E71E1A55F5B6EA4CA253F2DE71A5CC721F67C2045534814DC652535CB603DF496493ED4DD6CD4397765ADCB91A300A4CCCEC831AE29C109B827306DD00C55BA6A2002528B0236ABB86FAB274A7C463C646812F28CB05D008B1E1304B0921C65F77A2366ED11436D1F8D4161F5584E1323FF90AD00D79B542F0C8575D06FD2FB7D4304542F709DEC3EECAD35E3D5756BEF24E63114488797F30DF591AB6A137A4438D98EC87F1C2A992462E64DE86A1AD97F6C3706CED4AF744C7AB6DA15998D38C13B7096AE27EF2A023F72AB22F51AE20878C67613F5DC487A7721FF0D248F57EBD7B0291697DDB125B7E8175D4B8D828311A002E28029447E160DFD77D0CF703E67F1024F957A9BE4DB367E925BD2A5E73A507520272DC843A6E093B44EFD00342CB3703469F628B79F2623E48D2295B67BEDDC61DFBB443C77340E4A57BA6161CCFC2583A332534AC7738929C3B77CB43F0123F4B5F22597F7399B697E33495CC4905C720A95E2F64CB3B5BE26111713862B6DE19650F1D11D7BFDC97157F6451C95B1C36FAF8D0C59035085F6BAF843A0E24707657F1E3042F5A24038F56DAEE2B1C619F47C1EDBF1ECF1E3BADFA21E1D9F9EE21E74D71A37ED697F2CFFE9A6CE645709E7326033EC09585E6815B795D00C324A1EB1631EF0695073F97D0B17D00E22B6B71394C3773E3161561DAA63F7F4BE45588025C4482F2F55418ED938488141E4B18936C12DFA19695D401C5DF0EB63BB1C1A014BA442FA4828A68E4A994FB288B1D245148D09215EB097924E4">
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
  emithls.func @main_node_1(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>, %arg2: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_array_0 = emithls.variable as const !emithls.array<8xi32> = dense<[1523322938, 1544671534, 1547120170, 1546480699, 1536582660, 1557469766, 1525058969, 1526555847]>
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8xi32>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8xi32>) type=complete dim=1
    %const_int32_0 = emithls.variable as const i32 = 127
    %const_int32_1 = emithls.variable as const i32 = -128
    %const_int64_0 = emithls.variable as const i64 = 41
    %const_int64_1 = emithls.variable as const i64 = -1073741824
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_2 = emithls.variable as const i32 = 0
    %const_int64_3 = emithls.variable as const i64 = 1099511627776
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
        emithls.stream.write %14 to %arg2[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_node_2(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_index_0 = emithls.variable as const index = 30
    %const_index_1 = emithls.variable as const index = 1
    %const_int8_0 = emithls.variable as const i8 = -128
    emithls.for %idx0 = 0 to 32 step 1 {
      emithls.for %idx1 = 0 to 32 step 1 {
        emithls.pragma.pipeline II=1 style=flp
        %expr0 = emithls.expr : i1 {
          %0 = emithls.arith.cmp ge, %idx0, %const_index_1 : index
          %1 = emithls.arith.cmp le, %idx0, %const_index_0 : index
          %2 = emithls.arith.cmp ge, %idx1, %const_index_1 : index
          %3 = emithls.arith.cmp le, %idx1, %const_index_0 : index
          %4 = emithls.arith.logical_and %0, %1, %2, %3 : i1
          emithls.yield %4 : i1
        }
        emithls.if %expr0 {
          emithls.for %idx2 = 0 to 8 step 1 {
            %0 = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
            emithls.stream.write %0 to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
          }
        } else {
          emithls.for %idx2 = 0 to 8 step 1 {
            emithls.stream.write %const_int8_0 to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
          }
        }
      }
    }
  }
  emithls.func @main_node_3(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
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
    %const_array_0 = emithls.variable as const !emithls.array<8x3x3x8xi8> = dense<"0x17267CCEFA0F535FC845DF1CBF6C7541DD0BF157195A9D7A8CC0D718E13105E0368C3094529C1B4D2E11EE97F0C97331C455EC6A84FFFCCA3D548132EA8FC277B0BAD7412FD5CBC1DD27AF5912792B539A31083C6712DF54F88FECC75BFDBC9619B910CC03CB1CFD715EA02C084ABDD525ACF70DC7ACF05573A27FBBB7D806DA509010EF44C96E40AAAF49EE64ABF90248BC025AFBBE52F313ACEF0B2538A2006A12B3AE0C3B27DF4BE0BCB9E51A8856850EAC34D5178251714787624ADD2C28F4F09A0EE97D112E7B30F65663174762B1E45854E9A78146FD205767E1FC96EC900322618154D69C0D993A6F04B73238566D0CE8DA51BF3094606527F9B6B4FC17D96A2D23EFC487301441CC2DEA17E60C24BDA394D9C2D9674CE5097BF20C7D72F3EFF1FF6D9471E5742F03A99C2F7E78D4796C9FB5C2D335BF81056969753000691A741116F8C7A5844A0A5B459A862DAFB84B510A522D012B6EF8C490C5A156FE6EE3FB7527CA6DDABD96832D777DE2508F2081BF9DB2F13772C033047DFD9DBD8EC893EF751C4B24058C469A90638D5FF2FD26142AAEE4D808601813662440FB87DC0F395A5502A715EE6B855208B5D28D5E03E67307BCEF25A7F67111BAD926765394CB1117E7B345A720248150056E8DF6C584C8DD1D8E36D83019468171004519CB2AB34BCF0BD4EC8D833B5046C1613B1983DAE3867BBFAD0832AA9216D397CB144A369CAEC813AF2812225C7D7F141EE0848234A86AB113648EEDBA5D7E0B075CB635367AED9088FEBDEB6CA181B6736297DA27E16EDF5FBB30C79B">
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=rom_1p impl=lutram
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
          emithls.for %idx2 = 0 to 8 step 1 {
            %var_int32_0 = emithls.variable as i32 = 0
            emithls.for %idx3 = 0 to 1 step 1 {
              emithls.pragma.pipeline II=1 style=flp
              emithls.for %idx4 = 0 to 3 step 1 {
                emithls.for %idx5 = 0 to 3 step 1 {
                  emithls.for %idx6 = 0 to 8 step 1 {
                    %0 = emithls.array.read %var_array_0[%idx6, %idx4, %idx5] : !emithls.array<8x3x3xi8> -> i8
                    %1 = emithls.array.read %const_array_0[%idx2, %idx4, %idx5, %idx6] : !emithls.array<8x3x3x8xi8> -> i8
                    %2 = emithls.arith.cast %0 : i8 to i32
                    %3 = emithls.arith.sub %2, %const_int32_0 : i32
                    %4 = emithls.arith.cast %1 : i8 to i32
                    %5 = emithls.arith.mul %3, %4 : i32
                    emithls.arith.fused add, %var_int32_0, %5 : i32
                  }
                }
              }
            }
            emithls.stream.write %var_int32_0 to %arg1[%idx2] : i32 -> !emithls.array<8x!emithls.stream<i32>>
          }
        }
      }
    }
  }
  emithls.func @main_node_4(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_array_0 = emithls.variable as const !emithls.array<8xi32> = dense<[1241604906, 1329922038, 1341576575, 1322904542, 1326948123, 1339807398, 1325877483, 1340943676]>
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
  emithls.func @main_node_5(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
    emithls.pragma.inline off
    %const_int64_0 = emithls.variable as const i64 = 10
    %const_int64_1 = emithls.variable as const i64 = 512
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_0 = emithls.variable as const i32 = -128
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i8>> -> i8
        %1 = emithls.arith.cast %0 : i8 to i32
        %2 = emithls.arith.sub %1, %const_int32_0 : i32
        %3 = emithls.arith.cast %2 : i32 to i64
        %4 = emithls.arith.mul %3, %const_int64_2 : i64
        %5 = emithls.arith.add %4, %const_int64_1 : i64
        %6 = emithls.arith.shr %5, %const_int64_0 : i64
        %7 = emithls.arith.cast %6 : i64 to i32
        emithls.stream.write %7 to %arg1[%idx1] : i32 -> !emithls.array<8x!emithls.stream<i32>>
      }
    }
  }
  emithls.func @main_node_6(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
    emithls.pragma.inline off
    %const_int64_0 = emithls.variable as const i64 = 32
    %const_int64_1 = emithls.variable as const i64 = -1073741824
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_0 = emithls.variable as const i32 = 0
    %const_int64_3 = emithls.variable as const i64 = 2147483648
    %const_int64_4 = emithls.variable as const i64 = 1633833687
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %1 = emithls.arith.cast %0 : i32 to i64
        %2 = emithls.arith.mul %1, %const_int64_4 : i64
        %3 = emithls.arith.add %2, %const_int64_3 : i64
        %4 = emithls.arith.cmp ge, %0, %const_int32_0 : i32
        %5 = emithls.arith.select %4, %const_int64_2, %const_int64_1 : i64
        %6 = emithls.arith.add %5, %3 : i64
        %7 = emithls.arith.shr %6, %const_int64_0 : i64
        %8 = emithls.arith.cast %7 : i64 to i32
        emithls.stream.write %8 to %arg1[%idx1] : i32 -> !emithls.array<8x!emithls.stream<i32>>
      }
    }
  }
  emithls.func @main_node_7(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
    emithls.pragma.inline off
    %const_int64_0 = emithls.variable as const i64 = 11
    %const_int64_1 = emithls.variable as const i64 = 1024
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_0 = emithls.variable as const i32 = -128
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i8>> -> i8
        %1 = emithls.arith.cast %0 : i8 to i32
        %2 = emithls.arith.sub %1, %const_int32_0 : i32
        %3 = emithls.arith.cast %2 : i32 to i64
        %4 = emithls.arith.mul %3, %const_int64_2 : i64
        %5 = emithls.arith.add %4, %const_int64_1 : i64
        %6 = emithls.arith.shr %5, %const_int64_0 : i64
        %7 = emithls.arith.cast %6 : i64 to i32
        emithls.stream.write %7 to %arg1[%idx1] : i32 -> !emithls.array<8x!emithls.stream<i32>>
      }
    }
  }
  emithls.func @main_node_8(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i32>>, %arg2: !emithls.array<8x!emithls.stream<i32>>) {
    emithls.pragma.inline off
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %1 = emithls.stream.read %arg1[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %2 = emithls.arith.add %0, %1 : i32
        emithls.stream.write %2 to %arg2[%idx1] : i32 -> !emithls.array<8x!emithls.stream<i32>>
      }
    }
  }
  emithls.func @main_node_9(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_int32_0 = emithls.variable as const i32 = 127
    %const_int32_1 = emithls.variable as const i32 = -128
    %const_int64_0 = emithls.variable as const i64 = 49
    %const_int64_1 = emithls.variable as const i64 = -1073741824
    %const_int32_2 = emithls.variable as const i32 = 0
    %const_int64_2 = emithls.variable as const i64 = 281474976710656
    %const_int64_3 = emithls.variable as const i64 = 1073741824
    emithls.for %idx0 = 0 to 900 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %1 = emithls.arith.cast %0 : i32 to i64
        %2 = emithls.arith.mul %1, %const_int64_3 : i64
        %3 = emithls.arith.add %2, %const_int64_2 : i64
        %4 = emithls.arith.cmp ge, %0, %const_int32_2 : i32
        %5 = emithls.arith.select %4, %const_int64_3, %const_int64_1 : i64
        %6 = emithls.arith.add %5, %3 : i64
        %7 = emithls.arith.shr %6, %const_int64_0 : i64
        %8 = emithls.arith.cast %7 : i64 to i32
        %9 = emithls.arith.add %8, %const_int32_1 : i32
        %10 = emithls.arith.max %9, %const_int32_1 : i32
        %11 = emithls.arith.min %10, %const_int32_0 : i32
        %12 = emithls.arith.cast %11 : i32 to i8
        emithls.stream.write %12 to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_node_10(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
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
    %var_array_1 = emithls.variable as !emithls.array<8x2x30xi8>
    emithls.pragma.bind_storage variable=%var_array_1(!emithls.array<8x2x30xi8>) type=ram_2p impl=bram
    emithls.pragma.array_partition variable=%var_array_1(!emithls.array<8x2x30xi8>) type=complete dim=1
    emithls.pragma.array_partition variable=%var_array_1(!emithls.array<8x2x30xi8>) type=complete dim=2
    %const_array_0 = emithls.variable as const !emithls.array<8x3x3x8xi8> = dense<"0x9F9F3BCAF4377099E429251884D61EA18C8199B9064B9D317A7F96A5E7FB44171EC7E93624713912363A8ECF112398EAAD61F48CA5DB63D2A0E785167C2C0A5383392CA3867BCEA76C82758DDD17F59CA1D5E7319CB90B61ED4B92D0A1DAE80A35741025CEF31312FE9D7D702E7BD9A03CA69B6BFA36421BD1FD81428984634AA5C43103E84A19C1D555C8A0183B47F84FF9F7BF2B1F60B90F011912F1F28DFB93ACF531D6AEE95E7637A489DBF170CCBA0F22D19510741D7CEEFCE81461385C25F12A9268AAD0BF046C3461B81496EF818B07AC3E0FECB7D3E225D2B5E1A9154144EEA042EE670D6D13F573276A1DDF1D1DF26D5377F3CED69FDD137DB0590917D13D873150DE411FEC7FE4760BA1F5A395C8BA13F557079A0B4D0C9CC6559173F221D392F28E094781B0B5DB9DA2D88484670A69B795472CE02E924AE7F90695BDCC6CF757EE38CEB0667910FC752C1D95E11F38EACF83CEAB3F4C49C2B91B392F23711D997D54EB6D33469714D150FCD2767870ED00FBA3F22EE6EA26A36F77CB9A55BF764CCBB307374A352F469E5BC5BCDC165149DF4E27BFE29285B8DECBF4A028288106F500D869CB6D8D5E0C6E38EA1D5E04C426C0994D541265D0DB61E0C76DFC07C3FDBDFE65B99C4769E86948932003A853B74D6ED42735DC431FDC5448A2A971057C74D7EC167F304B0DECE460785DE070C36BBD1601EC20B88C11CB935DE2D337D6D9943C4A1FBF262C81484D7D8270219AB064E5A45A2ECC78EF0E683A3E97585D87E2397149A15DC7D85997AF5E4694298ABE860755939CDF">
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=2
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=3
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8x3x3x8xi8>) type=complete dim=4
    %const_int32_0 = emithls.variable as const i32 = -128
    emithls.for %idx0 = 0 to 30 step 1 {
      emithls.for %idx1 = 0 to 30 step 1 {
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
              emithls.update %var_array_0[%idx3, %const_index_2, %const_index_0] with %var_array_1[%idx3, %expr3, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x30xi8>
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
              emithls.update %var_array_0[%idx3, %const_index_1, %const_index_0] with %var_array_1[%idx3, %expr5, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x30xi8>
            } else {
              emithls.update %var_array_0[%idx3, %const_index_1, %const_index_0] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
            }
            %expr6 = emithls.expr : index {
              %1 = emithls.arith.rem %idx0, %const_index_0 : index
              emithls.yield %1 : index
            }
            emithls.update %var_array_0[%idx3, %const_index_0, %const_index_0] with %0 : !emithls.array<8x3x3xi8> <- i8
            emithls.update %var_array_1[%idx3, %expr6, %idx1] with %0 : !emithls.array<8x2x30xi8> <- i8
          }
        }
        %expr7 = emithls.expr : i1 {
          %0 = emithls.arith.cmp ge, %idx0, %const_index_0 : index
          %1 = emithls.arith.cmp ge, %idx1, %const_index_0 : index
          %2 = emithls.arith.logical_and %0, %1 : i1
          emithls.yield %2 : i1
        }
        emithls.if %expr7 {
          emithls.for %idx2 = 0 to 8 step 1 {
            %var_int32_0 = emithls.variable as i32 = 0
            emithls.for %idx3 = 0 to 1 step 1 {
              emithls.pragma.pipeline II=1 style=flp
              emithls.for %idx4 = 0 to 3 step 1 {
                emithls.for %idx5 = 0 to 3 step 1 {
                  emithls.for %idx6 = 0 to 8 step 1 {
                    %0 = emithls.array.read %var_array_0[%idx6, %idx4, %idx5] : !emithls.array<8x3x3xi8> -> i8
                    %1 = emithls.array.read %const_array_0[%idx2, %idx4, %idx5, %idx6] : !emithls.array<8x3x3x8xi8> -> i8
                    %2 = emithls.arith.cast %0 : i8 to i32
                    %3 = emithls.arith.sub %2, %const_int32_0 : i32
                    %4 = emithls.arith.cast %1 : i8 to i32
                    %5 = emithls.arith.mul %3, %4 : i32
                    emithls.arith.fused add, %var_int32_0, %5 : i32
                  }
                }
              }
            }
            emithls.stream.write %var_int32_0 to %arg1[%idx2] : i32 -> !emithls.array<8x!emithls.stream<i32>>
          }
        }
      }
    }
  }
  emithls.func @main_node_11(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_array_0 = emithls.variable as const !emithls.array<8xi8> = dense<[39, 39, 39, 40, 40, 39, 40, 40]>
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<8xi8>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<8xi8>) type=complete dim=1
    %const_array_1 = emithls.variable as const !emithls.array<8xi32> = dense<[1079388899, 1080650220, 1080683627, 2123794270, 2107169143, 1079132406, 2033284180, 2112265834]>
    emithls.pragma.bind_storage variable=%const_array_1(!emithls.array<8xi32>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_1(!emithls.array<8xi32>) type=complete dim=1
    %const_int32_0 = emithls.variable as const i32 = 127
    %const_int32_1 = emithls.variable as const i32 = -128
    %const_int32_2 = emithls.variable as const i32 = 31
    %const_int64_0 = emithls.variable as const i64 = -1073741824
    %const_int64_1 = emithls.variable as const i64 = 1073741824
    %const_int32_3 = emithls.variable as const i32 = 0
    %const_int64_2 = emithls.variable as const i64 = 1
    emithls.for %idx0 = 0 to 784 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
        %1 = emithls.array.read %const_array_1[%idx1] : !emithls.array<8xi32> -> i32
        %2 = emithls.array.read %const_array_0[%idx1] : !emithls.array<8xi8> -> i8
        %3 = emithls.arith.cast %2 : i8 to i32
        %4 = emithls.arith.cast %0 : i32 to i64
        %5 = emithls.arith.cast %1 : i32 to i64
        %6 = emithls.arith.mul %4, %5 : i64
        %7 = emithls.arith.cast %2 : i8 to i64
        %8 = emithls.arith.shl %const_int64_2, %7 : i64
        %9 = emithls.arith.shr %8, %const_int64_2 : i64
        %10 = emithls.arith.add %6, %9 : i64
        %11 = emithls.arith.cmp ge, %0, %const_int32_3 : i32
        %12 = emithls.arith.select %11, %const_int64_1, %const_int64_0 : i64
        %13 = emithls.arith.add %12, %10 : i64
        %14 = emithls.arith.cmp gt, %3, %const_int32_2 : i32
        %15 = emithls.arith.select %14, %13, %10 : i64
        %16 = emithls.arith.shr %15, %7 : i64
        %17 = emithls.arith.cast %16 : i64 to i32
        %18 = emithls.arith.add %17, %const_int32_1 : i32
        %19 = emithls.arith.max %18, %const_int32_1 : i32
        %20 = emithls.arith.min %19, %const_int32_0 : i32
        %21 = emithls.arith.cast %20 : i32 to i8
        emithls.stream.write %21 to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_top(%arg0: !emithls.ptr<i64>, %arg1: !emithls.ptr<i64>) {
    %var_array_0 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) depth=10
    %var_array_1 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_1(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_1(!emithls.array<8x!emithls.stream<i8>>) depth=2
    %var_array_2 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) depth=17
    %var_array_3 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_3(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_3(!emithls.array<8x!emithls.stream<i8>>) depth=2
    %var_array_4 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_4(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_4(!emithls.array<8x!emithls.stream<i8>>) depth=9
    %var_array_5 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_5(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_5(!emithls.array<8x!emithls.stream<i8>>) depth=17
    %var_array_6 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_6(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_6(!emithls.array<8x!emithls.stream<i32>>) depth=16
    %var_array_7 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_7(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_7(!emithls.array<8x!emithls.stream<i8>>) depth=9
    %var_array_8 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_8(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_8(!emithls.array<8x!emithls.stream<i32>>) depth=10
    %var_array_9 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_9(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_9(!emithls.array<8x!emithls.stream<i32>>) depth=27
    %var_array_10 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_10(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_10(!emithls.array<8x!emithls.stream<i32>>) depth=2
    %var_array_11 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_11(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_11(!emithls.array<8x!emithls.stream<i32>>) depth=14
    %var_array_12 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_12(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_12(!emithls.array<8x!emithls.stream<i8>>) depth=17
    %var_array_13 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_13(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_13(!emithls.array<8x!emithls.stream<i32>>) depth=23
    emithls.pragma.dataflow {
      emithls.call @main_top_read_i8_0(%arg0, %var_array_0) : (!emithls.ptr<i64>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_0(%var_array_0, %var_array_2) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_1(%var_array_2, %var_array_3, %var_array_4) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_2(%var_array_3, %var_array_5) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_3(%var_array_5, %var_array_6) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_4(%var_array_6, %var_array_7) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_5(%var_array_4, %var_array_8) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_6(%var_array_8, %var_array_9) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_7(%var_array_7, %var_array_10) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_8(%var_array_9, %var_array_10, %var_array_11) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_9(%var_array_11, %var_array_12) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_10(%var_array_12, %var_array_13) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_11(%var_array_13, %var_array_1) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_top_write_i8_0(%var_array_1, %arg1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.ptr<i64>) -> ()
    }
  }
}

