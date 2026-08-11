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
  emithls.func @main_top_write_i8_0(%arg0: !emithls.array<16x!emithls.stream<i8>>, %arg1: !emithls.ptr<i128>) {
    emithls.pragma.inline off
    %const_index_0 = emithls.variable as const index = 1
    %const_index_1 = emithls.variable as const index = 8
    emithls.for %idx0 = 0 to 784 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      %var_int128_0 = emithls.variable as i128 = 0
      emithls.for %idx1 = 0 to 16 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<16x!emithls.stream<i8>> -> i8
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
          %1 = emithls.arith.data_range %var_int128_0(%expr0, %expr1) : i128 -> i8
          emithls.yield %1 : i8
        }
        emithls.update %expr2 with %0 : i8 <- i8
      }
      emithls.array.ptr_write %var_int128_0, %arg1[%idx0] : i128 -> !emithls.ptr<i128>
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
  emithls.func @main_node_2(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<16x!emithls.stream<i32>>) {
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
    %const_array_0 = emithls.variable as const !emithls.array<16x3x3x8xi8> = dense<"0x00BC7FCDE88F3D0F0C7E098FBDB7DC71F44758532320E2A847A3A28B0BFCD065B5F225A5987E046D4B9BC51789037E3713D076AA8225565F31F0A1D9C3B46D93FE4B0DDC5CC7174941EF36D60D7B6075288628613BF4106998676755639D7676B77F4998531A8ACE907B8B0C74FB690219F86471531D49FE1826CC2F755EDAB9EF62F9C843263BBF9A49EF226F04DE02F489176C84A51767877D47BE0313C212FF84A9911A4AD65D1B4823A80DABE9EFF53022223AA71D478168946B5CDC147E2B06CF7D129AFB3E44A9A5BC4A46AC5B113935727CF9D6E0331277F62D3A4BC8BC1BB35064FFE5D6AA1E769EA28EEE3AEEE762B27C910039DFC1E68178361E1A5B89543744AF2D41B3AA3C6708C8816D714287D577B48D49B8144FC58C36AEBEB392BB25349A347C1BC77FABCA74BF25B4338334C79902747FC7E2BC523FE06861F2FA115555B30E692CED2A794A84CB4AAF416812CC15631FD0EECD752971C0570A48D4BCB077AE16574D34D4CE9443FBDC26E1FD7AC7719CEC2090E72D7E7E769F71D48A8D02EA7D6181CBED06A94107307F4CD98127C1C7673E32A1990770AB0AA88AB794BEC1CD7E9C8E1986D1B13A4CB6C865D1B2105C8190AF77E990AC425E7306E7BD09BB2BF164D96405601073D678C05762633F67E35DD28CACBC2DD351235B29790BD1033B881240A36DB15E0AEF7A4CF157349C8A20F49FE5FD331677FED30F4EF06AE7F91041D66AA56B7A5C4246BC523AB6D0650F268F594D13110D7AE64B65171888ABFCF4D07F95E3E6A799A4992ED2CE43ADD9AEA2FED041C8AE84C9590AC5A964066683AFD4DB8B4FD2EF213BCB790F016CD963B0E2D01CAEE6D89508A511FE12AB517929787083111722272DA8512CC7372C25C9A0321481E8DA584442F35598DF36F44EA3E02FDBC37BBE25173AAB62E8F3DF6203E87095A69957FC5A935907B74055DF6F8131799AA93CF2DB42B792B011C3E8F7032023D3D0BCEC464515F3100ABD7D7B97763AD74D2B42BE63A73E314E87A47F0AEF17FF3AB307D97252040AA59A4BF5B6FE1271C63207215564BF4BB72540C4194C332622DEEA63CF91A32F89245DF648FE6FCF8995096F6633E566CAF368EB66AA2E14695172DCB68858F423C06C1BA6994C166578E3AB963D238B0392D8EF3CA63F337AC11F7B6A132A597DC9E09D51168A207F7026025E1F789ABB35BA3F41FA65289E410FE22A9BCDB96244A7DF9258BABCD8693AA58411CFBD87BCD9E97F7AA3FDFA213377D9AA9FE6F451F24AE31A4179A86677C11D9DF8C617D17D2400D529038C37E40BF7531AAD5120AEFD56C74C3D0966498A0531869CB5810E628C6803462EF18CD0456E83059D82EF400F59BCF412963C46B71022AE57D131CE2C314F77E4500D88DD26BA626382F2EF75572DDDB6EC7517CE23779FF47E2C7CE766884E683661B06A652570E58A07014438D17831DB8D14D24048E981105BD6A5C3C018C27EF39A44976F6FDD2E515C7ED7AA7627F13827F512F56B4BD044FB9E77AD8153D7DEED432C4E83BACAA90C28DA76CF1ABCC1D2504EF1EE8D554D455C8CEC48DF8E0E26538CAC0AF2B1BCBB0275E8E7E204B093D65E99344F82E2921BED">
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<16x3x3x8xi8>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<16x3x3x8xi8>) type=complete dim=2
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<16x3x3x8xi8>) type=complete dim=3
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<16x3x3x8xi8>) type=complete dim=4
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
          emithls.for %idx2 = 0 to 16 step 1 {
            %var_int32_0 = emithls.variable as i32 = 0
            emithls.for %idx3 = 0 to 1 step 1 {
              emithls.pragma.pipeline II=1 style=flp
              emithls.for %idx4 = 0 to 3 step 1 {
                emithls.for %idx5 = 0 to 3 step 1 {
                  emithls.for %idx6 = 0 to 8 step 1 {
                    %0 = emithls.array.read %var_array_0[%idx6, %idx4, %idx5] : !emithls.array<8x3x3xi8> -> i8
                    %1 = emithls.array.read %const_array_0[%idx2, %idx4, %idx5, %idx6] : !emithls.array<16x3x3x8xi8> -> i8
                    %2 = emithls.arith.cast %0 : i8 to i32
                    %3 = emithls.arith.sub %2, %const_int32_0 : i32
                    %4 = emithls.arith.cast %1 : i8 to i32
                    %5 = emithls.arith.mul %3, %4 : i32
                    emithls.arith.fused add, %var_int32_0, %5 : i32
                  }
                }
              }
            }
            emithls.stream.write %var_int32_0 to %arg1[%idx2] : i32 -> !emithls.array<16x!emithls.stream<i32>>
          }
        }
      }
    }
  }
  emithls.func @main_node_3(%arg0: !emithls.array<16x!emithls.stream<i32>>, %arg1: !emithls.array<16x!emithls.stream<i8>>) {
    emithls.pragma.inline off
    %const_array_0 = emithls.variable as const !emithls.array<16xi32> = dense<[2047621329, 1985519193, 2031086687, 2027732936, 2009713599, 2007728853, 2045591835, 2012502904, 2022136137, 2028641361, 1994345684, 2001504669, 2014968057, 1993305016, 2039619951, 2032920635]>
    emithls.pragma.bind_storage variable=%const_array_0(!emithls.array<16xi32>) type=rom_1p impl=lutram
    emithls.pragma.array_partition variable=%const_array_0(!emithls.array<16xi32>) type=complete dim=1
    %const_int32_0 = emithls.variable as const i32 = 127
    %const_int32_1 = emithls.variable as const i32 = -128
    %const_int64_0 = emithls.variable as const i64 = 40
    %const_int64_1 = emithls.variable as const i64 = -1073741824
    %const_int64_2 = emithls.variable as const i64 = 1073741824
    %const_int32_2 = emithls.variable as const i32 = 0
    %const_int64_3 = emithls.variable as const i64 = 549755813888
    emithls.for %idx0 = 0 to 784 step 1 {
      emithls.pragma.pipeline II=1 style=flp
      emithls.for %idx1 = 0 to 16 step 1 {
        %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<16x!emithls.stream<i32>> -> i32
        %1 = emithls.array.read %const_array_0[%idx1] : !emithls.array<16xi32> -> i32
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
        emithls.stream.write %14 to %arg1[%idx1] : i8 -> !emithls.array<16x!emithls.stream<i8>>
      }
    }
  }
  emithls.func @main_top(%arg0: !emithls.ptr<i64>, %arg1: !emithls.ptr<i128>) {
    emithls.top_interface
    %var_array_0 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) depth=10
    %var_array_1 = emithls.variable as !emithls.array<16x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_1(!emithls.array<16x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_1(!emithls.array<16x!emithls.stream<i8>>) depth=2
    %var_array_2 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_2(!emithls.array<8x!emithls.stream<i32>>) depth=16
    %var_array_3 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
    emithls.pragma.bind_storage variable=%var_array_3(!emithls.array<8x!emithls.stream<i8>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_3(!emithls.array<8x!emithls.stream<i8>>) depth=25
    %var_array_4 = emithls.variable as !emithls.array<16x!emithls.stream<i32>>
    emithls.pragma.bind_storage variable=%var_array_4(!emithls.array<16x!emithls.stream<i32>>) type=fifo impl=srl
    emithls.pragma.stream variable=%var_array_4(!emithls.array<16x!emithls.stream<i32>>) depth=16
    emithls.pragma.dataflow {
      emithls.call @main_top_read_i8_0(%arg0, %var_array_0) : (!emithls.ptr<i64>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_0(%var_array_0, %var_array_2) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_1(%var_array_2, %var_array_3) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
      emithls.call @main_node_2(%var_array_3, %var_array_4) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<16x!emithls.stream<i32>>) -> ()
      emithls.call @main_node_3(%var_array_4, %var_array_1) : (!emithls.array<16x!emithls.stream<i32>>, !emithls.array<16x!emithls.stream<i8>>) -> ()
      emithls.call @main_top_write_i8_0(%var_array_1, %arg1) : (!emithls.array<16x!emithls.stream<i8>>, !emithls.ptr<i128>) -> ()
    }
  }
}

