// This is the output of command
// ladle input.mlir -p convert-to-emithls -t emithls-to-cpp -o main.cpp

#include "algorithm"
#include "ap_int.h"
#include "cstddef"
#include "hls_stream.h"
void main_top_read_i8_0(ap_int<64> *arg0, hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 1024; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    ap_int<64> elem_tmp0 = arg0[idx0];
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      arg1[idx1].write(elem_tmp0.range((idx1 + 1) * 8 - 1, idx1 * 8));
    }
  }
}
void main_top_write_i8_0(hls::stream<ap_int<8>> arg0[8], ap_int<64> *arg1)
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 784; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    ap_int<64> var_tmp0 = 0;
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<8> data_tmp0 = arg0[idx1].read();
      var_tmp0.range((idx1 + 1) * 8 - 1, idx1 * 8) = data_tmp0;
    }
    arg1[idx0] = var_tmp0;
  }
}
void main_node_0(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  ap_int<8> array0[8][3][3];
  #pragma HLS BIND_STORAGE variable=array0 type=ram_2p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=3
  ap_int<8> array1[8][2][32];
  #pragma HLS BIND_STORAGE variable=array1 type=ram_2p impl=bram
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=2
  const ap_int<8> cst0[8][3][3][8] = {-88, -31, -16, -39, -74, -43, -78, -41, -30, 8, 32, 81, -119, 65, -25, 30, 26, 85, -11, -74, -22, 76, -94, 83, -14, -34, 113, -91, -52, 114, 31, 103, -62, 4, 85, 52, -127, 77, -58, 82, 83, 92, -74, 3, -33, 73, 100, -109, -19, 77, -42, -51, 67, -105, 118, 90, -36, -71, 26, 48, 10, 76, -52, -20, -125, 26, -30, -100, 16, -101, -126, 115, 6, -35, 0, -59, 91, -90, -94, 0, 37, 40, -80, 35, 106, -69, -122, -6, -78, 116, -89, -60, 99, -58, 70, -127, 47, 40, -53, 5, -48, 8, -79, -31, 48, 75, 9, 33, -58, 95, 119, -94, 54, 110, -47, 20, 54, -47, -8, -44, 22, 31, 85, -124, -31, 50, 63, -7, 10, -48, 13, 121, -75, 66, -16, -56, 87, 93, 6, -3, 47, -73, -44, 48, 69, 66, -9, 9, -34, -61, -18, -54, -45, 94, 61, 87, 86, -66, -14, 78, 99, 17, 68, -120, 121, 127, 48, -33, 89, 26, -74, -95, 55, -92, 67, -115, -104, -20, -121, -15, -62, -87, -110, 70, 46, 100, -34, -122, -95, -83, -105, -10, -61, 112, 108, -19, 74, -9, 68, -57, -85, 109, -95, 89, -104, -45, -116, 19, -73, 9, 106, -30, 126, -14, -96, 35, -9, 42, -78, 47, 81, -82, 32, -121, -116, 103, 97, 63, 93, -60, -121, -89, 114, 31, -16, -46, 72, -11, 126, -67, 123, 2, -111, 105, 125, -37, 18, 91, 126, -127, 117, -44, -72, -40, 40, 49, 26, 0, 46, 40, 2, -108, 71, -31, 96, -33, -41, 125, 12, -9, 3, -26, 127, 16, 36, -7, 87, -87, -66, 77, -77, 103, -23, 37, -67, 42, 94, 115, -91, 7, 82, 2, 114, -36, -124, 58, 110, 9, 59, 68, -17, -48, 3, 66, -53, 55, 3, 70, -97, 98, -117, 121, -14, 98, 62, 72, -46, 41, 91, 103, -66, -35, -58, 29, -5, -76, 67, -57, 115, 64, -28, -91, 123, -90, 22, 28, -49, -62, 88, 58, 51, 37, 52, -84, 119, 56, -110, -100, 59, 119, -53, 67, -16, 18, 63, 75, 95, 34, 89, 127, 115, -103, -74, -105, -29, 52, -107, -52, 73, 5, -57, 32, -87, 94, 47, 100, -53, 59, 91, -30, 97, 17, 113, 56, 98, -74, -34, 25, 101, 15, 29, 17, -41, -65, -36, -105, 21, 127, 100, 81, -55, 91, 28, 54, -6, -8, -48, -59, -112, 53, 8, 95, 107, -81, -124, 58, 14, 36, 112, 118, 87, -15, -29, 4, 47, 90, 36, 3, -113, 86, -38, -18, 43, 28, 97, -97, 71, -63, -19, -65, 30, -49, 30, 59, -83, -6, 33, -31, -39, -7, -18, 33, -25, 77, 113, -93, 126, -42, -105, -14, -49, -2, -102, 108, -26, 69, 112, -98, 115, 38, 3, 62, -64, -107, -123, -26, -127, 91, 121, 93, 0, -61, 36, -95, -21, 22, 49, -17, 6, -107, 7, 63, -105, -48, -79, 125, 0, -30, 43, 107, 113, 57, 76, 55, 115, -29, 22, 21, 97, -38, -90, 63, 127, 75, -28, 85, -120, 2, 92, 68, -126, -14, -11, 84, 24, -19, -109, -124, -120, 20, 30, 75, 24, -109, 108, 18, -33, -95, -106, -107, -44, 1, -59, -33, 14, -74, 59, -79, -63, -96, 20, -70, 68, 47, -92, -126, -118, 104, -28, -87, -108, -5, 40, -117, 29, 36, 81, 72, -48, -110, 21, -21, 9, 121, 36, -28};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=3
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=4
  for (size_t idx0 = 0; idx0 < 32; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 32; idx1 += 1) {
      for (size_t idx2 = 0; idx2 < 1; idx2 += 1) {
        #pragma HLS PIPELINE II=1 style=flp
        for (size_t idx3 = 0; idx3 < 8; idx3 += 1) {
          ap_int<8> data_tmp0 = arg0[idx3].read();
          for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
            for (size_t idx5 = 0; idx5 < 2; idx5 += 1) {
              if (idx0 == 0) {
                array0[idx3][idx4][idx5] = 0;
              } else {
                array0[idx3][idx4][idx5] = array0[idx3][idx4][idx5 + 1];
              }
            }
          }
          if (idx0 >= 2) {
            array0[idx3][0][2] = array1[idx3][(idx0 - 2) % 2][idx1];
          } else {
            array0[idx3][0][2] = 0;
          }
          if (idx0 >= 1) {
            array0[idx3][1][2] = array1[idx3][(idx0 - 1) % 2][idx1];
          } else {
            array0[idx3][1][2] = 0;
          }
          array0[idx3][2][2] = data_tmp0;
          array1[idx3][idx0 % 2][idx1] = data_tmp0;
        }
      }
      if (idx0 >= 2 && idx1 >= 2) {
        for (size_t idx2 = 0; idx2 < 1; idx2 += 1) {
          #pragma HLS PIPELINE II=1 style=flp
          for (size_t idx3 = 0; idx3 < 8; idx3 += 1) {
            ap_int<32> var_tmp0 = 0;
            for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
              for (size_t idx5 = 0; idx5 < 3; idx5 += 1) {
                for (size_t idx6 = 0; idx6 < 8; idx6 += 1) {
                  ap_int<8> elem_tmp0 = array0[idx6][idx4][idx5];
                  ap_int<8> elem_tmp1 = cst0[idx3][idx4][idx5][idx6];
                  ap_int<32> cast_tmp0 = (ap_int<32>)elem_tmp0;
                  ap_int<32> diff_tmp0 = cast_tmp0 - (-128);
                  ap_int<32> cast_tmp1 = (ap_int<32>)elem_tmp1;
                  ap_int<32> prod_tmp0 = diff_tmp0 * cast_tmp1;
                  var_tmp0 += prod_tmp0;
                }
              }
            }
            arg1[idx3].write(var_tmp0);
          }
        }
      }
    }
  }
}
void main_node_1(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<8>> arg1[8], hls::stream<ap_int<8>> arg2[8])
{
  #pragma HLS INLINE off
  const ap_int<32> cst0[8] = {1523322938, 1544671534, 1547120170, 1546480699, 1536582660, 1557469766, 1525058969, 1526555847};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=1
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<32> elem_tmp0 = cst0[idx1];
      ap_int<64> cast_tmp0 = (ap_int<64>)data_tmp0;
      ap_int<64> cast_tmp1 = (ap_int<64>)elem_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp0 * cast_tmp1;
      ap_int<64> sum_tmp0 = prod_tmp0 + 1099511627776;
      bool cmp_tmp0 = data_tmp0 >= 0;
      ap_int<64> mux_tmp0 = cmp_tmp0 ? 1073741824 : (-1073741824);
      ap_int<64> sum_tmp1 = mux_tmp0 + sum_tmp0;
      ap_int<64> shr_tmp0 = sum_tmp1 >> 41;
      ap_int<32> cast_tmp2 = (ap_int<32>)shr_tmp0;
      ap_int<32> sum_tmp2 = cast_tmp2 + (-128);
      ap_int<32> max_tmp0 = std::max(sum_tmp2, (ap_int<32>)(-128));
      ap_int<32> min_tmp0 = std::min(max_tmp0, (ap_int<32>)127);
      ap_int<8> cast_tmp3 = (ap_int<8>)min_tmp0;
      arg1[idx1].write(cast_tmp3);
      arg2[idx1].write(cast_tmp3);
    }
  }
}
void main_node_2(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 32; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 32; idx1 += 1) {
      #pragma HLS PIPELINE II=1 style=flp
      if (idx0 >= 1 && idx0 <= 30 && idx1 >= 1 && idx1 <= 30) {
        for (size_t idx2 = 0; idx2 < 8; idx2 += 1) {
          ap_int<8> data_tmp0 = arg0[idx2].read();
          arg1[idx2].write(data_tmp0);
        }
      } else {
        for (size_t idx2 = 0; idx2 < 8; idx2 += 1) {
          arg1[idx2].write((-128));
        }
      }
    }
  }
}
void main_node_3(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  ap_int<8> array0[8][3][3];
  #pragma HLS BIND_STORAGE variable=array0 type=ram_2p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=3
  ap_int<8> array1[8][2][32];
  #pragma HLS BIND_STORAGE variable=array1 type=ram_2p impl=bram
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=2
  const ap_int<8> cst0[8][3][3][8] = {23, 38, 124, -50, -6, 15, 83, 95, -56, 69, -33, 28, -65, 108, 117, 65, -35, 11, -15, 87, 25, 90, -99, 122, -116, -64, -41, 24, -31, 49, 5, -32, 54, -116, 48, -108, 82, -100, 27, 77, 46, 17, -18, -105, -16, -55, 115, 49, -60, 85, -20, 106, -124, -1, -4, -54, 61, 84, -127, 50, -22, -113, -62, 119, -80, -70, -41, 65, 47, -43, -53, -63, -35, 39, -81, 89, 18, 121, 43, 83, -102, 49, 8, 60, 103, 18, -33, 84, -8, -113, -20, -57, 91, -3, -68, -106, 25, -71, 16, -52, 3, -53, 28, -3, 113, 94, -96, 44, 8, 74, -67, -43, 37, -84, -9, 13, -57, -84, -16, 85, 115, -94, 127, -69, -73, -40, 6, -38, 80, -112, 16, -17, 68, -55, 110, 64, -86, -81, 73, -18, 100, -85, -7, 2, 72, -68, 2, 90, -5, -66, 82, -13, 19, -84, -17, 11, 37, 56, -94, 0, 106, 18, -77, -82, 12, 59, 39, -33, 75, -32, -68, -71, -27, 26, -120, 86, -123, 14, -84, 52, -43, 23, -126, 81, 113, 71, -121, 98, 74, -35, 44, 40, -12, -16, -102, 14, -23, 125, 17, 46, 123, 48, -10, 86, 99, 23, 71, 98, -79, -28, 88, 84, -23, -89, -127, 70, -3, 32, 87, 103, -31, -4, -106, -20, -112, 3, 34, 97, -127, 84, -42, -100, 13, -103, 58, 111, 4, -73, 50, 56, 86, 109, 12, -24, -38, 81, -65, 48, -108, 96, 101, 39, -7, -74, -76, -4, 23, -39, 106, 45, 35, -17, -60, -121, 48, 20, 65, -52, 45, -22, 23, -26, 12, 36, -67, -93, -108, -39, -62, -39, 103, 76, -27, 9, 123, -14, 12, 125, 114, -13, -17, -15, -1, 109, -108, 113, -27, 116, 47, 3, -87, -100, 47, 126, 120, -44, 121, 108, -97, -75, -62, -45, 53, -65, -127, 5, 105, 105, 117, 48, 0, 105, 26, 116, 17, 22, -8, -57, -91, -124, 74, 10, 91, 69, -102, -122, 45, -81, -72, 75, 81, 10, 82, 45, 1, 43, 110, -8, -60, -112, -59, -95, 86, -2, 110, -29, -5, 117, 39, -54, 109, -38, -67, -106, -125, 45, 119, 125, -30, 80, -113, 32, -127, -65, -99, -78, -15, 55, 114, -64, 51, 4, 125, -3, -99, -67, -114, -56, -109, -17, 117, 28, 75, 36, 5, -116, 70, -102, -112, 99, -115, 95, -14, -3, 38, 20, 42, -82, -28, -40, 8, 96, 24, 19, 102, 36, 64, -5, -121, -36, 15, 57, 90, 85, 2, -89, 21, -18, 107, -123, 82, 8, -75, -46, -115, 94, 3, -26, 115, 7, -68, -17, 37, -89, -10, 113, 17, -70, -39, 38, 118, 83, -108, -53, 17, 23, -25, -77, 69, -89, 32, 36, -127, 80, 5, 110, -115, -10, -59, -124, -56, -35, 29, -114, 54, -40, 48, 25, 70, -127, 113, 0, 69, 25, -53, 42, -77, 75, -49, 11, -44, -20, -115, -125, 59, 80, 70, -63, 97, 59, 25, -125, -38, -29, -122, 123, -65, -83, 8, 50, -86, -110, 22, -45, -105, -53, 20, 74, 54, -100, -82, -56, 19, -81, 40, 18, 34, 92, 125, 127, 20, 30, -32, -124, -126, 52, -88, 106, -79, 19, 100, -114, -19, -70, 93, 126, 11, 7, 92, -74, 53, 54, 122, -19, -112, -120, -2, -67, -21, 108, -95, -127, -74, 115, 98, -105, -38, 39, -31, 110, -33, 95, -69, 48, -57, -101};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=3
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=4
  for (size_t idx0 = 0; idx0 < 32; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 32; idx1 += 1) {
      for (size_t idx2 = 0; idx2 < 1; idx2 += 1) {
        #pragma HLS PIPELINE II=1 style=flp
        for (size_t idx3 = 0; idx3 < 8; idx3 += 1) {
          ap_int<8> data_tmp0 = arg0[idx3].read();
          for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
            for (size_t idx5 = 0; idx5 < 2; idx5 += 1) {
              if (idx0 == 0) {
                array0[idx3][idx4][idx5] = 0;
              } else {
                array0[idx3][idx4][idx5] = array0[idx3][idx4][idx5 + 1];
              }
            }
          }
          if (idx0 >= 2) {
            array0[idx3][0][2] = array1[idx3][(idx0 - 2) % 2][idx1];
          } else {
            array0[idx3][0][2] = 0;
          }
          if (idx0 >= 1) {
            array0[idx3][1][2] = array1[idx3][(idx0 - 1) % 2][idx1];
          } else {
            array0[idx3][1][2] = 0;
          }
          array0[idx3][2][2] = data_tmp0;
          array1[idx3][idx0 % 2][idx1] = data_tmp0;
        }
      }
      if (idx0 >= 2 && idx1 >= 2) {
        for (size_t idx2 = 0; idx2 < 8; idx2 += 1) {
          ap_int<32> var_tmp0 = 0;
          for (size_t idx3 = 0; idx3 < 1; idx3 += 1) {
            #pragma HLS PIPELINE II=1 style=flp
            for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
              for (size_t idx5 = 0; idx5 < 3; idx5 += 1) {
                for (size_t idx6 = 0; idx6 < 8; idx6 += 1) {
                  ap_int<8> elem_tmp0 = array0[idx6][idx4][idx5];
                  ap_int<8> elem_tmp1 = cst0[idx2][idx4][idx5][idx6];
                  ap_int<32> cast_tmp0 = (ap_int<32>)elem_tmp0;
                  ap_int<32> diff_tmp0 = cast_tmp0 - (-128);
                  ap_int<32> cast_tmp1 = (ap_int<32>)elem_tmp1;
                  ap_int<32> prod_tmp0 = diff_tmp0 * cast_tmp1;
                  var_tmp0 += prod_tmp0;
                }
              }
            }
          }
          arg1[idx2].write(var_tmp0);
        }
      }
    }
  }
}
void main_node_4(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  const ap_int<32> cst0[8] = {1241604906, 1329922038, 1341576575, 1322904542, 1326948123, 1339807398, 1325877483, 1340943676};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=1
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<32> elem_tmp0 = cst0[idx1];
      ap_int<64> cast_tmp0 = (ap_int<64>)data_tmp0;
      ap_int<64> cast_tmp1 = (ap_int<64>)elem_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp0 * cast_tmp1;
      ap_int<64> sum_tmp0 = prod_tmp0 + 549755813888;
      bool cmp_tmp0 = data_tmp0 >= 0;
      ap_int<64> mux_tmp0 = cmp_tmp0 ? 1073741824 : (-1073741824);
      ap_int<64> sum_tmp1 = mux_tmp0 + sum_tmp0;
      ap_int<64> shr_tmp0 = sum_tmp1 >> 40;
      ap_int<32> cast_tmp2 = (ap_int<32>)shr_tmp0;
      ap_int<32> sum_tmp2 = cast_tmp2 + (-128);
      ap_int<32> max_tmp0 = std::max(sum_tmp2, (ap_int<32>)(-128));
      ap_int<32> min_tmp0 = std::min(max_tmp0, (ap_int<32>)127);
      ap_int<8> cast_tmp3 = (ap_int<8>)min_tmp0;
      arg1[idx1].write(cast_tmp3);
    }
  }
}
void main_node_5(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<8> data_tmp0 = arg0[idx1].read();
      ap_int<32> cast_tmp0 = (ap_int<32>)data_tmp0;
      ap_int<32> diff_tmp0 = cast_tmp0 - (-128);
      ap_int<64> cast_tmp1 = (ap_int<64>)diff_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp1 * 1073741824;
      ap_int<64> sum_tmp0 = prod_tmp0 + 512;
      ap_int<64> shr_tmp0 = sum_tmp0 >> 10;
      ap_int<32> cast_tmp2 = (ap_int<32>)shr_tmp0;
      arg1[idx1].write(cast_tmp2);
    }
  }
}
void main_node_6(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<64> cast_tmp0 = (ap_int<64>)data_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp0 * 1633833687;
      ap_int<64> sum_tmp0 = prod_tmp0 + 2147483648;
      bool cmp_tmp0 = data_tmp0 >= 0;
      ap_int<64> mux_tmp0 = cmp_tmp0 ? 1073741824 : (-1073741824);
      ap_int<64> sum_tmp1 = mux_tmp0 + sum_tmp0;
      ap_int<64> shr_tmp0 = sum_tmp1 >> 32;
      ap_int<32> cast_tmp1 = (ap_int<32>)shr_tmp0;
      arg1[idx1].write(cast_tmp1);
    }
  }
}
void main_node_7(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<8> data_tmp0 = arg0[idx1].read();
      ap_int<32> cast_tmp0 = (ap_int<32>)data_tmp0;
      ap_int<32> diff_tmp0 = cast_tmp0 - (-128);
      ap_int<64> cast_tmp1 = (ap_int<64>)diff_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp1 * 1073741824;
      ap_int<64> sum_tmp0 = prod_tmp0 + 1024;
      ap_int<64> shr_tmp0 = sum_tmp0 >> 11;
      ap_int<32> cast_tmp2 = (ap_int<32>)shr_tmp0;
      arg1[idx1].write(cast_tmp2);
    }
  }
}
void main_node_8(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<32>> arg1[8], hls::stream<ap_int<32>> arg2[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<32> data_tmp1 = arg1[idx1].read();
      ap_int<32> sum_tmp0 = data_tmp0 + data_tmp1;
      arg2[idx1].write(sum_tmp0);
    }
  }
}
void main_node_9(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<64> cast_tmp0 = (ap_int<64>)data_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp0 * 1073741824;
      ap_int<64> sum_tmp0 = prod_tmp0 + 281474976710656;
      bool cmp_tmp0 = data_tmp0 >= 0;
      ap_int<64> mux_tmp0 = cmp_tmp0 ? 1073741824 : (-1073741824);
      ap_int<64> sum_tmp1 = mux_tmp0 + sum_tmp0;
      ap_int<64> shr_tmp0 = sum_tmp1 >> 49;
      ap_int<32> cast_tmp1 = (ap_int<32>)shr_tmp0;
      ap_int<32> sum_tmp2 = cast_tmp1 + (-128);
      ap_int<32> max_tmp0 = std::max(sum_tmp2, (ap_int<32>)(-128));
      ap_int<32> min_tmp0 = std::min(max_tmp0, (ap_int<32>)127);
      ap_int<8> cast_tmp2 = (ap_int<8>)min_tmp0;
      arg1[idx1].write(cast_tmp2);
    }
  }
}
void main_node_10(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[8])
{
  #pragma HLS INLINE off
  ap_int<8> array0[8][3][3];
  #pragma HLS BIND_STORAGE variable=array0 type=ram_2p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=array0 type=complete dim=3
  ap_int<8> array1[8][2][30];
  #pragma HLS BIND_STORAGE variable=array1 type=ram_2p impl=bram
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=1
  #pragma HLS ARRAY_PARTITION variable=array1 type=complete dim=2
  const ap_int<8> cst0[8][3][3][8] = {-97, -97, 59, -54, -12, 55, 112, -103, -28, 41, 37, 24, -124, -42, 30, -95, -116, -127, -103, -71, 6, 75, -99, 49, 122, 127, -106, -91, -25, -5, 68, 23, 30, -57, -23, 54, 36, 113, 57, 18, 54, 58, -114, -49, 17, 35, -104, -22, -83, 97, -12, -116, -91, -37, 99, -46, -96, -25, -123, 22, 124, 44, 10, 83, -125, 57, 44, -93, -122, 123, -50, -89, 108, -126, 117, -115, -35, 23, -11, -100, -95, -43, -25, 49, -100, -71, 11, 97, -19, 75, -110, -48, -95, -38, -24, 10, 53, 116, 16, 37, -50, -13, 19, 18, -2, -99, 125, 112, 46, 123, -39, -96, 60, -90, -101, 107, -6, 54, 66, 27, -47, -3, -127, 66, -119, -124, 99, 74, -91, -60, 49, 3, -24, 74, 25, -63, -43, 85, -56, -96, 24, 59, 71, -8, 79, -7, -9, -65, 43, 31, 96, -71, 15, 1, 25, 18, -15, -14, -115, -5, -109, -84, -11, 49, -42, -82, -23, 94, 118, 55, -92, -119, -37, -15, 112, -52, -70, 15, 34, -47, -107, 16, 116, 29, 124, -18, -4, -24, 20, 97, 56, 92, 37, -15, 42, -110, 104, -86, -48, -65, 4, 108, 52, 97, -72, 20, -106, -17, -127, -117, 7, -84, 62, 15, -20, -73, -45, -30, 37, -46, -75, -31, -87, 21, 65, 68, -18, -96, 66, -18, 103, 13, 109, 19, -11, 115, 39, 106, 29, -33, 29, 29, -14, 109, 83, 119, -13, -50, -42, -97, -35, 19, 125, -80, 89, 9, 23, -47, 61, -121, 49, 80, -34, 65, 31, -20, 127, -28, 118, 11, -95, -11, -93, -107, -56, -70, 19, -11, 87, 7, -102, 11, 77, 12, -100, -58, 85, -111, 115, -14, 33, -45, -110, -14, -114, 9, 71, -127, -80, -75, -37, -99, -94, -40, -124, -124, 103, 10, 105, -73, -107, 71, 44, -32, 46, -110, 74, -25, -7, 6, -107, -67, -52, 108, -9, 87, -18, 56, -50, -80, 102, 121, 16, -4, 117, 44, 29, -107, -31, 31, 56, -22, -49, -125, -50, -85, 63, 76, 73, -62, -71, 27, 57, 47, 35, 113, 29, -103, 125, 84, -21, 109, 51, 70, -105, 20, -47, 80, -4, -46, 118, 120, 112, -19, 0, -5, -93, -14, 46, -26, -22, 38, -93, 111, 119, -53, -102, 85, -65, 118, 76, -53, -77, 7, 55, 74, 53, 47, 70, -98, 91, -59, -68, -36, 22, 81, 73, -33, 78, 39, -65, -30, -110, -123, -72, -34, -53, -12, -96, 40, 40, -127, 6, -11, 0, -40, 105, -53, 109, -115, 94, 12, 110, 56, -22, 29, 94, 4, -60, 38, -64, -103, 77, 84, 18, 101, -48, -37, 97, -32, -57, 109, -4, 7, -61, -3, -67, -2, 101, -71, -100, 71, 105, -24, 105, 72, -109, 32, 3, -88, 83, -73, 77, 110, -44, 39, 53, -36, 67, 31, -36, 84, 72, -94, -87, 113, 5, 124, 116, -41, -20, 22, 127, 48, 75, 13, -20, -28, 96, 120, 93, -32, 112, -61, 107, -67, 22, 1, -20, 32, -72, -116, 17, -53, -109, 93, -30, -45, 55, -42, -39, -108, 60, 74, 31, -65, 38, 44, -127, 72, 77, 125, -126, 112, 33, -102, -80, 100, -27, -92, 90, 46, -52, 120, -17, 14, 104, 58, 62, -105, 88, 93, -121, -30, 57, 113, 73, -95, 93, -57, -40, 89, -105, -81, 94, 70, -108, 41, -118, -66, -122, 7, 85, -109, -100, -33};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=2
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=3
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=4
  for (size_t idx0 = 0; idx0 < 30; idx0 += 1) {
    for (size_t idx1 = 0; idx1 < 30; idx1 += 1) {
      for (size_t idx2 = 0; idx2 < 1; idx2 += 1) {
        #pragma HLS PIPELINE II=1 style=flp
        for (size_t idx3 = 0; idx3 < 8; idx3 += 1) {
          ap_int<8> data_tmp0 = arg0[idx3].read();
          for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
            for (size_t idx5 = 0; idx5 < 2; idx5 += 1) {
              if (idx0 == 0) {
                array0[idx3][idx4][idx5] = 0;
              } else {
                array0[idx3][idx4][idx5] = array0[idx3][idx4][idx5 + 1];
              }
            }
          }
          if (idx0 >= 2) {
            array0[idx3][0][2] = array1[idx3][(idx0 - 2) % 2][idx1];
          } else {
            array0[idx3][0][2] = 0;
          }
          if (idx0 >= 1) {
            array0[idx3][1][2] = array1[idx3][(idx0 - 1) % 2][idx1];
          } else {
            array0[idx3][1][2] = 0;
          }
          array0[idx3][2][2] = data_tmp0;
          array1[idx3][idx0 % 2][idx1] = data_tmp0;
        }
      }
      if (idx0 >= 2 && idx1 >= 2) {
        for (size_t idx2 = 0; idx2 < 8; idx2 += 1) {
          ap_int<32> var_tmp0 = 0;
          for (size_t idx3 = 0; idx3 < 1; idx3 += 1) {
            #pragma HLS PIPELINE II=1 style=flp
            for (size_t idx4 = 0; idx4 < 3; idx4 += 1) {
              for (size_t idx5 = 0; idx5 < 3; idx5 += 1) {
                for (size_t idx6 = 0; idx6 < 8; idx6 += 1) {
                  ap_int<8> elem_tmp0 = array0[idx6][idx4][idx5];
                  ap_int<8> elem_tmp1 = cst0[idx2][idx4][idx5][idx6];
                  ap_int<32> cast_tmp0 = (ap_int<32>)elem_tmp0;
                  ap_int<32> diff_tmp0 = cast_tmp0 - (-128);
                  ap_int<32> cast_tmp1 = (ap_int<32>)elem_tmp1;
                  ap_int<32> prod_tmp0 = diff_tmp0 * cast_tmp1;
                  var_tmp0 += prod_tmp0;
                }
              }
            }
          }
          arg1[idx2].write(var_tmp0);
        }
      }
    }
  }
}
void main_node_11(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  const ap_int<8> cst0[8] = {39, 39, 39, 40, 40, 39, 40, 40};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=1
  const ap_int<32> cst1[8] = {1079388899, 1080650220, 1080683627, 2123794270, 2107169143, 1079132406, 2033284180, 2112265834};
  #pragma HLS BIND_STORAGE variable=cst1 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst1 type=complete dim=1
  for (size_t idx0 = 0; idx0 < 784; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 8; idx1 += 1) {
      ap_int<32> data_tmp0 = arg0[idx1].read();
      ap_int<32> elem_tmp0 = cst1[idx1];
      ap_int<8> elem_tmp1 = cst0[idx1];
      ap_int<32> cast_tmp0 = (ap_int<32>)elem_tmp1;
      ap_int<64> cast_tmp1 = (ap_int<64>)data_tmp0;
      ap_int<64> cast_tmp2 = (ap_int<64>)elem_tmp0;
      ap_int<64> prod_tmp0 = cast_tmp1 * cast_tmp2;
      ap_int<64> cast_tmp3 = (ap_int<64>)elem_tmp1;
      ap_int<64> shl_tmp0 = 1 << cast_tmp3;
      ap_int<64> shr_tmp0 = shl_tmp0 >> 1;
      ap_int<64> sum_tmp0 = prod_tmp0 + shr_tmp0;
      bool cmp_tmp0 = data_tmp0 >= 0;
      ap_int<64> mux_tmp0 = cmp_tmp0 ? 1073741824 : (-1073741824);
      ap_int<64> sum_tmp1 = mux_tmp0 + sum_tmp0;
      bool cmp_tmp1 = cast_tmp0 > 31;
      ap_int<64> mux_tmp1 = cmp_tmp1 ? sum_tmp1 : sum_tmp0;
      ap_int<64> shr_tmp1 = mux_tmp1 >> cast_tmp3;
      ap_int<32> cast_tmp4 = (ap_int<32>)shr_tmp1;
      ap_int<32> sum_tmp2 = cast_tmp4 + (-128);
      ap_int<32> max_tmp0 = std::max(sum_tmp2, (ap_int<32>)(-128));
      ap_int<32> min_tmp0 = std::min(max_tmp0, (ap_int<32>)127);
      ap_int<8> cast_tmp5 = (ap_int<8>)min_tmp0;
      arg1[idx1].write(cast_tmp5);
    }
  }
}
void main_top(ap_int<64> *arg0, ap_int<64> *arg1)
{
  #pragma HLS INTERFACE mode=m_axi port=arg0 offset=slave bundle=gmem_arg0
  #pragma HLS INTERFACE mode=s_axilite port=arg0 bundle=control
  #pragma HLS INTERFACE mode=m_axi port=arg1 offset=slave bundle=gmem_arg1
  #pragma HLS INTERFACE mode=s_axilite port=arg1 bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  hls::stream<ap_int<8>> stream0[8];
  #pragma HLS BIND_STORAGE variable=stream0 type=fifo impl=srl
  #pragma HLS STREAM variable=stream0 depth=10
  hls::stream<ap_int<8>> stream1[8];
  #pragma HLS BIND_STORAGE variable=stream1 type=fifo impl=srl
  #pragma HLS STREAM variable=stream1 depth=2
  hls::stream<ap_int<32>> stream2[8];
  #pragma HLS BIND_STORAGE variable=stream2 type=fifo impl=srl
  #pragma HLS STREAM variable=stream2 depth=17
  hls::stream<ap_int<8>> stream3[8];
  #pragma HLS BIND_STORAGE variable=stream3 type=fifo impl=srl
  #pragma HLS STREAM variable=stream3 depth=2
  hls::stream<ap_int<8>> stream4[8];
  #pragma HLS BIND_STORAGE variable=stream4 type=fifo impl=srl
  #pragma HLS STREAM variable=stream4 depth=9
  hls::stream<ap_int<8>> stream5[8];
  #pragma HLS BIND_STORAGE variable=stream5 type=fifo impl=srl
  #pragma HLS STREAM variable=stream5 depth=17
  hls::stream<ap_int<32>> stream6[8];
  #pragma HLS BIND_STORAGE variable=stream6 type=fifo impl=srl
  #pragma HLS STREAM variable=stream6 depth=16
  hls::stream<ap_int<8>> stream7[8];
  #pragma HLS BIND_STORAGE variable=stream7 type=fifo impl=srl
  #pragma HLS STREAM variable=stream7 depth=9
  hls::stream<ap_int<32>> stream8[8];
  #pragma HLS BIND_STORAGE variable=stream8 type=fifo impl=srl
  #pragma HLS STREAM variable=stream8 depth=10
  hls::stream<ap_int<32>> stream9[8];
  #pragma HLS BIND_STORAGE variable=stream9 type=fifo impl=srl
  #pragma HLS STREAM variable=stream9 depth=27
  hls::stream<ap_int<32>> stream10[8];
  #pragma HLS BIND_STORAGE variable=stream10 type=fifo impl=srl
  #pragma HLS STREAM variable=stream10 depth=2
  hls::stream<ap_int<32>> stream11[8];
  #pragma HLS BIND_STORAGE variable=stream11 type=fifo impl=srl
  #pragma HLS STREAM variable=stream11 depth=14
  hls::stream<ap_int<8>> stream12[8];
  #pragma HLS BIND_STORAGE variable=stream12 type=fifo impl=srl
  #pragma HLS STREAM variable=stream12 depth=17
  hls::stream<ap_int<32>> stream13[8];
  #pragma HLS BIND_STORAGE variable=stream13 type=fifo impl=srl
  #pragma HLS STREAM variable=stream13 depth=23

  #pragma HLS DATAFLOW
  main_top_read_i8_0(arg0, stream0);
  main_node_0(stream0, stream2);
  main_node_1(stream2, stream3, stream4);
  main_node_2(stream3, stream5);
  main_node_3(stream5, stream6);
  main_node_4(stream6, stream7);
  main_node_5(stream4, stream8);
  main_node_6(stream8, stream9);
  main_node_7(stream7, stream10);
  main_node_8(stream9, stream10, stream11);
  main_node_9(stream11, stream12);
  main_node_10(stream12, stream13);
  main_node_11(stream13, stream1);
  main_top_write_i8_0(stream1, arg1);

}

