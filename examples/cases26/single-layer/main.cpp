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
  for (size_t idx0 = 0; idx0 < 900; idx0 += 1) {
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
  const ap_int<8> cst0[8][3][3][8] = {58, -105, 27, 120, 99, -33, -82, -82, 110, -97, -89, 66, -90, -39, -85, 48, 27, 5, 42, 59, -15, 65, 63, -74, -20, 78, 40, -17, -58, 15, -12, -71, 101, 14, 23, 85, -124, 21, 83, -105, 14, 79, -127, 77, 82, -53, -52, -67, 67, -62, 78, -60, 115, -39, -114, 39, 121, -50, 72, -64, -11, 76, 49, -92, -127, 117, -81, 63, 83, -100, 50, -61, 53, 113, 68, -23, -115, -123, 32, 10, 50, 47, -44, 37, 111, 85, 89, -71, 121, -37, 19, 8, 24, 127, -117, -115, -80, -126, 17, 121, -117, -1, 64, 61, -39, -85, -46, -32, 110, -10, -43, -39, -79, -14, -115, 44, -4, -53, 35, 111, -127, -86, 14, -63, -32, 61, 0, -58, -18, 122, -60, 78, 0, -39, 102, 101, 14, 91, 107, -34, -76, -104, 27, 93, -127, -11, 117, -115, 63, -75, -11, 118, -120, 80, -50, 69, 54, -115, 100, 29, 5, -50, 9, -20, 116, -25, -101, -64, 102, 100, -28, 59, 2, -64, -20, -24, 117, -39, 5, -17, -22, -109, 18, 102, -127, -107, -87, 40, 23, 64, 40, -64, 120, 87, 112, 108, -10, 8, -19, 39, -117, -59, 52, 30, -89, -73, -40, -124, 90, -7, -119, 23, 109, 71, 117, 15, -112, 5, 76, -64, 56, 46, -13, -66, 16, 126, 51, -127, 41, 65, -86, 87, -83, 64, -78, 93, 114, 55, 5, -122, 41, -110, 85, -39, 26, -33, 68, -106, 31, 104, 87, -96, 33, 50, -110, -103, 32, 112, 52, 46, 29, -9, -99, -60, 5, 46, 44, -70, -22, 93, 89, 106, 2, -111, -118, 25, -53, -100, -102, -96, -12, 110, -14, 103, -23, 1, -5, 91, -16, 121, -79, -4, -28, 1, 91, 127, -75, 20, 45, 81, -121, 0, -105, 5, -7, 85, -3, -73, -34, -112, -2, -33, 84, -126, -9, 90, -83, -48, 85, -44, 100, 123, -62, -80, 104, 42, 100, 32, -106, 15, 86, -98, -33, 13, 101, -124, 60, -101, -67, -18, 17, 30, 26, -29, -43, 109, 81, -71, -9, -83, 124, -38, 15, -80, -20, 104, -19, 35, -95, -22, 117, -127, 2, 17, 113, 42, 64, -59, -24, 97, -108, 112, 29, -19, 77, 7, -97, -117, 11, -3, -52, 2, -72, 73, 123, -60, -73, 104, 66, 127, -90, 102, 102, -101, -23, 21, -92, -10, -24, 68, -12, -97, 109, 116, 59, 74, -35, -34, 107, -89, 97, 119, 40, 82, -115, -114, -99, 50, -102, 105, -67, 15, -59, 65, -44, -115, -127, 33, -2, 37, -27, 18, 21, -117, 82, -103, -44, -5, 28, 62, -37, 111, 19, -127, -126, -30, -90, 125, -35, 114, 119, -125, 26, -93, -92, -36, -113, -70, -111, 61, 60, 18, -101, -92, -95, -111, -59, -109, -123, 95, -52, -34, -42, -96, -47, 9, 64, 120, -22, -86, -63, -105, 100, 29, 84, 19, -35, 40, -109, -121, 81, 123, -114, -114, -72, 81, -72, -36, 10, -68, -54, 112, -89, -89, -1, -100, -127, -26, 125, 51, -90, 102, 93, -118, 23, 76, -120, -30, 74, -11, -65, 106, 25, 8, 52, -98, -127, -123, -79, 37, 82, 45, -92, -69, 79, 33, -116, -124, 45, 123, -55, 4, -10, -29, 28, -18, 127, -84, -39, -28, 72, 36, -96, 75, -37, 111, 13, 58, 103, 52, -76, 28, 2, -80, 68, -123, -86, 86, 54, -122, -89, 102, 127, 9, -104, 99};
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
void main_node_1(hls::stream<ap_int<32>> arg0[8], hls::stream<ap_int<8>> arg1[8])
{
  #pragma HLS INLINE off
  const ap_int<32> cst0[8] = {1288418624, 1300707578, 1298230215, 1293868576, 1310478957, 1307418510, 1313803263, 1306561467};
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
void main_top(ap_int<64> *arg0, ap_int<64> *arg1)
{
  hls::stream<ap_int<8>> stream0[8];
  #pragma HLS BIND_STORAGE variable=stream0 type=fifo impl=srl
  #pragma HLS STREAM variable=stream0 depth=10
  hls::stream<ap_int<8>> stream1[8];
  #pragma HLS BIND_STORAGE variable=stream1 type=fifo impl=srl
  #pragma HLS STREAM variable=stream1 depth=2
  hls::stream<ap_int<32>> stream2[8];
  #pragma HLS BIND_STORAGE variable=stream2 type=fifo impl=srl
  #pragma HLS STREAM variable=stream2 depth=16

  #pragma HLS DATAFLOW
  main_top_read_i8_0(arg0, stream0);
  main_node_0(stream0, stream2);
  main_node_1(stream2, stream1);
  main_top_write_i8_0(stream1, arg1);

}

