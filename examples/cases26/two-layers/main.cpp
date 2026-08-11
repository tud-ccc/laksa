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
void main_top_write_i8_0(hls::stream<ap_int<8>> arg0[16], ap_int<128> *arg1)
{
  #pragma HLS INLINE off
  for (size_t idx0 = 0; idx0 < 784; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    ap_int<128> var_tmp0 = 0;
    for (size_t idx1 = 0; idx1 < 16; idx1 += 1) {
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
void main_node_2(hls::stream<ap_int<8>> arg0[8], hls::stream<ap_int<32>> arg1[16])
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
  const ap_int<8> cst0[16][3][3][8] = {0, -68, 127, -51, -24, -113, 61, 15, 12, 126, 9, -113, -67, -73, -36, 113, -12, 71, 88, 83, 35, 32, -30, -88, 71, -93, -94, -117, 11, -4, -48, 101, -75, -14, 37, -91, -104, 126, 4, 109, 75, -101, -59, 23, -119, 3, 126, 55, 19, -48, 118, -86, -126, 37, 86, 95, 49, -16, -95, -39, -61, -76, 109, -109, -2, 75, 13, -36, 92, -57, 23, 73, 65, -17, 54, -42, 13, 123, 96, 117, 40, -122, 40, 97, 59, -12, 16, 105, -104, 103, 103, 85, 99, -99, 118, 118, -73, 127, 73, -104, 83, 26, -118, -50, -112, 123, -117, 12, 116, -5, 105, 2, 25, -8, 100, 113, 83, 29, 73, -2, 24, 38, -52, 47, 117, 94, -38, -71, -17, 98, -7, -56, 67, 38, 59, -65, -102, 73, -17, 34, 111, 4, -34, 2, -12, -119, 23, 108, -124, -91, 23, 103, -121, 125, 71, -66, 3, 19, -62, 18, -1, -124, -87, -111, 26, 74, -42, 93, 27, 72, 35, -88, 13, -85, -23, -17, -11, 48, 34, 34, 58, -89, 29, 71, -127, 104, -108, 107, 92, -36, 20, 126, 43, 6, -49, 125, 18, -102, -5, 62, 68, -87, -91, -68, 74, 70, -84, 91, 17, 57, 53, 114, 124, -7, -42, -32, 51, 18, 119, -10, 45, 58, 75, -56, -68, 27, -77, 80, 100, -1, -27, -42, -86, 30, 118, -98, -94, -114, -18, 58, -18, -25, 98, -78, 124, -111, 0, 57, -33, -63, -26, -127, 120, 54, 30, 26, 91, -119, 84, 55, 68, -81, 45, 65, -77, -86, 60, 103, 8, -56, -127, 109, 113, 66, -121, -43, 119, -76, -115, 73, -72, 20, 79, -59, -116, 54, -82, -66, -77, -110, -69, 37, 52, -102, 52, 124, 27, -57, 127, -85, -54, 116, -65, 37, -76, 51, -125, 52, -57, -103, 2, 116, 127, -57, -30, -68, 82, 63, -32, 104, 97, -14, -6, 17, 85, 85, -77, 14, 105, 44, -19, 42, 121, 74, -124, -53, 74, -81, 65, 104, 18, -52, 21, 99, 31, -48, -18, -51, 117, 41, 113, -64, 87, 10, 72, -44, -68, -80, 119, -82, 22, 87, 77, 52, -44, -50, -108, 67, -5, -36, 38, -31, -3, 122, -57, 113, -100, -20, 32, -112, -25, 45, 126, 126, 118, -97, 113, -44, -118, -115, 2, -22, 125, 97, -127, -53, -19, 6, -87, 65, 7, 48, 127, 76, -39, -127, 39, -63, -57, 103, 62, 50, -95, -103, 7, 112, -85, 10, -88, -118, -73, -108, -66, -63, -51, 126, -100, -114, 25, -122, -47, -79, 58, 76, -74, -56, 101, -47, -78, 16, 92, -127, -112, -81, 119, -23, -112, -84, 66, 94, 115, 6, -25, -67, 9, -69, 43, -15, 100, -39, 100, 5, 96, 16, 115, -42, 120, -64, 87, 98, 99, 63, 103, -29, 93, -46, -116, -84, -68, 45, -45, 81, 35, 91, 41, 121, 11, -47, 3, 59, -120, 18, 64, -93, 109, -79, 94, 10, -17, 122, 76, -15, 87, 52, -100, -118, 32, -12, -97, -27, -3, 51, 22, 119, -2, -45, 15, 78, -16, 106, -25, -7, 16, 65, -42, 106, -91, 107, 122, 92, 66, 70, -68, 82, 58, -74, -48, 101, 15, 38, -113, 89, 77, 19, 17, 13, 122, -26, 75, 101, 23, 24, -120, -85, -4, -12, -48, 127, -107, -29, -26, -89, -103, -92, -103, 46, -46, -50, 67, -83, -39, -82, -94, -2, -48, 65, -56, -82, -124, -55, 89, 10, -59, -87, 100, 6, 102, -125, -81, -44, -37, -117, 79, -46, -17, 33, 59, -53, 121, 15, 1, 108, -39, 99, -80, -30, -48, 28, -82, -26, -40, -107, 8, -91, 17, -2, 18, -85, 81, 121, 41, 120, 112, -125, 17, 23, 34, 39, 45, -88, 81, 44, -57, 55, 44, 37, -55, -96, 50, 20, -127, -24, -38, 88, 68, 66, -13, 85, -104, -33, 54, -12, 78, -93, -32, 47, -37, -61, 123, -66, 37, 23, 58, -85, 98, -24, -13, -33, 98, 3, -24, 112, -107, -90, -103, 87, -4, 90, -109, 89, 7, -73, 64, 85, -33, 111, -127, 49, 121, -102, -87, 60, -14, -37, 66, -73, -110, -80, 17, -61, -24, -9, 3, 32, 35, -45, -48, -68, -20, 70, 69, 21, -13, 16, 10, -67, 125, 123, -105, 118, 58, -41, 77, 43, 66, -66, 99, -89, 62, 49, 78, -121, -92, 127, 10, -17, 23, -1, 58, -77, 7, -39, 114, 82, 4, 10, -91, -102, 75, -11, -74, -2, 18, 113, -58, 50, 7, 33, 85, 100, -65, 75, -73, 37, 64, -60, 25, 76, 51, 38, 34, -34, -22, 99, -49, -111, -93, 47, -119, 36, 93, -10, 72, -2, 111, -49, -119, -107, 9, 111, 102, 51, -27, 102, -54, -13, 104, -21, 102, -86, 46, 20, 105, 81, 114, -36, -74, -120, 88, -12, 35, -64, 108, 27, -90, -103, 76, 22, 101, 120, -29, -85, -106, 61, 35, -117, 3, -110, -40, -17, 60, -90, 63, 51, 122, -63, 31, 123, 106, 19, 42, 89, 125, -55, -32, -99, 81, 22, -118, 32, 127, 112, 38, 2, 94, 31, 120, -102, -69, 53, -70, 63, 65, -6, 101, 40, -98, 65, 15, -30, 42, -101, -51, -71, 98, 68, -89, -33, -110, 88, -70, -68, -40, 105, 58, -91, -124, 17, -49, -67, -121, -68, -39, -23, 127, 122, -93, -3, -6, 33, 51, 119, -39, -86, -97, -26, -12, 81, -14, 74, -29, 26, 65, 121, -88, 102, 119, -63, 29, -99, -8, -58, 23, -47, 125, 36, 0, -43, 41, 3, -116, 55, -28, 11, -9, 83, 26, -83, 81, 32, -82, -3, 86, -57, 76, 61, 9, 102, 73, -118, 5, 49, -122, -100, -75, -127, 14, 98, -116, 104, 3, 70, 46, -15, -116, -48, 69, 110, -125, 5, -99, -126, -17, 64, 15, 89, -68, -12, 18, -106, 60, 70, -73, 16, 34, -82, 87, -47, 49, -50, 44, 49, 79, 119, -28, 80, 13, -120, -35, 38, -70, 98, 99, -126, -14, -17, 117, 87, 45, -35, -74, -20, 117, 23, -50, 35, 119, -97, -12, 126, 44, 124, -25, 102, -120, 78, 104, 54, 97, -80, 106, 101, 37, 112, -27, -118, 7, 1, 68, 56, -47, 120, 49, -37, -115, 20, -46, 64, 72, -23, -127, 16, 91, -42, -91, -61, -64, 24, -62, 126, -13, -102, 68, -105, 111, 111, -35, 46, 81, 92, 126, -41, -86, 118, 39, -15, 56, 39, -11, 18, -11, 107, 75, -48, 68, -5, -98, 119, -83, -127, 83, -41, -34, -19, 67, 44, 78, -125, -70, -54, -87, 12, 40, -38, 118, -49, 26, -68, -63, -46, 80, 78, -15, -18, -115, 85, 77, 69, 92, -116, -20, 72, -33, -114, 14, 38, 83, -116, -84, 10, -14, -79, -68, -69, 2, 117, -24, -25, -30, 4, -80, -109, -42, 94, -103, 52, 79, -126, -30, -110, 27, -19};
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
        for (size_t idx2 = 0; idx2 < 16; idx2 += 1) {
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
void main_node_3(hls::stream<ap_int<32>> arg0[16], hls::stream<ap_int<8>> arg1[16])
{
  #pragma HLS INLINE off
  const ap_int<32> cst0[16] = {2047621329, 1985519193, 2031086687, 2027732936, 2009713599, 2007728853, 2045591835, 2012502904, 2022136137, 2028641361, 1994345684, 2001504669, 2014968057, 1993305016, 2039619951, 2032920635};
  #pragma HLS BIND_STORAGE variable=cst0 type=rom_1p impl=lutram
  #pragma HLS ARRAY_PARTITION variable=cst0 type=complete dim=1
  for (size_t idx0 = 0; idx0 < 784; idx0 += 1) {
    #pragma HLS PIPELINE II=1 style=flp
    for (size_t idx1 = 0; idx1 < 16; idx1 += 1) {
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
void main_top(ap_int<64> *arg0, ap_int<128> *arg1)
{
  #pragma HLS INTERFACE mode=m_axi port=arg0 offset=slave bundle=gmem_arg0
  #pragma HLS INTERFACE mode=s_axilite port=arg0 bundle=control
  #pragma HLS INTERFACE mode=m_axi port=arg1 offset=slave bundle=gmem_arg1
  #pragma HLS INTERFACE mode=s_axilite port=arg1 bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
  hls::stream<ap_int<8>> stream0[8];
  #pragma HLS BIND_STORAGE variable=stream0 type=fifo impl=srl
  #pragma HLS STREAM variable=stream0 depth=10
  hls::stream<ap_int<8>> stream1[16];
  #pragma HLS BIND_STORAGE variable=stream1 type=fifo impl=srl
  #pragma HLS STREAM variable=stream1 depth=2
  hls::stream<ap_int<32>> stream2[8];
  #pragma HLS BIND_STORAGE variable=stream2 type=fifo impl=srl
  #pragma HLS STREAM variable=stream2 depth=16
  hls::stream<ap_int<8>> stream3[8];
  #pragma HLS BIND_STORAGE variable=stream3 type=fifo impl=srl
  #pragma HLS STREAM variable=stream3 depth=25
  hls::stream<ap_int<32>> stream4[16];
  #pragma HLS BIND_STORAGE variable=stream4 type=fifo impl=srl
  #pragma HLS STREAM variable=stream4 depth=16

  #pragma HLS DATAFLOW
  main_top_read_i8_0(arg0, stream0);
  main_node_0(stream0, stream2);
  main_node_1(stream2, stream3);
  main_node_2(stream3, stream4);
  main_node_3(stream4, stream1);
  main_top_write_i8_0(stream1, arg1);

}

