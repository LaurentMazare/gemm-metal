[[kernel]]
void dot_product(
  constant float *inA [[buffer(0)]],
  constant float *inB [[buffer(1)]],
  device float *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}
