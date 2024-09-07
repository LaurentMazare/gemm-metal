[[kernel]]
void dot_product(
  device const float *a [[buffer(0)]],
  device const float *b [[buffer(1)]],
  device float *c [[buffer(2)]],
  uint3 tpig[[thread_position_in_grid]])
{
  int index = tpig.x;
  c[index] = a[index] * b[index];
}
