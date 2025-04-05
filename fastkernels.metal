// linear layer kernels
// fused matmul + add bias kernel
kernel void matmul_bias(
    device float* x,     // [B x Din]
    device float* W,     // [Din x Dout]
    device float* b,     // [Dout]
    device float* y,     // [B x Dout]
    uint Din, uint Dout, uint B)
{

}

// cannot be in place. we need to store a bitmask of >0 <0 so we can .backwards() 
kernel void relu(
  
) {

}
// end linear layer kernels

// loss function

// softmax + cross entropy loss kernel




// backward softmax + cross entropy loss kernel

// end loss function


// optimizer / weight updates 




// end optimizer / weight updates
